"""Grounded answer generation over retrieved chunks."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings
from app.core.exceptions import GenerationError
from app.core.logging import get_logger
from app.models.query import QueryRequest, QueryResponse, Source
from app.models.search import SearchResult
from app.prompts import (
    SNIPPET_CHARS,
    build_system_prompt,
    build_user_prompt,
    format_context,
    new_fence_nonce,
)
from app.services.vector_store import VectorStoreService

log = get_logger(__name__)

#: A prepared prompt: the fenced context, the citations it maps to, and the nonce that
#: names the delimiters for this request.
type PreparedContext = tuple[str, list[Source], str]


def _sse(payload: dict[str, Any]) -> str:
    """Encode one server-sent event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _to_source(number: int, result: SearchResult) -> Source:
    """Build the client-facing citation for a retrieved chunk."""
    page = result.metadata.get("page")
    start_index = result.metadata.get("start_index", 0)
    return Source(
        source_num=number,
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        filename=str(result.metadata.get("filename", "unknown")),
        page=page if isinstance(page, int) and not isinstance(page, bool) else None,
        start_index=start_index if isinstance(start_index, int) else 0,
        score=result.score,
        snippet=result.text[:SNIPPET_CHARS],
    )


class GenerationService:
    """Turns a question plus retrieved context into a grounded, cited answer.

    The chat-model *factory* is injected rather than an instance, so tests exercise the
    same resolution path production uses and can assert which model was asked for.
    """

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStoreService,
        chat_model_factory: Callable[[str], BaseChatModel] | None = None,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._factory = chat_model_factory or self._build_chat_model
        # Built once per model name: each ChatOllama owns httpx connection pools that
        # nothing closes, so constructing one per request leaks them.
        self._models: dict[str, BaseChatModel] = {}
        # One GPU with one resident model — cap concurrent generations so a burst
        # queues instead of thrashing.
        self._slots = asyncio.Semaphore(settings.generation_concurrency)

    def _build_chat_model(self, model: str) -> BaseChatModel:
        """Construct a ChatOllama for ``model``.

        Imported lazily so the suite, which always injects a factory, never imports the
        Ollama client stack.
        """
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=str(self._settings.ollama_base_url),
            model=model,
            temperature=self._settings.generation_temperature,
            num_ctx=self._settings.generation_num_ctx,
            num_predict=self._settings.generation_num_predict,
            # `timeout` here is per-read, not a total budget — a steady token stream
            # never trips it. The wall-clock bound is asyncio.timeout below.
            client_kwargs={"timeout": self._settings.request_timeout_s},
        )

    def model_for(self, requested: str | None) -> tuple[BaseChatModel, str]:
        """Resolve the chat model for a request, honouring a validated override."""
        name = requested or self._settings.generation_model
        if name not in self._models:
            self._models[name] = self._factory(name)
        return self._models[name], name

    def retrieve_and_build(
        self, query: str, top_k: int, document_id: str | None = None
    ) -> PreparedContext:
        """Retrieve context for ``query`` and format it for the prompt.

        ``top_k`` counts parents: the store matches small children and returns the whole
        parent passages they belong to, so each ``[Source N]`` the model sees is a full
        passage rather than the fragment that happened to embed well. Citations therefore
        describe the parent's position in the document.

        Synchronous and blocking (it embeds the query); callers on the event loop must
        run it in a thread pool.
        """
        results = self._vector_store.search(query, top_k_parents=top_k, document_id=document_id)
        nonce = new_fence_nonce()
        context = format_context(results, nonce)
        sources = [_to_source(number, result) for number, result in enumerate(results, start=1)]
        return context, sources, nonce

    def _messages(
        self, context: str, question: str, nonce: str
    ) -> list[SystemMessage | HumanMessage]:
        return [
            SystemMessage(content=build_system_prompt(nonce)),
            HumanMessage(content=build_user_prompt(context, question)),
        ]

    async def generate(self, request: QueryRequest, prepared: PreparedContext) -> QueryResponse:
        """Produce a complete answer in one shot."""
        context, sources, nonce = prepared
        chat_model, model_name = self.model_for(request.model)
        try:
            async with self._slots:
                async with asyncio.timeout(self._settings.request_timeout_s):
                    message = await chat_model.ainvoke(
                        self._messages(context, request.query, nonce)
                    )
        except TimeoutError as exc:
            log.warning("generation.timeout", model=model_name)
            raise GenerationError("The language model took too long to respond.") from exc
        except Exception as exc:
            log.warning("generation.failed", error=type(exc).__name__, model=model_name)
            raise GenerationError("The language model could not be reached.") from exc

        return QueryResponse(
            query=request.query,
            answer=str(message.content),
            sources=sources,
            model=model_name,
        )

    async def astream(
        self,
        request: QueryRequest,
        prepared: PreparedContext,
        chat_model: BaseChatModel,
        model_name: str,
    ) -> AsyncIterator[str]:
        """Stream the answer as server-sent events.

        The model is resolved by the caller, before the response starts: a generator
        body does not run until the first ``__anext__``, which happens after the 200 and
        headers are sent, so anything that can fail must happen earlier or inside the
        try below. Everything here is in-band by necessity.
        """
        context, sources, nonce = prepared
        token_count = 0
        try:
            yield _sse({"type": "sources", "sources": [s.model_dump() for s in sources]})
            async with self._slots:
                async with asyncio.timeout(self._settings.request_timeout_s):
                    async for part in chat_model.astream(
                        self._messages(context, request.query, nonce)
                    ):
                        text = str(part.content)
                        if text:
                            token_count += 1
                            yield _sse({"type": "token", "text": text})
        except Exception as exc:
            # The 200 and headers are long gone, so a failure can only be reported
            # in-band. Send an explicit error event rather than truncating silently.
            log.warning(
                "generation.stream_failed",
                error=type(exc).__name__,
                model=model_name,
                token_count=token_count,
            )
            yield _sse({"type": "error", "detail": "Generation failed before completion."})
            return

        log.info("generation.stream_complete", model=model_name, token_count=token_count)
        yield _sse({"type": "done", "model": model_name})
