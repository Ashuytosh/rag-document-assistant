"""Grounded question answering over the indexed documents."""

import time
import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.core.exceptions import UnsupportedModelError
from app.core.logging import get_logger
from app.dependencies import get_generation_service
from app.models.query import QueryRequest, QueryResponse
from app.services.generation import GenerationService

router = APIRouter(tags=["query"])
log = get_logger(__name__)


@router.post("/query", response_model=None, responses={200: {"model": QueryResponse}})
async def query(
    request: QueryRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    generation: Annotated[GenerationService, Depends(get_generation_service)],
) -> StreamingResponse | QueryResponse:
    """Answer a question from the indexed documents, with citations.

    Retrieval and model resolution both happen here rather than inside the stream: an
    async generator's body does not run until after ``StreamingResponse`` has sent the
    200 and headers, so anything failing in there could only appear as a malformed
    stream. Doing both first keeps those failures ordinary HTTP errors.
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=str(uuid.uuid4()))
    started = time.perf_counter()

    if request.model is not None and request.model not in settings.allowed_generation_models:
        # An unknown-but-well-formed name would otherwise reach Ollama, fail, and
        # surface as a 503 — blaming the dependency for the caller's bad input.
        raise UnsupportedModelError(f"Unsupported model: {request.model}")

    top_k = request.top_k or settings.generation_top_k
    document_id = str(request.document_id) if request.document_id else None
    # Retrieval embeds the query: blocking CPU work, off the event loop.
    prepared = await run_in_threadpool(
        generation.retrieve_and_build, request.query, top_k, document_id
    )
    chat_model, model_name = generation.model_for(request.model)

    log.info(
        "query.retrieved",
        top_k=top_k,
        document_id=document_id,
        source_count=len(prepared[1]),
        model=model_name,
        stream=request.stream,
        retrieval_ms=round((time.perf_counter() - started) * 1000),
    )

    if request.stream:
        return StreamingResponse(
            generation.astream(request, prepared, chat_model, model_name),
            media_type="text/event-stream",
            headers={
                # Answers carry document content; keep them out of browser and proxy
                # caches, and stop proxies buffering away the point of streaming.
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            },
        )

    response = await generation.generate(request, prepared)
    log.info(
        "query.complete",
        source_count=len(response.sources),
        answer_chars=len(response.answer),
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )
    return response
