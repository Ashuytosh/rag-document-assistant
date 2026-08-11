"""The embedding model: loaded once, reused for every request."""

from typing import Any

from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

#: Used only when the injected embeddings expose no tokenizer (tests).
CHARS_PER_TOKEN_FALLBACK = 4


class EmbeddingService:
    """Wraps the sentence-transformers model behind a small, testable surface.

    Construction loads the model, which is slow and allocates memory — build one at
    startup and inject it. Nothing here is safe to call per-request without that.
    """

    def __init__(self, settings: Settings, embeddings: Embeddings | None = None) -> None:
        self._settings = settings
        self._injected = embeddings is not None
        self._embeddings = embeddings or self._load_model(settings)
        self._tokenizer = self._resolve_tokenizer()
        if self._tokenizer is None and not self._injected:
            # The real model always has one; missing means an upstream rename, and the
            # character fallback would silently misreport every token count.
            log.warning("embedding.tokenizer_unavailable", model=settings.embedding_model_name)
        log.info(
            "embedding.model_loaded",
            model=settings.embedding_model_name,
            device=settings.embedding_device,
            normalize=settings.embedding_normalize,
            injected=self._injected,
        )

    @staticmethod
    def _load_model(settings: Settings) -> Embeddings:
        """Build the real HuggingFace embeddings.

        Imported here rather than at module scope so the suite — which always injects a
        fake — never pays to import torch and sentence-transformers.
        """
        from langchain_huggingface import HuggingFaceEmbeddings

        model_kwargs: dict[str, Any] = {
            "device": settings.embedding_device,
            # Never execute model-repo code, even if a future revision ships some.
            "trust_remote_code": False,
            "local_files_only": settings.embedding_local_files_only,
        }
        if settings.embedding_model_revision:
            model_kwargs["revision"] = settings.embedding_model_revision
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": settings.embedding_normalize},
        )

    def _resolve_tokenizer(self) -> Any | None:
        """Find the underlying SentenceTransformer's tokenizer, if there is one.

        langchain-huggingface keeps the model on ``_client``; older releases used
        ``client``. Resolved once here rather than probed per call, so a rename shows up
        as one warning at startup instead of silently degrading every count.
        """
        model = getattr(self._embeddings, "_client", None) or getattr(
            self._embeddings, "client", None
        )
        return getattr(model, "tokenizer", None)

    def as_langchain(self) -> Embeddings:
        """Return the underlying embeddings object, for the vector store to own."""
        return self._embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        return self._embeddings.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        return self._embeddings.embed_documents(texts)

    def count_tokens(self, text: str) -> int:
        """Count tokens the way the embedding model will.

        Falls back to a character-based estimate only when no tokenizer is available —
        the case for an injected fake in tests.
        """
        if self._tokenizer is None:
            return len(text) // CHARS_PER_TOKEN_FALLBACK
        return len(self._tokenizer.encode(text, add_special_tokens=True))

    def assert_within_limit(self, text: str) -> bool:
        """Warn when ``text`` would be truncated by the model's context window.

        Deliberately a warning, not an error: the model silently truncates, so an
        oversized chunk degrades retrieval quality rather than breaking the request.
        Phase 2's 800-character chunks sit well inside the 256-token window, so this
        firing means the chunk settings drifted.
        """
        token_count = self.count_tokens(text)
        if token_count > self._settings.embedding_max_tokens:
            log.warning(
                "embedding.chunk_over_limit",
                token_count=token_count,
                limit=self._settings.embedding_max_tokens,
            )
            return False
        return True
