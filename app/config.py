"""Application configuration, loaded from the environment or a local `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Upper bound on how many results a single search may request, so one call cannot ask
#: the store for an unbounded number of vectors.
MAX_TOP_K = 50

PDF_CONTENT_TYPE = "application/pdf"
DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

#: Extension to use on disk for each accepted content type. Uploads are stored under a
#: generated uuid plus one of these extensions, so the client-supplied filename never
#: reaches the filesystem and cannot be used for path traversal.
CONTENT_TYPE_EXTENSIONS: dict[str, str] = {
    PDF_CONTENT_TYPE: ".pdf",
    DOCX_CONTENT_TYPE: ".docx",
}


class Settings(BaseSettings):
    """Runtime settings. Every field has a usable default for local development."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG Document Assistant"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    upload_dir: Path = Path("data/uploads")
    max_upload_bytes: int = 20 * 1024 * 1024
    allowed_content_types: set[str] = set(CONTENT_TYPE_EXTENSIONS)

    #: A DOCX is a zip container, and python-docx reads members fully into memory. These
    #: bound decompression so a small upload cannot expand into an out-of-memory kill.
    max_uncompressed_bytes: int = 100 * 1024 * 1024
    max_compression_ratio: int = 100

    #: Chunking is character-based for now; Phase 3 swaps in the embedding model's real
    #: tokenizer. 800 chars stays inside all-MiniLM-L6-v2's ~256-token window with margin.
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    #: Split priority: paragraph, then line, then sentence, then word, then hard cut.
    chunk_separators: list[str] = ["\n\n", "\n", ". ", " ", ""]
    #: Ceiling on a chunk file before a read refuses it, so serving chunks for a huge
    #: document cannot be used to amplify memory use on repeated requests.
    max_chunks_file_bytes: int = 32 * 1024 * 1024

    #: Embeddings run on the CPU by default: the 8 GB VRAM budget is reserved for Ollama.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    #: Pin the exact model snapshot. Without it, a moved upstream `main` would be
    #: downloaded and loaded silently on the next restart.
    embedding_model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    #: Load from the local cache only. Startup otherwise makes ~20 calls to
    #: huggingface.co to revalidate an already-complete cache, which both slows boot and
    #: makes the service unable to start when that host is unreachable. Set false for
    #: the first run, which populates the cache.
    embedding_local_files_only: bool = False
    embedding_device: str = "cpu"
    #: Normalized vectors make cosine distance equivalent to the dot product, which is
    #: what lets `score = 1 - distance` be a meaningful similarity.
    embedding_normalize: bool = True
    embedding_max_tokens: int = Field(default=256, gt=0)

    chroma_persist_dir: Path = Path("chroma_db")
    #: Chroma requires 3-512 chars from [a-zA-Z0-9._-], starting and ending alphanumeric.
    chroma_collection: str = Field(
        default="documents",
        min_length=3,
        max_length=512,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$",
    )
    #: The HNSW space. `score = 1 - distance` only holds for cosine, so this is a Literal
    #: rather than a free string — a typo would silently produce meaningless scores.
    chroma_distance: Literal["cosine"] = "cosine"

    search_top_k: int = Field(default=5, gt=0, le=MAX_TOP_K)

    @model_validator(mode="after")
    def _check_chunk_bounds(self) -> "Settings":
        """Fail at startup rather than per-request when the chunk window is nonsensical.

        The window must advance by at least half a chunk each step. Merely requiring
        ``overlap < size`` is not enough: at 99/100 the splitter makes so little forward
        progress that it emits many identical chunks and silently drops the rest of the
        document — text that would then never be indexed.
        """
        max_overlap = self.chunk_size // 2
        if self.chunk_overlap > max_overlap:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be at most half of "
                f"chunk_size ({self.chunk_size}), i.e. <= {max_overlap}."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, cached for use as a FastAPI dependency."""
    return Settings()
