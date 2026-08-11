"""Application configuration, loaded from the environment or a local `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
