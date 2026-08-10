"""Application configuration, loaded from the environment or a local `.env`."""

from functools import lru_cache
from pathlib import Path

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


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, cached for use as a FastAPI dependency."""
    return Settings()
