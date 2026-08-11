"""Persistence of uploaded documents and their extracted text."""

import json
import uuid
from pathlib import Path

from fastapi import UploadFile
from pydantic import ValidationError

from app.config import CONTENT_TYPE_EXTENSIONS, Settings
from app.core.exceptions import (
    NOT_FOUND_DETAIL,
    DocumentNotFoundError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.models.chunk import Chunk

#: Read the upload in 1 MB slices so a large file never lands in memory whole.
CHUNK_BYTES = 1024 * 1024

log = get_logger(__name__)


async def save_upload(upload: UploadFile, settings: Settings) -> tuple[str, Path, int]:
    """Stream ``upload`` to disk under a generated id, enforcing the size limit.

    The stored filename is ``{uuid4}{ext}``, where ``ext`` is derived from the content
    type rather than from ``upload.filename`` — an attacker-controlled name therefore
    never influences the path we write to.

    Returns the generated document id, the path written, and the number of bytes stored.

    Raises:
        UnsupportedFileTypeError: the content type has no known extension.
        FileTooLargeError: the upload exceeded ``settings.max_upload_bytes``.
    """
    content_type = upload.content_type or ""
    extension = CONTENT_TYPE_EXTENSIONS.get(content_type)
    if extension is None:
        raise UnsupportedFileTypeError(f"Unsupported content type: {content_type or 'unknown'}")

    document_id = str(uuid.uuid4())
    destination = settings.upload_dir / f"{document_id}{extension}"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    bytes_written = 0
    try:
        with destination.open("wb") as sink:
            while chunk := await upload.read(CHUNK_BYTES):
                bytes_written += len(chunk)
                if bytes_written > settings.max_upload_bytes:
                    raise FileTooLargeError(
                        f"Upload exceeds the maximum of {settings.max_upload_bytes} bytes."
                    )
                sink.write(chunk)
    except BaseException:
        # Never leave a partial or orphaned file behind, whether we rejected the upload
        # for size or the write itself failed. A failure to clean up must not replace
        # the original exception — that would turn a 413 into a 500.
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            log.warning("storage.cleanup_failed", document_id=document_id)
        raise

    return document_id, destination, bytes_written


def save_text(document_id: str, text: str, settings: Settings) -> Path:
    """Write extracted ``text`` to ``{upload_dir}/{document_id}.txt`` and return its path."""
    destination = settings.upload_dir / f"{document_id}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def chunks_path(document_id: str, settings: Settings) -> Path:
    """Path of the persisted chunk file for ``document_id``.

    Callers must have already validated ``document_id`` — it is client-supplied on the
    read path, and it is interpolated into a filename here.
    """
    return settings.upload_dir / f"{document_id}.chunks.json"


def save_chunks(document_id: str, chunks: list[Chunk], settings: Settings) -> Path:
    """Write ``chunks`` to ``{upload_dir}/{document_id}.chunks.json`` and return its path.

    Interim persistence: Phase 3 moves chunks into the vector store.
    """
    destination = chunks_path(document_id, settings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [chunk.model_dump() for chunk in chunks]
    # Write-then-replace, so a crash mid-write cannot leave a truncated file that a
    # later read would have to treat as a missing document.
    staging = destination.with_suffix(".tmp")
    staging.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    staging.replace(destination)
    return destination


def load_chunks(
    document_id: str, settings: Settings, limit: int | None = None
) -> tuple[int, list[Chunk]]:
    """Read the persisted chunks for ``document_id``.

    Returns the total number stored and at most ``limit`` validated chunks. Only the
    returned window is turned into models: a caller asking for 1 chunk of 15,000 should
    not pay to validate all 15,000.

    Raises:
        DocumentNotFoundError: no chunk file exists, or the one on disk is unusable.
    """
    source = chunks_path(document_id, settings)
    try:
        if source.stat().st_size > settings.max_chunks_file_bytes:
            log.warning("storage.chunks_oversize", document_id=document_id)
            raise DocumentNotFoundError(NOT_FOUND_DETAIL)
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise TypeError(f"expected a JSON list, got {type(raw).__name__}")
        window = raw if limit is None else raw[:limit]
        return len(raw), [Chunk.model_validate(item) for item in window]
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise DocumentNotFoundError(NOT_FOUND_DETAIL) from exc
    except (ValueError, ValidationError, TypeError) as exc:
        # Corrupt, truncated, or non-UTF-8: not a fault the caller can act on, and
        # surfacing it as a 500 would leak parser detail. Treat it as absent, but log it.
        # ValueError covers both JSONDecodeError and UnicodeDecodeError.
        log.warning("storage.chunks_unreadable", document_id=document_id, error=type(exc).__name__)
        raise DocumentNotFoundError(NOT_FOUND_DETAIL) from exc
