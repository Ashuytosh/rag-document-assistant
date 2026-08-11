"""Document ingestion endpoint."""

import re
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.core.exceptions import (
    NOT_FOUND_DETAIL,
    DocumentNotFoundError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.models.chunk import ChunksResponse
from app.models.document import FILENAME_MAX_LENGTH, PREVIEW_CHARS, DocumentMetadata, IngestResponse
from app.services import storage
from app.services.chunking import chunk_document
from app.services.extraction import extract_text

router = APIRouter(tags=["ingestion"])
log = get_logger(__name__)

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _safe_filename(raw: str | None, fallback: str) -> str:
    """Reduce a client-supplied filename to a bounded, printable basename.

    The value never touches the filesystem — uploads are stored under a generated uuid —
    but it is logged and echoed back, so it is capped and stripped of control characters
    to keep log lines bounded and to avoid handing a future frontend raw markup.
    """
    if not raw:
        return fallback
    # Strip any directory component under either separator convention.
    basename = PureWindowsPath(PurePosixPath(raw).name).name
    cleaned = _CONTROL_CHARS.sub("", basename).strip()
    return cleaned[:FILENAME_MAX_LENGTH] or fallback


def _discard(*paths: Path) -> None:
    """Remove stored artifacts we can no longer use, without masking the real error.

    A partially-ingested document is worse than none: later phases would index text with
    no chunks beside it.
    """
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.warning("ingest.cleanup_failed", path=path.name)


def _validated_document_id(document_id: str) -> str:
    """Confirm ``document_id`` is a uuid before it is interpolated into a path.

    This is the only guard between a client-supplied id and the filesystem. A backslash
    is not a URL path separator, so ``..\\..\\secrets`` matches the route and would
    otherwise escape ``upload_dir`` on Windows. If a later phase adopts a different id
    scheme (Phase 7 dedup), this is the one place to loosen.
    """
    try:
        parsed = uuid.UUID(document_id)
    except ValueError as exc:
        raise DocumentNotFoundError(NOT_FOUND_DETAIL) from exc
    # uuid.UUID is lenient: it accepts `urn:uuid:...`, `{braces}`, unbracketed hex,
    # underscores, and non-ASCII digits. Require the canonical spelling so exactly one
    # id maps to one document, and so nothing unexpected reaches the filesystem.
    if str(parsed) != document_id:
        raise DocumentNotFoundError(NOT_FOUND_DETAIL)
    return str(parsed)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(description="A PDF or DOCX document.")],
) -> IngestResponse:
    """Accept a document upload, extract its text, persist both, and return metadata.

    The extracted text is written next to the upload for later pipeline phases; only
    metadata and a short preview come back to the client.
    """
    # Clear first: contextvars persist across requests served by the same task.
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=str(uuid.uuid4()))
    content_type = file.content_type or ""
    filename = _safe_filename(file.filename, fallback="upload")
    log.info("ingest.start", filename=filename, content_type=content_type)

    if content_type not in settings.allowed_content_types:
        log.warning("ingest.rejected", reason="unsupported_content_type", content_type=content_type)
        raise UnsupportedFileTypeError(f"Unsupported content type: {content_type or 'unknown'}")

    try:
        document_id, saved_path, size_bytes = await storage.save_upload(file, settings)
    except BaseException as exc:
        log.warning("ingest.failed", stage="save", error=type(exc).__name__)
        raise

    try:
        # extract_text is synchronous and CPU-bound — keep it off the event loop.
        extracted = await run_in_threadpool(extract_text, saved_path, content_type, settings)
    except BaseException as exc:
        # Nothing references an upload we failed to extract, so don't leave it behind.
        _discard(saved_path)
        log.warning("ingest.failed", stage="extract", error=type(exc).__name__)
        raise

    try:
        text_path = await run_in_threadpool(
            storage.save_text, document_id, extracted.text, settings
        )
    except BaseException as exc:
        # An upload with no extracted text beside it is unusable to later phases.
        _discard(saved_path)
        log.warning("ingest.failed", stage="save_text", error=type(exc).__name__)
        raise

    doc_metadata = {
        "filename": filename,
        "content_type": content_type,
        "page_count": extracted.page_count,
    }
    try:
        # Splitting is synchronous and CPU-bound — keep it off the event loop.
        chunks = await run_in_threadpool(
            chunk_document, extracted.text, document_id, doc_metadata, settings
        )
        await run_in_threadpool(storage.save_chunks, document_id, chunks, settings)
    except BaseException as exc:
        # Leave nothing half-ingested: text with no chunks would look complete to Phase 3.
        _discard(saved_path, text_path, storage.chunks_path(document_id, settings))
        log.warning("ingest.failed", stage="chunk", error=type(exc).__name__)
        raise

    metadata = DocumentMetadata(
        id=document_id,
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        page_count=extracted.page_count,
        char_count=extracted.char_count,
        text_preview=extracted.text[:PREVIEW_CHARS],
    )
    log.info(
        "ingest.success",
        document_id=document_id,
        char_count=extracted.char_count,
        page_count=extracted.page_count,
        chunk_count=len(chunks),
    )
    return IngestResponse(status="ok", document=metadata, chunk_count=len(chunks))


@router.get("/documents/{document_id}/chunks", response_model=ChunksResponse)
async def get_chunks(
    document_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
    limit: Annotated[int, Query(ge=1, le=500, description="Maximum chunks to return.")] = 50,
) -> ChunksResponse:
    """Return the persisted chunks for a document, capped at ``limit``.

    ``total_chunks`` always reports the full count, so a caller can tell that the list
    was truncated.
    """
    validated_id = _validated_document_id(document_id)
    total, chunks = await run_in_threadpool(storage.load_chunks, validated_id, settings, limit)
    return ChunksResponse(document_id=validated_id, total_chunks=total, chunks=chunks)
