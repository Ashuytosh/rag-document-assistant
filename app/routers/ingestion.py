"""Document ingestion endpoint."""

import re
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.core.exceptions import UnsupportedFileTypeError
from app.core.logging import get_logger
from app.models.document import FILENAME_MAX_LENGTH, PREVIEW_CHARS, DocumentMetadata, IngestResponse
from app.services import storage
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


def _discard(path: Path) -> None:
    """Remove a stored upload we can no longer use, without masking the real error."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("ingest.cleanup_failed", path=path.name)


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
    except Exception as exc:
        log.warning("ingest.failed", stage="save", error=type(exc).__name__)
        raise

    try:
        # extract_text is synchronous and CPU-bound — keep it off the event loop.
        extracted = await run_in_threadpool(extract_text, saved_path, content_type, settings)
    except Exception as exc:
        # Nothing references an upload we failed to extract, so don't leave it behind.
        _discard(saved_path)
        log.warning("ingest.failed", stage="extract", error=type(exc).__name__)
        raise

    try:
        await run_in_threadpool(storage.save_text, document_id, extracted.text, settings)
    except Exception as exc:
        # An upload with no extracted text beside it is unusable to later phases.
        _discard(saved_path)
        log.warning("ingest.failed", stage="save_text", error=type(exc).__name__)
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
    )
    return IngestResponse(status="ok", document=metadata)
