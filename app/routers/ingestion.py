"""Document ingestion endpoint."""

import re
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.core.exceptions import UnsupportedFileTypeError
from app.core.logging import get_logger
from app.dependencies import get_embedding_service, get_vector_store
from app.models.chunk import ChildChunk, ChunksResponse
from app.models.document import FILENAME_MAX_LENGTH, PREVIEW_CHARS, DocumentMetadata, IngestResponse
from app.services import storage
from app.services.chunking import chunk_document
from app.services.embedding import EmbeddingService
from app.services.extraction import extract_text
from app.services.vector_store import VectorStoreService

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


def _discard_vectors(vector_store: VectorStoreService, document_id: str) -> None:
    """Drop a document's vectors, without masking the error that prompted the rollback."""
    try:
        vector_store.delete_document(document_id)
    except Exception:
        log.warning("ingest.vector_cleanup_failed", document_id=document_id)


def _embed_and_store(
    children: list[ChildChunk], embeddings: EmbeddingService, vector_store: VectorStoreService
) -> int:
    """Check child sizes against the model's window, then embed and store them.

    Children are what gets embedded, so they are what the 256-token window applies to.
    At ~400 characters they sit well inside it; this firing means the sizes drifted.

    Synchronous and CPU-bound: the caller runs this in a thread pool.
    """
    oversized = sum(1 for child in children if not embeddings.assert_within_limit(child.text))
    if oversized:
        log.warning("ingest.children_over_token_limit", count=oversized, total=len(children))
    return vector_store.add_children(children)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    settings: Annotated[Settings, Depends(get_settings)],
    embeddings: Annotated[EmbeddingService, Depends(get_embedding_service)],
    vector_store: Annotated[VectorStoreService, Depends(get_vector_store)],
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
        parents, children = await run_in_threadpool(
            chunk_document,
            extracted.text,
            document_id,
            doc_metadata,
            settings,
            extracted.page_offsets,
        )
        await run_in_threadpool(storage.save_parents, document_id, parents, settings)
    except BaseException as exc:
        # Leave nothing half-ingested: text with no parents beside it would look complete
        # to retrieval, which has no other copy of what an answer is written from.
        _discard(saved_path, text_path, storage.parents_path(document_id, settings))
        log.warning("ingest.failed", stage="chunk", error=type(exc).__name__)
        raise

    try:
        vectors_added = await run_in_threadpool(
            _embed_and_store, children, embeddings, vector_store
        )
    except BaseException as exc:
        # A partial vector add would still be counted by /stats and returned by /search,
        # so drop the document's vectors along with its files. Vectors first: children
        # that outlive their parents still match, and would then eat a slot in every
        # later query's child budget only to resolve to nothing.
        _discard_vectors(vector_store, document_id)
        _discard(saved_path, text_path, storage.parents_path(document_id, settings))
        log.warning("ingest.failed", stage="embed", error=type(exc).__name__)
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
        parent_count=len(parents),
        child_count=len(children),
        vectors_added=vectors_added,
    )
    return IngestResponse(
        status="ok",
        document=metadata,
        parent_count=len(parents),
        child_count=len(children),
    )


@router.get("/documents/{document_id}/chunks", response_model=ChunksResponse)
async def get_chunks(
    document_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
    limit: Annotated[int, Query(ge=1, le=500, description="Maximum parents to return.")] = 50,
) -> ChunksResponse:
    """Return the persisted parent chunks for a document, capped at ``limit``.

    Parents, not children: they are what is persisted, and what an answer is written
    from. Children exist only as vectors in the collection.

    ``total_chunks`` always reports the full count, so a caller can tell that the list
    was truncated.
    """
    validated_id = storage.validated_document_id(document_id)
    total, parents = await run_in_threadpool(
        storage.load_parents_page, validated_id, settings, limit
    )
    return ChunksResponse(document_id=validated_id, total_chunks=total, chunks=parents)
