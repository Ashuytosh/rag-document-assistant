"""Persistence of uploaded documents and their extracted text."""

import json
import time
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
from app.models.chunk import ParentChunk

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


def validated_document_id(document_id: str) -> str:
    """Confirm ``document_id`` is a uuid before it is interpolated into a path.

    This is the only guard between a client-supplied id and the filesystem. A backslash
    is not a URL path separator, so ``..\\..\\secrets`` matches the route and would
    otherwise escape ``upload_dir`` on Windows. If a later phase adopts a different id
    scheme (Phase 9 dedup), this is the one place to loosen.
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


def parents_path(document_id: str, settings: Settings) -> Path:
    """Path of the persisted parents file for ``document_id``.

    Callers must have already validated ``document_id`` — it is client-supplied on the
    read path, and it is interpolated into a filename here.
    """
    return settings.upload_dir / f"{document_id}.parents.json"


def save_parents(document_id: str, parents: list[ParentChunk], settings: Settings) -> Path:
    """Write ``parents`` to ``{upload_dir}/{document_id}.parents.json``, returning its path.

    Parents are not embedded, so this file — not the collection — is the only copy of the
    text an answer is written from. Retrieval reads it back on every query.
    """
    destination = parents_path(document_id, settings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [parent.model_dump() for parent in parents]
    # Write-then-replace, so a crash mid-write cannot leave a truncated file that a
    # later read would have to treat as a missing document.
    staging = destination.with_suffix(".tmp")
    try:
        staging.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        staging.replace(destination)
    except BaseException:
        # A failed write must not leave its scratch file behind: nothing else ever looks
        # at it, so it would sit in the upload directory until someone noticed.
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            log.warning("storage.parents_staging_cleanup_failed", document_id=document_id)
        raise
    return destination


#: Attempts to read a parents file before giving up, and the pause between them. A
#: re-ingest rewrites the file with an atomic replace, and on Windows a reader that opens
#: it during that replace gets a sharing violation rather than either version. Without a
#: retry, re-indexing one document would silently cost every concurrent query a source.
_READ_ATTEMPTS = 3
_READ_RETRY_SECONDS = 0.02


def _read_with_retry(source: Path, document_id: str, settings: Settings) -> str:
    """Read ``source``, retrying briefly while it is locked by a concurrent rewrite.

    Only a transient lock is retried. A missing file is answered immediately: it means
    the document was deleted, which no amount of waiting will change.
    """
    for attempt in range(_READ_ATTEMPTS):
        try:
            if source.stat().st_size > settings.max_chunks_file_bytes:
                log.warning("storage.parents_oversize", document_id=document_id)
                raise DocumentNotFoundError(NOT_FOUND_DETAIL)
            return source.read_text(encoding="utf-8")
        except (FileNotFoundError, NotADirectoryError):
            raise
        except PermissionError:
            if attempt == _READ_ATTEMPTS - 1:
                log.warning("storage.parents_locked", document_id=document_id)
                raise
            time.sleep(_READ_RETRY_SECONDS)
    raise DocumentNotFoundError(NOT_FOUND_DETAIL)  # pragma: no cover - loop always returns


def _read_parents(document_id: str, settings: Settings) -> list[dict[str, object]]:
    """Read and sanity-check the raw parents payload for ``document_id``.

    Raises:
        DocumentNotFoundError: no parents file exists, or the one on disk is unusable.
    """
    source = parents_path(document_id, settings)
    try:
        raw = json.loads(_read_with_retry(source, document_id, settings))
        if not isinstance(raw, list):
            raise TypeError(f"expected a JSON list, got {type(raw).__name__}")
        return raw
    except OSError as exc:
        # Every OSError, not just FileNotFoundError: a file we cannot open must degrade
        # to one fewer source the way a missing one does, not 500 the whole query.
        raise DocumentNotFoundError(NOT_FOUND_DETAIL) from exc
    except (ValueError, TypeError) as exc:
        # Corrupt, truncated, or non-UTF-8: not a fault the caller can act on, and
        # surfacing it as a 500 would leak parser detail. Treat it as absent, but log it.
        # ValueError covers both JSONDecodeError and UnicodeDecodeError.
        log.warning("storage.parents_unreadable", document_id=document_id, error=type(exc).__name__)
        raise DocumentNotFoundError(NOT_FOUND_DETAIL) from exc


def load_parents_page(
    document_id: str, settings: Settings, limit: int | None = None
) -> tuple[int, list[ParentChunk]]:
    """Read the persisted parents for ``document_id``, in document order.

    Returns the total number stored and at most ``limit`` validated parents. Only the
    returned window is turned into models: a caller asking for 1 parent of 15,000 should
    not pay to validate all 15,000.
    """
    raw = _read_parents(document_id, settings)
    window = raw if limit is None else raw[:limit]
    try:
        return len(raw), [ParentChunk.model_validate(item) for item in window]
    except ValidationError as exc:
        log.warning("storage.parents_unreadable", document_id=document_id, error=type(exc).__name__)
        raise DocumentNotFoundError(NOT_FOUND_DETAIL) from exc


def load_parents(document_id: str, settings: Settings) -> dict[str, ParentChunk]:
    """Read every parent for ``document_id``, keyed by parent id.

    Keyed rather than ordered because retrieval arrives holding parent ids taken from the
    matched children, and needs to resolve them without scanning.
    """
    _total, parents = load_parents_page(document_id, settings)
    return {parent.id: parent for parent in parents}


def load_parents_by_id(
    document_id: str, wanted: set[str], settings: Settings
) -> dict[str, ParentChunk]:
    """Read only the parents in ``wanted``, keyed by id.

    The retrieval path needs a handful of parents from a document that may hold hundreds,
    and it runs on every query. Validating the whole file to return four entries is the
    difference between parsing a few kilobytes and a few megabytes per question, so the
    filter is applied to the raw payload before any model is built.
    """
    found: dict[str, ParentChunk] = {}
    for item in _read_parents(document_id, settings):
        if not isinstance(item, dict) or item.get("id") not in wanted:
            continue
        try:
            parent = ParentChunk.model_validate(item)
        except ValidationError as exc:
            log.warning(
                "storage.parent_unreadable", document_id=document_id, error=type(exc).__name__
            )
            continue
        found[parent.id] = parent
    return found


def load_parent(parent_id: str, settings: Settings) -> ParentChunk:
    """Read the single parent named by ``parent_id`` (``{document_id}:p{index}``).

    The document id is recovered from the prefix and validated before it reaches a path:
    parent ids travel through the vector store's metadata, so they are not automatically
    more trustworthy than any other stored value.

    Raises:
        DocumentNotFoundError: the id is malformed, or no such parent is stored.
    """
    document_id, separator, _ = parent_id.partition(":p")
    if not separator:
        raise DocumentNotFoundError(NOT_FOUND_DETAIL)
    parents = load_parents(validated_document_id(document_id), settings)
    parent = parents.get(parent_id)
    if parent is None:
        raise DocumentNotFoundError(NOT_FOUND_DETAIL)
    return parent


def delete_parents(document_id: str, settings: Settings) -> bool:
    """Remove the parents file for ``document_id``; True when one was there to remove.

    The staging file goes too, so a delete after a failed write leaves nothing behind.
    """
    destination = parents_path(document_id, settings)
    try:
        destination.with_suffix(".tmp").unlink(missing_ok=True)
    except OSError:
        log.warning("storage.parents_staging_cleanup_failed", document_id=document_id)
    try:
        destination.unlink()
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        log.warning("storage.parents_delete_failed", document_id=document_id)
        return False
    return True
