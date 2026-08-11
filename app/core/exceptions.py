"""Custom ingestion exceptions and the FastAPI handlers that render them as JSON."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class IngestionError(Exception):
    """Base class for errors raised while ingesting a document.

    Each subclass maps to exactly one HTTP status code via ``status_code``.
    """

    status_code: int = status.HTTP_400_BAD_REQUEST

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class UnsupportedFileTypeError(IngestionError):
    """The upload's content type is not one we can extract text from."""

    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


class FileTooLargeError(IngestionError):
    """The upload exceeded the configured maximum size."""

    status_code = status.HTTP_413_CONTENT_TOO_LARGE


class ExtractionError(IngestionError):
    """The document could not be parsed, or yielded no text."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


#: Fixed text for every not-found path, so a client-supplied id is never reflected back
#: into a response body. The id belongs in the structured log, not the payload.
NOT_FOUND_DETAIL = "No chunks found for the requested document."


class DocumentNotFoundError(IngestionError):
    """No stored document matches the requested id.

    Also raised for a malformed id: the client cannot distinguish the two, which keeps
    the endpoint from confirming what a valid id looks like.
    """

    status_code = status.HTTP_404_NOT_FOUND


class UnsupportedModelError(IngestionError):
    """The requested generation model is not in the configured allowlist."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


class GenerationError(IngestionError):
    """The language model could not be reached or failed to produce an answer.

    A 503 for the same reason as :class:`VectorStoreError`: it is a dependency being
    unavailable, not a problem with the caller's request.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class VectorStoreError(IngestionError):
    """The vector store could not be reached, read, or written.

    A 503 rather than a 500: the request may well succeed once the store recovers, and
    the caller learns nothing about why it failed.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


async def ingestion_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Render an :class:`IngestionError` as ``{"error": ..., "detail": ...}``.

    Only the exception type name and its message are exposed — never a stack trace or
    a filesystem path.
    """
    if not isinstance(exc, IngestionError):  # pragma: no cover - registration guarantees this
        raise exc
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": type(exc).__name__, "detail": exc.detail},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the JSON handler for ingestion errors.

    Registered against the base class: Starlette resolves handlers by walking the
    exception's MRO, so every current and future subclass is covered automatically.
    """
    app.add_exception_handler(IngestionError, ingestion_error_handler)
