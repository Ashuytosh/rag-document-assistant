"""Request/response schemas for document ingestion."""

from pydantic import BaseModel, Field

#: Number of leading characters of extracted text echoed back to the client.
PREVIEW_CHARS = 500
#: Upper bound on the echoed original filename, matching common filesystem limits.
FILENAME_MAX_LENGTH = 255


class DocumentMetadata(BaseModel):
    """What we know about a document after extracting its text.

    The full extracted text is deliberately *not* part of this model — it is persisted
    alongside the upload and consumed by later pipeline phases.
    """

    id: str = Field(description="Server-generated uuid4 identifying the stored document.")
    filename: str = Field(
        max_length=FILENAME_MAX_LENGTH,
        description="Original filename as supplied by the client, reduced to a basename.",
    )
    content_type: str = Field(description="MIME type the document was accepted as.")
    size_bytes: int = Field(ge=0, description="Size of the stored upload on disk.")
    page_count: int | None = Field(
        default=None, description="Page count for PDFs; None for formats without pages."
    )
    char_count: int = Field(ge=0, description="Character count of the extracted text.")
    text_preview: str = Field(description=f"First ~{PREVIEW_CHARS} characters of the text.")


class IngestResponse(BaseModel):
    """Response body for a successful ``POST /ingest``."""

    status: str = "ok"
    document: DocumentMetadata
    chunk_count: int = Field(
        ge=0, description="Number of chunks the extracted text was split into and persisted."
    )
