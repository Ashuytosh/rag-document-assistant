"""Request/response schemas for similarity search over stored chunks."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import MAX_TOP_K
from app.models.chunk import ChunkMetadata

#: Ceiling on a search query. The text is embedded, so an unbounded string would be
#: unbounded model work on an unauthenticated endpoint.
QUERY_MAX_LENGTH = 1000


class SearchRequest(BaseModel):
    """Body for ``POST /search``."""

    # Reject unknown fields: a misspelled `topk` silently falling back to the default
    # is worse than a 422, especially for a frontend-driven API.
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=QUERY_MAX_LENGTH,
        description="Natural-language query to match against stored chunks.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=MAX_TOP_K,
        description="How many results to return; falls back to the configured default.",
    )
    #: Typed as a UUID to match the id scheme `/ingest` mints and the validation the
    #: chunks endpoint already applies — one standard for one identifier. It also keeps
    #: anything but a plain string out of Chroma's filter language, where a dict would
    #: be interpreted as an operator such as `$ne`.
    document_id: UUID | None = Field(
        default=None, description="Restrict the search to a single document."
    )

    @field_validator("query")
    @classmethod
    def _reject_blank_query(cls, value: str) -> str:
        """A whitespace-only query embeds fine and returns confident nonsense."""
        if not value.strip():
            raise ValueError("query must contain non-whitespace characters")
        return value


class SearchResult(BaseModel):
    """One matching chunk, with its similarity to the query."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float = Field(
        description="Cosine similarity in [-1, 1]; 1.0 is identical. Higher is better."
    )
    metadata: ChunkMetadata = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response body for ``POST /search``."""

    query: str
    count: int = Field(ge=0, description="Number of results returned.")
    results: list[SearchResult]


class StatsResponse(BaseModel):
    """Response body for ``GET /stats``."""

    total_vectors: int = Field(ge=0, description="Vectors currently in the collection.")
    collection: str
