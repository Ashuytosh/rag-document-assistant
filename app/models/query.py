"""Request/response schemas for grounded question answering."""

import re
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import MAX_GENERATION_TOP_K
from app.models.search import QUERY_MAX_LENGTH

#: Shape check only — the authoritative control is the allowlist in Settings, checked in
#: the router. This keeps obviously malformed names out of the request at all.
MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*(:[a-zA-Z0-9._-]+)?$")
MODEL_NAME_MAX_LENGTH = 128


class QueryRequest(BaseModel):
    """Body for ``POST /query``."""

    # Consistent with SearchRequest: a misspelled field silently taking the default is
    # worse than a 422.
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=QUERY_MAX_LENGTH,
        description="The question to answer from the indexed documents.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=MAX_GENERATION_TOP_K,
        description="How many chunks to retrieve as context; defaults to the configured value.",
    )
    document_id: UUID | None = Field(
        default=None, description="Restrict retrieval to a single document."
    )
    model: str | None = Field(
        default=None,
        max_length=MODEL_NAME_MAX_LENGTH,
        description="Override the generation model for this request.",
    )
    stream: bool = Field(default=True, description="Stream the answer as server-sent events.")

    @field_validator("query")
    @classmethod
    def _reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain non-whitespace characters")
        return value

    @field_validator("model")
    @classmethod
    def _check_model_name(cls, value: str | None) -> str | None:
        if value is not None and not MODEL_NAME_PATTERN.match(value):
            raise ValueError("model must be a valid Ollama model name")
        return value


class Source(BaseModel):
    """One retrieved chunk, as cited in the answer.

    ``source_num`` matches the ``[Source N]`` label the model was given, so a citation in
    the answer text resolves back to this entry.
    """

    source_num: int = Field(ge=1, description="The [Source N] label used in the prompt.")
    chunk_id: str
    document_id: str
    filename: str
    page: int | None = Field(
        default=None,
        description="1-based page the chunk starts on; None for formats without pages.",
    )
    start_index: int = Field(ge=0, description="Character offset of the chunk in the document.")
    score: float = Field(description="Cosine similarity to the question.")
    snippet: str = Field(description="Leading characters of the chunk, for display.")


class QueryResponse(BaseModel):
    """Response body for a non-streaming ``POST /query``."""

    query: str
    answer: str
    sources: list[Source]
    model: str = Field(description="The model that produced the answer.")
