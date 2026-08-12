"""Schemas for the chunks a document is split into before embedding."""

from pydantic import BaseModel, Field

#: Document-level metadata propagated onto every chunk. Values stay JSON-scalar because
#: Phase 3 hands them to ChromaDB, whose metadata values must be str/int/float/bool —
#: `page_count` being None will need handling there.
type ChunkMetadata = dict[str, str | int | float | bool | None]


class ParentChunk(BaseModel):
    """A large span of a document — the unit an answer is written from.

    Parents are never embedded. They are persisted whole and loaded back after retrieval
    has matched one of their children, so the model sees enough surrounding context to
    answer a question whose evidence spans more than one small passage.
    """

    id: str = Field(description="Deterministic identifier, `{document_id}:p{chunk_index}`.")
    document_id: str = Field(description="Id of the document this parent came from.")
    chunk_index: int = Field(ge=0, description="Position in the document, starting at 0.")
    text: str = Field(description="The parent's text.")
    char_count: int = Field(ge=0, description="Length of `text` in characters.")
    start_index: int = Field(
        ge=0, description="Character offset of this parent in the source text."
    )
    metadata: ChunkMetadata = Field(
        default_factory=dict,
        description="Document metadata propagated to the parent (filename, content_type, ...).",
    )


class ChildChunk(BaseModel):
    """A small span of a parent — the unit that is embedded and matched.

    ``start_index`` is absolute within the *document*, not within the parent, so a match
    can always be located back in the original text regardless of which parent it came
    from.
    """

    id: str = Field(description="Deterministic identifier, `{parent_id}:c{chunk_index}`.")
    parent_id: str = Field(description="Id of the parent this child was split from.")
    document_id: str = Field(description="Id of the document this child came from.")
    chunk_index: int = Field(ge=0, description="Position within the parent, starting at 0.")
    text: str = Field(description="The child's text.")
    char_count: int = Field(ge=0, description="Length of `text` in characters.")
    token_estimate: int = Field(
        ge=0, description="Token count as the embedding model counts them, where available."
    )
    start_index: int = Field(
        ge=0, description="Character offset of this child in the source document."
    )
    metadata: ChunkMetadata = Field(
        default_factory=dict,
        description="Document metadata propagated to the child (filename, content_type, ...).",
    )


class ChunksResponse(BaseModel):
    """Response body for ``GET /documents/{document_id}/chunks``.

    Serves parents: they are what is persisted, and what an answer is actually built
    from. Children exist only as vectors in the collection.
    """

    document_id: str
    total_chunks: int = Field(ge=0, description="Total parents stored, before `limit`.")
    chunks: list[ParentChunk] = Field(description="At most `limit` parents, from the start.")
