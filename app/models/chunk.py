"""Schemas for the chunks a document is split into before embedding."""

from pydantic import BaseModel, Field

#: Document-level metadata propagated onto every chunk. Values stay JSON-scalar because
#: Phase 3 hands them to ChromaDB, whose metadata values must be str/int/float/bool —
#: `page_count` being None will need handling there.
type ChunkMetadata = dict[str, str | int | float | bool | None]


class Chunk(BaseModel):
    """One span of a document's extracted text, ready to be embedded.

    ``start_index`` is the chunk's character offset in the source text. It carries no
    weight yet, but it is the groundwork for citations: it is what will let a retrieved
    passage be located back in the original document.
    """

    id: str = Field(description="Deterministic identifier, `{document_id}:{chunk_index}`.")
    document_id: str = Field(description="Id of the document this chunk came from.")
    chunk_index: int = Field(ge=0, description="Position in the document, starting at 0.")
    text: str = Field(description="The chunk's text.")
    char_count: int = Field(ge=0, description="Length of `text` in characters.")
    token_estimate: int = Field(
        ge=0,
        description="Rough token count (chars // 4); replaced by the real tokenizer in Phase 3.",
    )
    start_index: int = Field(ge=0, description="Character offset of this chunk in the source text.")
    metadata: ChunkMetadata = Field(
        default_factory=dict,
        description="Document metadata propagated to the chunk (filename, content_type, ...).",
    )


class ChunksResponse(BaseModel):
    """Response body for ``GET /documents/{document_id}/chunks``."""

    document_id: str
    total_chunks: int = Field(ge=0, description="Total stored for the document, before `limit`.")
    chunks: list[Chunk] = Field(description="At most `limit` chunks, from the start.")
