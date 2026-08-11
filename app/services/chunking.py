"""Splitting extracted document text into overlapping chunks."""

from bisect import bisect_right

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.core.logging import get_logger
from app.models.chunk import Chunk, ChunkMetadata

#: Rough characters-per-token ratio for English prose. Deliberately crude — Phase 3
#: replaces this with the embedding model's real tokenizer.
CHARS_PER_TOKEN = 4

log = get_logger(__name__)


def _build_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    """Construct the splitter from configuration.

    ``add_start_index`` makes the splitter record each chunk's offset in the source text,
    which is what later phases need to cite a passage back to its position.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=settings.chunk_separators,
        add_start_index=True,
        length_function=len,
    )


def page_for_offset(offset: int, page_offsets: list[int]) -> int:
    """Return the 1-based page containing ``offset``.

    ``page_offsets[i]`` is where page ``i + 1`` starts, so the page containing an offset
    is the last one that starts at or before it.

    Raises:
        ValueError: ``page_offsets`` is empty — "no page data" must not silently become
            "page 1", which would be a confidently wrong citation.
    """
    if not page_offsets:
        raise ValueError("page_offsets is empty; the document has no page information")
    return max(1, bisect_right(page_offsets, offset))


def chunk_document(
    text: str,
    document_id: str,
    doc_metadata: ChunkMetadata,
    settings: Settings,
    page_offsets: list[int] | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks carrying ``doc_metadata``.

    When ``page_offsets`` is supplied (PDFs), each chunk also records the page it starts
    on, so an answer can cite the passage's real location rather than the document's
    total page count. Formats without pages simply carry no ``page`` key.

    Short text yields a single chunk, which is fine. Empty text cannot reach here —
    extraction rejects it in Phase 1 — but it would simply yield an empty list.

    This is synchronous and CPU-bound; callers on the event loop must run it in a
    thread pool.
    """
    splitter = _build_splitter(settings)
    splits = splitter.create_documents([text], metadatas=[doc_metadata])

    chunks: list[Chunk] = []
    for index, split in enumerate(splits):
        # The splitter injects start_index into the metadata it propagates; lift it out
        # so it lands on the typed field rather than being duplicated inside `metadata`.
        # Indexed, not `.get`: if a refactor ever drops add_start_index, failing loudly
        # beats emitting a document's worth of chunks that all claim offset 0.
        metadata = dict(split.metadata)
        start_index = metadata.pop("start_index")
        if page_offsets:
            metadata["page"] = page_for_offset(start_index, page_offsets)
        char_count = len(split.page_content)
        chunks.append(
            Chunk(
                id=f"{document_id}:{index}",
                document_id=document_id,
                chunk_index=index,
                text=split.page_content,
                char_count=char_count,
                token_estimate=char_count // CHARS_PER_TOKEN,
                start_index=start_index,
                metadata=metadata,
            )
        )

    log.info("chunking.complete", document_id=document_id, chunk_count=len(chunks))
    return chunks
