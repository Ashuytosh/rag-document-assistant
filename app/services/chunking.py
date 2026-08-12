"""Splitting extracted document text into parent and child chunks.

Two levels, because matching and answering want opposite things. Children are small, so a
vector describes one idea and matches it sharply. Parents are large, so the model has the
surrounding context needed to answer a question whose evidence spans several sentences.
Only children are embedded; only parents are persisted and shown to the model.
"""

from bisect import bisect_right

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.core.logging import get_logger
from app.models.chunk import ChildChunk, ChunkMetadata, ParentChunk

#: Rough characters-per-token ratio for English prose, used for the child token estimate.
#: Crude on purpose: the authoritative count is the embedding model's own tokenizer, which
#: `EmbeddingService.assert_within_limit` applies at ingestion.
CHARS_PER_TOKEN = 4

log = get_logger(__name__)


def _build_splitter(settings: Settings, size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    """Construct a splitter at the given size, sharing the configured separators.

    ``add_start_index`` makes the splitter record each piece's offset in the text it was
    given, which is what citations are built from.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
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


def _with_page(
    metadata: ChunkMetadata, start_index: int, page_offsets: list[int] | None
) -> ChunkMetadata:
    """Copy ``metadata``, stamping the page ``start_index`` falls on when pages are known."""
    stamped = dict(metadata)
    if page_offsets:
        stamped["page"] = page_for_offset(start_index, page_offsets)
    return stamped


def _split_parent(
    parent: ParentChunk,
    splitter: RecursiveCharacterTextSplitter,
    page_offsets: list[int] | None,
) -> list[ChildChunk]:
    """Split one parent into the children that will be embedded in its place.

    The splitter is passed in rather than built here: its configuration never varies, and
    a document is many parents.

    Offsets are made absolute (``parent.start_index + local``) before anything is stored,
    so a child cites its position in the document rather than in its parent — the parent
    is an implementation detail of retrieval, not something a reader can locate.
    """
    splits = splitter.create_documents([parent.text])

    children: list[ChildChunk] = []
    for index, split in enumerate(splits):
        # Indexed, not `.get`: if a refactor ever drops add_start_index, failing loudly
        # beats emitting a document's worth of children that all claim offset 0.
        start_index = parent.start_index + split.metadata["start_index"]
        char_count = len(split.page_content)
        children.append(
            ChildChunk(
                id=f"{parent.id}:c{index}",
                parent_id=parent.id,
                document_id=parent.document_id,
                chunk_index=index,
                text=split.page_content,
                char_count=char_count,
                token_estimate=char_count // CHARS_PER_TOKEN,
                start_index=start_index,
                metadata=_with_page(parent.metadata, start_index, page_offsets),
            )
        )
    return children


def chunk_document(
    text: str,
    document_id: str,
    doc_metadata: ChunkMetadata,
    settings: Settings,
    page_offsets: list[int] | None = None,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Split ``text`` into parents, and each parent into children.

    When ``page_offsets`` is supplied (PDFs), every chunk at both levels records the page
    it starts on, so an answer can cite the passage's real location rather than the
    document's total page count. Formats without pages simply carry no ``page`` key.

    Short text yields one parent and one child, which is fine. Empty text cannot reach
    here — extraction rejects it in Phase 1 — but it would simply yield two empty lists.

    This is synchronous and CPU-bound; callers on the event loop must run it in a
    thread pool.
    """
    parent_splitter = _build_splitter(
        settings, settings.parent_chunk_size, settings.parent_chunk_overlap
    )
    child_splitter = _build_splitter(
        settings, settings.child_chunk_size, settings.child_chunk_overlap
    )
    splits = parent_splitter.create_documents([text], metadatas=[doc_metadata])

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    for index, split in enumerate(splits):
        # The splitter injects start_index into the metadata it propagates; lift it out
        # so it lands on the typed field rather than being duplicated inside `metadata`.
        metadata = dict(split.metadata)
        start_index = metadata.pop("start_index")
        parent = ParentChunk(
            id=f"{document_id}:p{index}",
            document_id=document_id,
            chunk_index=index,
            text=split.page_content,
            char_count=len(split.page_content),
            start_index=start_index,
            metadata=_with_page(metadata, start_index, page_offsets),
        )
        parents.append(parent)
        children.extend(_split_parent(parent, child_splitter, page_offsets))

    log.info(
        "chunking.complete",
        document_id=document_id,
        parent_count=len(parents),
        child_count=len(children),
    )
    return parents, children
