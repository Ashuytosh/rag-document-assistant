"""Splitting extracted document text into overlapping chunks."""

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


def chunk_document(
    text: str, document_id: str, doc_metadata: ChunkMetadata, settings: Settings
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks carrying ``doc_metadata``.

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
