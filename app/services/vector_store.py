"""Persistent vector storage and similarity search over document chunks."""

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.models.chunk import Chunk, ChunkMetadata
from app.models.search import SearchResult

log = get_logger(__name__)


def _chunk_metadata(chunk: Chunk) -> ChunkMetadata:
    """Flatten a chunk into the metadata stored alongside its vector.

    Document metadata is spread first so the identity fields below always win: a
    document whose own metadata happened to carry ``document_id`` must not be able to
    shadow the real one and break the search filter.

    ``None`` values are dropped. Chroma discards them itself, but doing it here makes the
    stored shape explicit and version-independent: a DOCX chunk simply carries no
    ``page_count`` key, and consumers test for presence rather than a sentinel.
    """
    metadata: ChunkMetadata = {
        **chunk.metadata,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "start_index": chunk.start_index,
    }
    return {key: value for key, value in metadata.items() if value is not None}


class VectorStoreService:
    """A persistent Chroma collection of chunk embeddings.

    The embedding function is injected rather than constructed here, so tests can supply
    a fast deterministic fake and exercise real Chroma behaviour without the model.
    """

    def __init__(self, settings: Settings, embeddings: Embeddings) -> None:
        self._settings = settings
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

        # Own the client so `count()` uses public API, and so telemetry is explicitly
        # off rather than off by accident (chromadb defaults it on; it currently no-ops
        # only because `posthog` happens not to be installed).
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": settings.chroma_distance},
        )
        self._assert_distance_matches()
        self._store = Chroma(
            client=self._client,
            collection_name=settings.chroma_collection,
            embedding_function=embeddings,
        )

    def _assert_distance_matches(self) -> None:
        """Fail fast when an existing collection uses a different distance metric.

        ``get_or_create_collection`` ignores the metadata when the collection already
        exists, so a directory created with, say, L2 would silently make ``1 - distance``
        meaningless — unbounded, and not a cosine similarity at all.
        """
        space = (self._collection.metadata or {}).get("hnsw:space")
        if space != self._settings.chroma_distance:
            raise VectorStoreError(
                f"Collection '{self._settings.chroma_collection}' uses distance "
                f"'{space}', but '{self._settings.chroma_distance}' is configured."
            )

    @property
    def collection_name(self) -> str:
        return self._settings.chroma_collection

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and store ``chunks``, returning how many were written.

        Ids are the chunks' deterministic ``{document_id}:{index}``, so adding the same
        chunks again overwrites them rather than duplicating. Note this is idempotent
        *per document id*: ``/ingest`` mints a fresh uuid per upload, so re-uploading the
        same file still produces a second copy. Content-based dedup is Phase 7.
        """
        if not chunks:
            return 0
        try:
            self._store.add_texts(
                texts=[chunk.text for chunk in chunks],
                metadatas=[_chunk_metadata(chunk) for chunk in chunks],
                ids=[chunk.id for chunk in chunks],
            )
        except Exception as exc:
            raise VectorStoreError("Could not store document vectors.") from exc
        return len(chunks)

    def search(self, query: str, top_k: int, document_id: str | None = None) -> list[SearchResult]:
        """Return the ``top_k`` chunks most similar to ``query``, best first.

        An explicit ``document_id`` always filters, including the empty string: treating
        a falsy id as "no filter" would silently widen a scoped query to the whole
        corpus, which is the opposite of what the caller asked for.
        """
        where = None if document_id is None else {"document_id": document_id}
        try:
            matches = self._store.similarity_search_with_score(query, k=top_k, filter=where)
        except Exception as exc:
            raise VectorStoreError("Could not run the similarity search.") from exc

        results = [
            SearchResult(
                chunk_id=self._chunk_id(document),
                document_id=str(document.metadata.get("document_id", "")),
                chunk_index=int(document.metadata.get("chunk_index", 0)),
                text=document.page_content,
                # Vectors are normalized and the space is cosine, so distance is
                # 1 - cosine_similarity; inverting it recovers the similarity.
                score=1.0 - distance,
                metadata=dict(document.metadata),
            )
            for document, distance in matches
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results

    @staticmethod
    def _chunk_id(document: object) -> str:
        """Recover a chunk's id, falling back to its metadata if Chroma omits it."""
        identifier = getattr(document, "id", None)
        if identifier:
            return str(identifier)
        metadata = getattr(document, "metadata", {}) or {}
        return f"{metadata.get('document_id', '')}:{metadata.get('chunk_index', 0)}"

    def delete_document(self, document_id: str) -> int:
        """Remove every vector belonging to ``document_id``, returning how many went.

        The delete is issued by filter rather than by the ids just read, so a chunk
        written between the count and the delete is still removed.
        """
        try:
            # include=[] fetches ids only; the default would pull every chunk's text
            # into memory just to count them.
            existing = self._collection.get(where={"document_id": document_id}, include=[])
            removed = len(existing.get("ids") or [])
            if removed:
                self._collection.delete(where={"document_id": document_id})
        except Exception as exc:
            raise VectorStoreError("Could not delete document vectors.") from exc
        return removed

    def count(self) -> int:
        """Number of vectors currently in the collection."""
        return self._collection.count()
