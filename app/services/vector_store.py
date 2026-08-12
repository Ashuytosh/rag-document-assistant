"""Persistent vector storage and small-to-big retrieval over document chunks.

Children are embedded and matched; parents are loaded back from the parent store and
returned, so what is searched over and what is answered from are different units.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import MAX_TOP_K, Settings
from app.core.exceptions import DocumentNotFoundError, VectorStoreError
from app.core.logging import get_logger
from app.models.chunk import ChildChunk, ChunkMetadata, ParentChunk
from app.models.search import SearchResult
from app.services import storage

log = get_logger(__name__)


def _child_metadata(child: ChildChunk) -> ChunkMetadata:
    """Flatten a child into the metadata stored alongside its vector.

    Document metadata is spread first so the identity fields below always win: a
    document whose own metadata happened to carry ``document_id`` must not be able to
    shadow the real one and break the search filter — or carry a ``parent_id`` pointing
    retrieval at someone else's text.

    ``None`` values are dropped. Chroma discards them itself, but doing it here makes the
    stored shape explicit and version-independent: a DOCX chunk simply carries no
    ``page_count`` key, and consumers test for presence rather than a sentinel.
    """
    metadata: ChunkMetadata = {
        **child.metadata,
        "parent_id": child.parent_id,
        "document_id": child.document_id,
        "chunk_index": child.chunk_index,
        "start_index": child.start_index,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _parent_result(parent: ParentChunk, score: float) -> SearchResult:
    """Present a parent as a search result, scored by its best matching child."""
    return SearchResult(
        chunk_id=parent.id,
        document_id=parent.document_id,
        chunk_index=parent.chunk_index,
        text=parent.text,
        score=score,
        metadata={**parent.metadata, "start_index": parent.start_index},
    )


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

    def add_children(self, children: list[ChildChunk]) -> int:
        """Embed and store ``children``, returning how many were written.

        Only children are embedded; their parents live in the parent store. Ids are the
        children's deterministic ``{document_id}:p{i}:c{j}``, so adding the same children
        again overwrites them rather than duplicating. Note this is idempotent *per
        document id*: ``/ingest`` mints a fresh uuid per upload, so re-uploading the same
        file still produces a second copy. Content-based dedup is Phase 9.
        """
        if not children:
            return 0
        try:
            self._store.add_texts(
                texts=[child.text for child in children],
                metadatas=[_child_metadata(child) for child in children],
                ids=[child.id for child in children],
            )
        except Exception as exc:
            raise VectorStoreError("Could not store document vectors.") from exc
        return len(children)

    def _child_budget(self, limit: int) -> int:
        """How many children to fetch to have a chance of ``limit`` distinct parents.

        A caller asking for more parents than the child budget could never be satisfied:
        each parent needs at least one matched child to be found at all. Scaling by the
        parent/child size ratio accounts for the reason this phase exists — matched
        children usually cluster in the same parent, so one child per requested parent
        would leave a large request quietly under-delivering after the collapse.

        Capped at MAX_TOP_K so a large request cannot ask the store for an unbounded
        number of vectors.
        """
        children_per_parent = max(
            1, self._settings.parent_chunk_size // self._settings.child_chunk_size
        )
        return min(max(self._settings.retrieval_child_k, limit * children_per_parent), MAX_TOP_K)

    def _best_child_per_parent(
        self, query: str, child_k: int, document_id: str | None
    ) -> dict[str, float]:
        """Match children and collapse them to ``{parent_id: best child score}``.

        Insertion order is the order the best-scoring child of each parent was seen, so
        the mapping is already ranked and equal scores keep a stable, retrieval-order
        tie-break.
        """
        where = None if document_id is None else {"document_id": document_id}
        try:
            matches = self._store.similarity_search_with_score(query, k=child_k, filter=where)
        except Exception as exc:
            raise VectorStoreError("Could not run the similarity search.") from exc

        best: dict[str, float] = {}
        for document, distance in sorted(matches, key=lambda match: match[1]):
            parent_id = str(document.metadata.get("parent_id", ""))
            if not parent_id:
                # A vector written before this phase, or by something else entirely.
                # It cannot be resolved to a parent, so it cannot be answered from.
                log.warning("search.child_without_parent", chunk_id=self._chunk_id(document))
                continue
            # Vectors are normalized and the space is cosine, so distance is
            # 1 - cosine_similarity; inverting it recovers the similarity. Sorted by
            # distance ascending, so the first sighting of a parent is its best child.
            best.setdefault(parent_id, 1.0 - distance)
        return best

    def search(
        self, query: str, top_k_parents: int | None = None, document_id: str | None = None
    ) -> list[SearchResult]:
        """Return the parents whose children best match ``query``, best first.

        Small-to-big: the search runs over small children for a sharp match, then returns
        the larger parents they belong to, so the model reads whole passages rather than
        the fragments that happened to embed well. Children are over-retrieved because
        several matches usually share a parent.

        An explicit ``document_id`` always filters, including the empty string: treating
        a falsy id as "no filter" would silently widen a scoped query to the whole
        corpus, which is the opposite of what the caller asked for.
        """
        limit = top_k_parents if top_k_parents is not None else self._settings.max_parents
        child_k = self._child_budget(limit)
        best_scores = self._best_child_per_parent(query, child_k, document_id)

        # Every candidate is resolved, not just the first `limit`: a parent that cannot
        # be read must cost one source, not truncate the answer. There are at most
        # `child_k` of them, and each is validated individually, so this stays cheap.
        ranked = list(best_scores)
        parents = self._load_ranked_parents(ranked)

        results: list[SearchResult] = []
        for parent_id in ranked:
            if len(results) >= limit:
                break
            parent = parents.get(parent_id)
            if parent is None:
                log.warning("search.parent_unresolved", parent_id=parent_id)
                continue
            results.append(_parent_result(parent, best_scores[parent_id]))
        return results

    def _load_ranked_parents(self, parent_ids: list[str]) -> dict[str, ParentChunk]:
        """Load exactly the parents named by ``parent_ids``, one read per document.

        Batched by document for two reasons: a document contributing several matched
        children would otherwise be read once per child, and only the handful of parents
        actually cited need validating — a 500-page PDF stores hundreds.

        A parents file that is missing or unreadable means the document was deleted
        between the match and the load, or that its vectors outlived it. Neither is worth
        failing the whole query over — the other retrieved parents still answer the
        question — so it degrades to a warning and one fewer source.
        """
        wanted: dict[str, set[str]] = {}
        for parent_id in parent_ids:
            document_id, separator, _ = parent_id.partition(":p")
            if not separator:
                log.warning("search.malformed_parent_id", parent_id=parent_id)
                continue
            wanted.setdefault(document_id, set()).add(parent_id)

        parents: dict[str, ParentChunk] = {}
        for document_id, ids in wanted.items():
            try:
                parents |= storage.load_parents_by_id(
                    storage.validated_document_id(document_id), ids, self._settings
                )
            except DocumentNotFoundError:
                log.warning("search.parents_missing", document_id=document_id)
        return parents

    @staticmethod
    def _chunk_id(document: object) -> str:
        """Recover a child's id, falling back to its metadata if Chroma omits it."""
        identifier = getattr(document, "id", None)
        if identifier:
            return str(identifier)
        metadata = getattr(document, "metadata", {}) or {}
        return f"{metadata.get('parent_id', '')}:c{metadata.get('chunk_index', 0)}"

    def delete_document(self, document_id: str) -> int:
        """Remove every vector belonging to ``document_id``, returning how many went.

        The parents file goes too: parents that outlive their children are unreachable —
        nothing can match them — and would leave the document half-deleted on disk.

        Vectors go first, deliberately. If the vector delete fails after the parents file
        is gone, the surviving children are not merely untidy: they still match, so they
        consume slots in every later query's fixed child budget and then resolve to
        nothing. The opposite half-failure — parents outliving their children — costs
        only disk.

        The delete is issued by filter rather than by the ids just read, so a child
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

        try:
            validated = storage.validated_document_id(document_id)
        except DocumentNotFoundError:
            # Not a uuid, so it never named a parents file — the vector delete above is
            # all there was to do. The filter is data, not a path.
            log.warning("delete.unvalidated_document_id")
            return removed
        if not storage.delete_parents(validated, self._settings) and removed:
            # Children existed, so parents should have too. Worth a distinct event: the
            # document is now deleted from the collection but may still be on disk.
            log.warning("delete.parents_not_removed", document_id=validated)
        return removed

    def count(self) -> int:
        """Number of vectors currently in the collection."""
        return self._collection.count()
