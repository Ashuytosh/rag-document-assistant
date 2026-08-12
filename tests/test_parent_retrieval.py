"""Tests for parent-document (small-to-big) retrieval.

The property under test throughout: what is *matched* is not what is *answered from*.
Children are embedded and searched; parents are returned, deduplicated, and passed to the
model. Chroma is real here, as elsewhere — only the embedding model and the LLM are fakes.
"""

import json
import math
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from app.config import DOCX_CONTENT_TYPE, MAX_TOP_K, PDF_CONTENT_TYPE, Settings
from app.core.exceptions import DocumentNotFoundError
from app.models.chunk import ParentChunk
from app.services import storage
from app.services.chunking import chunk_document, page_for_offset
from app.services.embedding import EmbeddingService
from app.services.generation import GenerationService
from app.services.vector_store import VectorStoreService
from tests.conftest import DOC_ONE, DOC_THREE, DOC_TWO

DOC_METADATA = {"filename": "report.pdf", "content_type": PDF_CONTENT_TYPE, "page_count": 3}
CITATIONS_TEXT = "grounded answers always cite their sources"
WEATHER_TEXT = "the weather forecast predicts rain tomorrow"


def _long_text(paragraphs: int = 40) -> str:
    """Several paragraphs of prose, comfortably longer than one parent chunk."""
    return "\n\n".join(
        f"Paragraph {i} discusses retrieval augmented generation. "
        f"It explains how grounded answers cite their sources carefully."
        for i in range(paragraphs)
    )


@pytest.fixture
def split(test_settings: Settings) -> tuple[list[ParentChunk], list[object]]:
    return chunk_document(_long_text(), DOC_ONE, dict(DOC_METADATA), test_settings)


class _RankedEmbeddings(Embeddings):
    """Cosine order follows the ``#n`` tag in the text: ``#0`` is nearest ``#0``, then ``#1``.

    `FakeEmbeddings` is hash-derived and only guarantees that a query lands nearer to a
    chunk sharing its words. That is enough for "the right passage wins", but not for the
    budget tests below, where the *exact* child ranking is the thing being asserted — with
    hashed vectors a test could pass because the ranking happened to fall the right way.
    """

    @staticmethod
    def _vector(text: str) -> list[float]:
        angle = float(text.rsplit("#", 1)[-1]) * (math.pi / 64)
        return [math.cos(angle), math.sin(angle)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _ranked_store(test_settings: Settings, child_k: int) -> VectorStoreService:
    """A store with a pinned child ranking and an explicit over-retrieval budget."""
    return VectorStoreService(
        test_settings.model_copy(update={"retrieval_child_k": child_k}), _RankedEmbeddings()
    )


class TestChunkingProducesBothLevels:
    def test_a_long_document_yields_several_parents(self, split: tuple) -> None:
        parents, _children = split

        assert len(parents) > 1

    def test_there_are_more_children_than_parents(self, split: tuple) -> None:
        """The whole point: many small vectors standing in for few large passages."""
        parents, children = split

        assert len(children) > len(parents)

    def test_every_child_resolves_to_an_existing_parent(self, split: tuple) -> None:
        parents, children = split
        parent_ids = {parent.id for parent in parents}

        assert children
        for child in children:
            assert child.parent_id in parent_ids

    def test_one_parent_holds_several_children(self, split: tuple) -> None:
        parents, children = split
        per_parent: dict[str, int] = {}
        for child in children:
            per_parent[child.parent_id] = per_parent.get(child.parent_id, 0) + 1

        assert max(per_parent.values()) > 1
        assert set(per_parent) == {parent.id for parent in parents}

    def test_parents_are_larger_than_children(self, split: tuple, test_settings: Settings) -> None:
        parents, children = split

        assert max(child.char_count for child in children) <= test_settings.child_chunk_size
        assert max(parent.char_count for parent in parents) > test_settings.child_chunk_size

    def test_child_offsets_are_absolute_in_the_document(self, test_settings: Settings) -> None:
        """Citations locate a passage in the document, not inside its parent."""
        text = _long_text()
        _parents, children = chunk_document(text, DOC_ONE, dict(DOC_METADATA), test_settings)

        for child in children:
            assert text[child.start_index : child.start_index + child.char_count] == child.text

    def test_ids_encode_the_parent_child_relationship(self, split: tuple) -> None:
        parents, children = split

        assert parents[0].id == f"{DOC_ONE}:p0"
        assert children[0].id == f"{DOC_ONE}:p0:c0"
        for child in children:
            assert child.id.startswith(f"{child.parent_id}:c")

    def test_short_text_still_yields_one_parent_and_one_child(
        self, test_settings: Settings
    ) -> None:
        parents, children = chunk_document(
            "A short sentence.", DOC_ONE, dict(DOC_METADATA), test_settings
        )

        assert len(parents) == 1
        assert len(children) == 1
        assert children[0].parent_id == parents[0].id


class TestSearchCollapsesChildrenToParents:
    def test_a_matching_child_returns_its_parents_text(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """The answer must be written from the whole passage, not the matched fragment."""
        parent_text = f"{CITATIONS_TEXT}. {WEATHER_TEXT}. Everything about the subject."
        index_parent(
            vector_store,
            DOC_ONE,
            0,
            parent_text,
            child_texts=[CITATIONS_TEXT, WEATHER_TEXT],
        )

        results = vector_store.search(CITATIONS_TEXT, top_k_parents=5)

        assert len(results) == 1
        assert results[0].text == parent_text
        assert results[0].chunk_id == f"{DOC_ONE}:p0"

    def test_two_matched_children_of_one_parent_return_it_once(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """Deduplication is what keeps the same passage out of the prompt twice."""
        index_parent(
            vector_store,
            DOC_ONE,
            0,
            f"{CITATIONS_TEXT} and more: {CITATIONS_TEXT}",
            child_texts=[CITATIONS_TEXT, CITATIONS_TEXT + " again"],
        )

        results = vector_store.search(CITATIONS_TEXT, top_k_parents=5)

        assert len(results) == 1

    def test_a_parent_is_scored_by_its_best_child(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """A weak child alongside an exact one must not drag the passage down the ranking."""
        index_parent(
            vector_store,
            DOC_ONE,
            0,
            f"{CITATIONS_TEXT} {WEATHER_TEXT}",
            child_texts=[CITATIONS_TEXT, WEATHER_TEXT],
        )

        result = vector_store.search(CITATIONS_TEXT, top_k_parents=1)[0]

        assert result.score == pytest.approx(1.0, abs=1e-4)

    def test_results_are_ranked_by_best_child_score(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        index_parent(vector_store, DOC_ONE, 0, WEATHER_TEXT)
        index_parent(vector_store, DOC_ONE, 1, CITATIONS_TEXT)

        results = vector_store.search(CITATIONS_TEXT, top_k_parents=2)

        assert results[0].chunk_id == f"{DOC_ONE}:p1"
        assert results[0].score > results[1].score

    def test_the_parent_limit_caps_the_result_count(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        for i in range(6):
            index_parent(vector_store, DOC_ONE, i, f"passage {i} about sources")

        assert len(vector_store.search("sources", top_k_parents=2)) == 2

    def test_the_configured_maximum_applies_when_no_limit_is_given(
        self,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
        test_settings: Settings,
    ) -> None:
        """`max_parents` bounds how much prompt text one question can pull in."""
        for i in range(test_settings.max_parents + 3):
            index_parent(vector_store, DOC_ONE, i, f"passage {i} about sources")

        assert len(vector_store.search("sources")) == test_settings.max_parents

    def test_the_document_filter_restricts_results(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)
        index_parent(vector_store, DOC_TWO, 0, CITATIONS_TEXT)

        results = vector_store.search(CITATIONS_TEXT, top_k_parents=5, document_id=DOC_TWO)

        assert [result.document_id for result in results] == [DOC_TWO]

    def test_children_are_over_retrieved_beyond_the_parent_limit(
        self,
        test_settings: Settings,
        embedding_service: object,
        index_parent: Callable[..., ParentChunk],
    ) -> None:
        """Asking for 2 parents must not stop after 2 children, or duplicates starve it.

        With every child of one parent ranking above another document's, a search that
        fetched only `top_k_parents` children would collapse them to a single parent and
        return one result where two exist.
        """
        store = VectorStoreService(
            test_settings.model_copy(update={"retrieval_child_k": 15}),
            embedding_service.as_langchain(),  # type: ignore[attr-defined]
        )
        index_parent(
            store,
            DOC_ONE,
            0,
            "sources sources sources",
            child_texts=["sources one", "sources two", "sources three"],
        )
        index_parent(store, DOC_TWO, 0, "sources elsewhere")

        assert len(store.search("sources", top_k_parents=2)) == 2


class TestMissingParents:
    def test_a_child_whose_parent_file_is_gone_is_skipped(
        self,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
        test_settings: Settings,
    ) -> None:
        """A deleted document must cost one source, not the whole query."""
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)
        index_parent(vector_store, DOC_TWO, 0, CITATIONS_TEXT)
        storage.parents_path(DOC_ONE, test_settings).unlink()

        results = vector_store.search(CITATIONS_TEXT, top_k_parents=5)

        assert [result.document_id for result in results] == [DOC_TWO]

    def test_every_parent_missing_returns_nothing_rather_than_raising(
        self,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
        test_settings: Settings,
    ) -> None:
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)
        storage.parents_path(DOC_ONE, test_settings).unlink()

        assert vector_store.search(CITATIONS_TEXT, top_k_parents=5) == []


class TestParentStore:
    def test_round_trips_through_disk(self, test_settings: Settings) -> None:
        parents, _children = chunk_document(
            _long_text(), DOC_ONE, dict(DOC_METADATA), test_settings
        )
        storage.save_parents(DOC_ONE, parents, test_settings)

        loaded = storage.load_parents(DOC_ONE, test_settings)

        assert loaded == {parent.id: parent for parent in parents}

    def test_load_parent_resolves_a_single_id(self, test_settings: Settings) -> None:
        parents, _children = chunk_document(
            _long_text(), DOC_ONE, dict(DOC_METADATA), test_settings
        )
        storage.save_parents(DOC_ONE, parents, test_settings)

        assert storage.load_parent(f"{DOC_ONE}:p1", test_settings) == parents[1]

    @pytest.mark.parametrize(
        "parent_id",
        [
            "not-an-id",
            "..\\..\\secrets:p0",
            "../../secrets:p0",
            f"{DOC_ONE}",
            f"urn:uuid:{DOC_ONE}:p0",
        ],
    )
    def test_a_malformed_parent_id_is_refused(
        self, parent_id: str, test_settings: Settings
    ) -> None:
        """The id arrives from vector metadata and is interpolated into a path."""
        with pytest.raises(DocumentNotFoundError):
            storage.load_parent(parent_id, test_settings)

    def test_an_unknown_parent_of_a_known_document_is_refused(
        self, test_settings: Settings
    ) -> None:
        parents, _children = chunk_document("short", DOC_ONE, dict(DOC_METADATA), test_settings)
        storage.save_parents(DOC_ONE, parents, test_settings)

        with pytest.raises(DocumentNotFoundError):
            storage.load_parent(f"{DOC_ONE}:p99", test_settings)

    def test_delete_parents_removes_the_file(self, test_settings: Settings) -> None:
        storage.save_parents(DOC_ONE, [], test_settings)

        assert storage.delete_parents(DOC_ONE, test_settings) is True
        assert storage.delete_parents(DOC_ONE, test_settings) is False


class TestGenerationUsesParents:
    def test_the_context_carries_parent_text_not_child_text(
        self,
        generation_service: GenerationService,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
    ) -> None:
        """The model must see the surrounding sentences, not the fragment that matched."""
        surrounding = "Context before. " + CITATIONS_TEXT + " Context after."
        index_parent(vector_store, DOC_ONE, 0, surrounding, page=4, child_texts=[CITATIONS_TEXT])

        context, sources, _nonce = generation_service.retrieve_and_build(CITATIONS_TEXT, top_k=1)

        assert surrounding in context
        assert sources[0].snippet.startswith("Context before.")

    def test_citations_describe_the_parent(
        self,
        generation_service: GenerationService,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
    ) -> None:
        index_parent(vector_store, DOC_ONE, 2, CITATIONS_TEXT, page=4)

        _context, sources, _nonce = generation_service.retrieve_and_build(CITATIONS_TEXT, top_k=1)

        assert sources[0].chunk_id == f"{DOC_ONE}:p2"
        assert sources[0].document_id == DOC_ONE
        assert sources[0].filename == "report.pdf"
        assert sources[0].page == 4
        assert sources[0].start_index == 200


class TestIngestionEndToEnd:
    def _ingest(self, client: TestClient, pdf: bytes, name: str = "sample.pdf") -> dict:
        response = client.post("/ingest", files={"file": (name, pdf, PDF_CONTENT_TYPE)})
        assert response.status_code == 200, response.text
        return response.json()

    def test_the_response_reports_both_counts(
        self, client: TestClient, long_docx_bytes: bytes
    ) -> None:
        response = client.post(
            "/ingest",
            files={
                "file": (
                    "long.docx",
                    long_docx_bytes,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

        body = response.json()
        assert body["parent_count"] > 1
        assert body["child_count"] > body["parent_count"]

    def test_parents_are_persisted_and_children_embedded(
        self, client: TestClient, long_docx_bytes: bytes, upload_dir: Path, test_settings: Settings
    ) -> None:
        response = client.post(
            "/ingest",
            files={
                "file": (
                    "long.docx",
                    long_docx_bytes,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        body = response.json()
        document_id = body["document"]["id"]

        parents = storage.load_parents(document_id, test_settings)

        assert (upload_dir / f"{document_id}.parents.json").exists()
        assert len(parents) == body["parent_count"]
        assert client.get("/stats").json()["total_vectors"] == body["child_count"]

    def test_re_ingesting_the_same_text_overwrites_rather_than_duplicating(
        self, client: TestClient, sample_pdf_bytes: bytes, test_settings: Settings
    ) -> None:
        """Deterministic ids at both levels are what make a re-index idempotent."""
        body = self._ingest(client, sample_pdf_bytes)
        document_id = body["document"]["id"]
        text = (test_settings.upload_dir / f"{document_id}.txt").read_text(encoding="utf-8")
        before = client.get("/stats").json()["total_vectors"]

        parents, children = chunk_document(text, document_id, dict(DOC_METADATA), test_settings)
        storage.save_parents(document_id, parents, test_settings)
        client.app.state.vector_store.add_children(children)

        assert client.get("/stats").json()["total_vectors"] == before
        assert len(storage.load_parents(document_id, test_settings)) == body["parent_count"]

    def test_deleting_a_document_removes_its_parents_file_too(
        self, client: TestClient, sample_pdf_bytes: bytes, test_settings: Settings
    ) -> None:
        """Parents that outlive their children are unreachable, not just untidy."""
        document_id = self._ingest(client, sample_pdf_bytes)["document"]["id"]

        client.app.state.vector_store.delete_document(document_id)

        assert not storage.parents_path(document_id, test_settings).exists()
        with pytest.raises(DocumentNotFoundError):
            storage.load_parents(document_id, test_settings)

    def test_a_question_is_answered_from_the_whole_passage(
        self, client: TestClient, long_docx_bytes: bytes
    ) -> None:
        """End to end: ingest, ask, and get parent-sized context back with citations."""
        client.post(
            "/ingest",
            files={
                "file": (
                    "long.docx",
                    long_docx_bytes,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

        body = client.post("/query", json={"query": "word", "stream": False}).json()

        assert body["sources"]
        assert len(body["sources"]) <= 4
        assert all(":p" in source["chunk_id"] for source in body["sources"])


class TestOverRetrievalBudget:
    """`retrieval_child_k` is the only thing standing between dedup and a starved answer.

    Children are fetched first and collapsed second, so the budget is spent in child units
    but the caller counts parents. Every test here pins the child ranking with
    `_RankedEmbeddings`, because "how many children were fetched" is only observable
    through which parents survive the collapse.
    """

    def test_the_budget_is_the_configured_one_when_it_exceeds_the_parent_limit(
        self,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Asking for 1 parent must still fetch `retrieval_child_k` children, not 1.

        Fetching only `limit` children would make dedup a no-op and quietly turn the phase
        back into flat retrieval — which nothing else observes when the corpus happens to
        hold one parent per match.
        """
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)
        seen: list[int] = []
        real = vector_store._store.similarity_search_with_score

        def spy(query: str, k: int = 4, **kwargs: Any) -> Any:
            seen.append(k)
            return real(query, k=k, **kwargs)

        monkeypatch.setattr(vector_store._store, "similarity_search_with_score", spy)

        vector_store.search(CITATIONS_TEXT, top_k_parents=1)

        assert seen == [15]

    def test_asking_for_more_parents_than_the_budget_raises_the_budget(
        self, test_settings: Settings, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """A limit above `retrieval_child_k` could otherwise never be satisfied.

        Each parent needs at least one matched child to be found at all, so a budget of two
        children would cap the answer at two parents however many the caller asked for.
        """
        store = _ranked_store(test_settings, child_k=2)
        for index in range(5):
            index_parent(store, DOC_ONE, index, f"passage {index}", child_texts=[f"#{index}"])

        assert len(store.search("#0", top_k_parents=5)) == 5

    def test_the_budget_scales_with_the_requested_parent_count(
        self, test_settings: Settings, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """Asking for N parents fetches roughly N parents' worth of children.

        One child per requested parent would be enough only if matches never clustered —
        the opposite of the assumption this whole phase rests on. A request for 2 parents
        must survive the first parent contributing several of the top children.
        """
        store = _ranked_store(test_settings, child_k=3)
        index_parent(store, DOC_ONE, 0, "dense passage", child_texts=["#0", "#1", "#2"])
        index_parent(store, DOC_TWO, 0, "other passage", child_texts=["#3"])

        results = store.search("#0", top_k_parents=2)

        assert [result.document_id for result in results] == [DOC_ONE, DOC_TWO]

    def test_the_budget_still_bounds_how_many_distinct_parents_can_be_found(
        self, test_settings: Settings, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """The cost of finite over-retrieval: a dense enough parent still crowds another out.

        Scaling raises the ceiling; it does not remove it. This is the trade-off the budget
        encodes, and it is worth pinning so that changing it is a visible decision.
        """
        store = _ranked_store(test_settings, child_k=3)
        index_parent(store, DOC_ONE, 0, "dense passage", child_texts=[f"#{i}" for i in range(10)])
        index_parent(store, DOC_TWO, 0, "other passage", child_texts=["#20"])

        results = store.search("#0", top_k_parents=2)

        assert [result.document_id for result in results] == [DOC_ONE]

    def test_the_budget_is_capped_so_a_large_request_cannot_scan_the_collection(
        self,
        test_settings: Settings,
        index_parent: Callable[..., ParentChunk],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Scaling multiplies the request; MAX_TOP_K is what stops it running away."""
        store = _ranked_store(test_settings, child_k=15)
        index_parent(store, DOC_ONE, 0, "passage", child_texts=["#0"])
        seen: list[int] = []
        real = store._store.similarity_search_with_score

        def spy(query: str, k: int = 4, **kwargs: Any) -> Any:
            seen.append(k)
            return real(query, k=k, **kwargs)

        monkeypatch.setattr(store._store, "similarity_search_with_score", spy)

        store.search("#0", top_k_parents=MAX_TOP_K)

        assert seen == [MAX_TOP_K]

    def test_a_budget_one_larger_recovers_the_crowded_out_parent(
        self, test_settings: Settings, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """The same corpus with one more child of budget: the second parent reappears.

        Paired with the test above, this shows the starvation is the budget's doing and not
        the ranking's — the only thing that changed is `retrieval_child_k`.
        """
        store = _ranked_store(test_settings, child_k=4)
        index_parent(store, DOC_ONE, 0, "dense passage", child_texts=["#0", "#1", "#2"])
        index_parent(store, DOC_TWO, 0, "other passage", child_texts=["#3"])

        results = store.search("#0", top_k_parents=2)

        assert [result.document_id for result in results] == [DOC_ONE, DOC_TWO]

    def test_the_budget_is_spent_inside_the_document_filter(
        self, test_settings: Settings, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """The filter must reach Chroma, not be applied after the children come back.

        Filtering afterwards would let a loud out-of-scope document consume the whole child
        budget and leave a scoped query with no sources at all. The failure mode is an
        empty answer rather than a wrong one, so nothing else would flag it.
        """
        store = _ranked_store(test_settings, child_k=3)
        index_parent(store, DOC_TWO, 0, "loud passage", child_texts=["#0", "#1", "#2", "#3"])
        index_parent(store, DOC_ONE, 0, "quiet passage", child_texts=["#40"])

        results = store.search("#0", top_k_parents=2, document_id=DOC_ONE)

        assert [result.chunk_id for result in results] == [f"{DOC_ONE}:p0"]

    def test_the_filter_still_collapses_children_to_one_parent(
        self, test_settings: Settings, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """Scoping a query must not reintroduce the duplicate passages dedup removes."""
        store = _ranked_store(test_settings, child_k=10)
        index_parent(store, DOC_ONE, 0, "passage", child_texts=["#0", "#1", "#2"])
        index_parent(store, DOC_TWO, 0, "elsewhere", child_texts=["#3"])

        results = store.search("#0", top_k_parents=5, document_id=DOC_ONE)

        assert [result.chunk_id for result in results] == [f"{DOC_ONE}:p0"]


class TestOrderingAndTies:
    def test_equally_scoring_parents_are_both_returned(
        self, test_settings: Settings, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """Two parents matched by identical children are two sources, not one.

        Dedup is keyed on parent id; keying it on the score or on the text — a plausible
        simplification — would silently drop a genuine second passage.
        """
        store = _ranked_store(test_settings, child_k=10)
        index_parent(store, DOC_ONE, 0, "first passage", child_texts=["#0"])
        index_parent(store, DOC_TWO, 0, "second passage", child_texts=["#0"])

        results = store.search("#0", top_k_parents=5)

        assert {result.document_id for result in results} == {DOC_ONE, DOC_TWO}
        assert results[0].score == pytest.approx(results[1].score, abs=1e-9)

    def test_a_tie_resolves_the_same_way_on_every_call(
        self, test_settings: Settings, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """Identical scores must not make the source list shuffle between two identical asks.

        Insertion order is the order each parent's best child was seen, so a tie inherits
        Chroma's stable retrieval order rather than set or dict iteration order.
        """
        store = _ranked_store(test_settings, child_k=10)
        for document_id in (DOC_ONE, DOC_TWO, DOC_THREE):
            index_parent(store, document_id, 0, f"passage {document_id}", child_texts=["#0"])

        orders = {
            tuple(result.chunk_id for result in store.search("#0", top_k_parents=3))
            for _ in range(5)
        }

        assert len(orders) == 1

    def test_a_tie_truncated_by_the_parent_limit_keeps_the_same_winner(
        self, test_settings: Settings, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """When the limit cuts through a tie, which parent survives must still be stable."""
        store = _ranked_store(test_settings, child_k=10)
        for document_id in (DOC_ONE, DOC_TWO, DOC_THREE):
            index_parent(store, document_id, 0, f"passage {document_id}", child_texts=["#0"])

        winners = {store.search("#0", top_k_parents=1)[0].chunk_id for _ in range(5)}

        assert len(winners) == 1

    def test_scores_are_returned_in_non_increasing_order(
        self, test_settings: Settings, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """Collapsing matches into a dict must not lose the ranking the sort established."""
        store = _ranked_store(test_settings, child_k=20)
        for index in range(6):
            index_parent(store, DOC_ONE, index, f"passage {index}", child_texts=[f"#{index * 3}"])

        scores = [result.score for result in store.search("#0", top_k_parents=6)]

        assert scores == sorted(scores, reverse=True)

    def test_a_late_child_cannot_demote_a_parent_already_seen(
        self, test_settings: Settings, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """A parent ranks by its best child, and its position is fixed when first seen.

        Interleaving a weak child of the top parent below a strong child of another must
        not reorder them; `setdefault` over a distance-sorted list is what guarantees it.
        """
        store = _ranked_store(test_settings, child_k=10)
        index_parent(store, DOC_ONE, 0, "best passage", child_texts=["#0", "#40"])
        index_parent(store, DOC_TWO, 0, "runner up", child_texts=["#5"])

        results = store.search("#0", top_k_parents=2)

        assert [result.document_id for result in results] == [DOC_ONE, DOC_TWO]
        assert results[0].score > results[1].score


class TestUnresolvableChildren:
    """Vectors that cannot be traced back to answerable text.

    A child is useful only because its `parent_id` names text on disk. Anything breaking
    that link — a pre-phase-6 vector, a blank id, a hand-crafted one — must cost one source
    rather than the query, and must never reach the filesystem unchecked.
    """

    @staticmethod
    def _add_raw(store: VectorStoreService, text: str, metadata: dict[str, Any], id_: str) -> None:
        """Write a vector straight to the collection, bypassing `add_children`.

        `ChildChunk` requires a `parent_id`, so a child without one cannot be built through
        the normal path — but a collection written by an earlier phase is full of them.
        """
        store._store.add_texts(texts=[text], metadatas=[metadata], ids=[id_])

    def test_a_vector_written_before_this_phase_is_skipped(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """Phase 2 embedded whole chunks with no `parent_id`; those cannot be answered from.

        Falling back to such a vector's own text would reintroduce flat retrieval for
        exactly the documents indexed before the upgrade, with no signal it had happened.
        """
        self._add_raw(
            vector_store,
            CITATIONS_TEXT,
            {"document_id": DOC_TWO, "chunk_index": 0, "start_index": 0},
            f"{DOC_TWO}:legacy0",
        )
        index_parent(vector_store, DOC_ONE, 0, f"{CITATIONS_TEXT} in context")

        results = vector_store.search(CITATIONS_TEXT, top_k_parents=5)

        assert [result.document_id for result in results] == [DOC_ONE]

    def test_a_collection_of_only_legacy_vectors_answers_nothing(
        self, vector_store: VectorStoreService
    ) -> None:
        """Abstaining beats answering from text that no citation could describe."""
        self._add_raw(vector_store, CITATIONS_TEXT, {"document_id": DOC_ONE}, f"{DOC_ONE}:legacy0")

        assert vector_store.search(CITATIONS_TEXT, top_k_parents=5) == []

    def test_a_blank_parent_id_is_skipped(self, vector_store: VectorStoreService) -> None:
        """An empty string is present-but-useless; a bare `.get` check would not catch it."""
        self._add_raw(
            vector_store,
            CITATIONS_TEXT,
            {"document_id": DOC_ONE, "parent_id": ""},
            f"{DOC_ONE}:blank0",
        )

        assert vector_store.search(CITATIONS_TEXT, top_k_parents=5) == []

    def test_a_parent_id_with_no_document_prefix_is_skipped(
        self, vector_store: VectorStoreService
    ) -> None:
        """`partition(':p')` finds no separator, so nothing is interpolated into a path."""
        self._add_raw(
            vector_store,
            CITATIONS_TEXT,
            {"document_id": DOC_ONE, "parent_id": "not-a-parent-id"},
            f"{DOC_ONE}:odd0",
        )

        assert vector_store.search(CITATIONS_TEXT, top_k_parents=5) == []

    @pytest.mark.parametrize(
        "parent_id",
        [
            "..\\..\\secrets:p0",
            "../../secrets:p0",
            f"urn:uuid:{DOC_ONE}:p0",
            f"{DOC_ONE} :p0",
        ],
    )
    def test_a_traversal_shaped_parent_id_cannot_reach_the_filesystem(
        self, vector_store: VectorStoreService, upload_dir: Path, parent_id: str
    ) -> None:
        """Parent ids arrive from vector metadata, which is no more trustworthy than input.

        `load_parent` is already guarded and tested; this pins the same guard on the
        retrieval path, which reaches `load_parents` by a different route.
        """
        (upload_dir.parent / "secrets.parents.json").write_text("[]", encoding="utf-8")
        self._add_raw(
            vector_store,
            CITATIONS_TEXT,
            {"document_id": DOC_ONE, "parent_id": parent_id},
            f"{DOC_ONE}:evil0",
        )

        assert vector_store.search(CITATIONS_TEXT, top_k_parents=5) == []

    def test_an_unresolvable_child_does_not_consume_the_parent_budget(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """A skipped parent must cost a source, not a slot.

        Counting it against `top_k_parents` would silently shrink the context of every
        query that happened to match a stale vector first.
        """
        self._add_raw(
            vector_store,
            CITATIONS_TEXT,
            {"document_id": DOC_TWO, "parent_id": f"{DOC_TWO}:p404"},
            f"{DOC_TWO}:ghost0",
        )
        index_parent(vector_store, DOC_ONE, 0, f"{CITATIONS_TEXT} one")
        index_parent(vector_store, DOC_ONE, 1, f"{CITATIONS_TEXT} two")

        results = vector_store.search(CITATIONS_TEXT, top_k_parents=2)

        assert len(results) == 2


class TestCorruptParentStore:
    """A parents file that is present but unreadable, as distinct from absent.

    The missing-file path is already covered. Corruption is the more likely production
    case — a half-written file, a truncated copy, the wrong file — and it takes a different
    branch in `_read_parents`, so it needs its own coverage on the retrieval path.
    """

    @staticmethod
    def _corrupt(document_id: str, settings: Settings, payload: bytes) -> None:
        storage.parents_path(document_id, settings).write_bytes(payload)

    @pytest.mark.parametrize(
        ("name", "payload"),
        [
            ("truncated", b'[{"id": "x", "document_id"'),
            ("not_a_list", b'{"id": "x"}'),
            ("empty_file", b""),
            ("not_utf8", b'[{"id": "\xff\xfe"}]'),
            ("wrong_shape", b'[{"unexpected": true}]'),
        ],
    )
    def test_a_corrupt_parents_file_costs_one_source_not_the_query(
        self,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
        test_settings: Settings,
        name: str,
        payload: bytes,
    ) -> None:
        """The other retrieved passages still answer the question."""
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)
        index_parent(vector_store, DOC_TWO, 0, CITATIONS_TEXT)
        self._corrupt(DOC_ONE, test_settings, payload)

        results = vector_store.search(CITATIONS_TEXT, top_k_parents=5)

        assert [result.document_id for result in results] == [DOC_TWO], name

    def test_a_corrupt_file_for_every_document_returns_nothing_rather_than_raising(
        self,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
        test_settings: Settings,
    ) -> None:
        """A 500 here would blame the caller's question for a fault on disk."""
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)
        self._corrupt(DOC_ONE, test_settings, b"not json at all")

        assert vector_store.search(CITATIONS_TEXT, top_k_parents=5) == []

    def test_an_oversize_parents_file_is_refused_during_retrieval(
        self,
        test_settings: Settings,
        embedding_service: EmbeddingService,
        index_parent: Callable[..., ParentChunk],
    ) -> None:
        """The memory guard must hold on the query path too, not only on `/chunks`.

        Retrieval reads the whole parents file on every query, so this is the path where an
        unbounded read would actually be repeated under load.
        """
        settings = test_settings.model_copy(update={"max_chunks_file_bytes": 32})
        store = VectorStoreService(settings, embedding_service.as_langchain())
        index_parent(store, DOC_ONE, 0, CITATIONS_TEXT * 20)

        assert store.search(CITATIONS_TEXT, top_k_parents=5) == []

    def test_an_unreadable_document_is_read_once_not_once_per_matched_child(
        self,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
        test_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reads are batched per document, so a bad file is parsed once, not once per match.

        Both children of the parent match, so an implementation that resolved each match
        independently would read — and fail to parse — the same file twice.
        """
        index_parent(
            vector_store, DOC_ONE, 0, CITATIONS_TEXT, child_texts=[CITATIONS_TEXT, "sources again"]
        )
        self._corrupt(DOC_ONE, test_settings, b"not json")
        reads: list[str] = []
        real = storage.load_parents_by_id

        def counting(
            document_id: str, wanted: set[str], settings: Settings
        ) -> dict[str, ParentChunk]:
            reads.append(document_id)
            return real(document_id, wanted, settings)

        monkeypatch.setattr("app.services.vector_store.storage.load_parents_by_id", counting)

        vector_store.search(CITATIONS_TEXT, top_k_parents=5)

        assert reads == [DOC_ONE]

    def test_a_file_holding_another_documents_parents_yields_nothing(
        self,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
        test_settings: Settings,
    ) -> None:
        """Valid JSON, valid models, wrong ids — the lookup must miss rather than mismatch.

        Answering from a parent whose id does not match the matched child would attach a
        citation to text the query never retrieved.
        """
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)
        stolen = ParentChunk(
            id=f"{DOC_TWO}:p0",
            document_id=DOC_TWO,
            chunk_index=0,
            text="someone else's passage",
            char_count=22,
            start_index=0,
        )
        storage.parents_path(DOC_ONE, test_settings).write_text(
            json.dumps([stolen.model_dump()]), encoding="utf-8"
        )

        assert vector_store.search(CITATIONS_TEXT, top_k_parents=5) == []


class TestParentStoreEncoding:
    def test_non_ascii_parent_text_round_trips_through_disk(self, test_settings: Settings) -> None:
        """`ensure_ascii=False` plus explicit utf-8 on both sides; either half alone corrupts."""
        text = "Les réponses citent leurs sources — 引用 — ✅ 🌍"
        parent = ParentChunk(
            id=f"{DOC_ONE}:p0",
            document_id=DOC_ONE,
            chunk_index=0,
            text=text,
            char_count=len(text),
            start_index=0,
            metadata={"filename": "rapport-café.pdf"},
        )
        storage.save_parents(DOC_ONE, [parent], test_settings)

        loaded = storage.load_parents(DOC_ONE, test_settings)

        assert loaded[parent.id].text == text
        assert loaded[parent.id].metadata["filename"] == "rapport-café.pdf"

    def test_non_ascii_parent_text_survives_retrieval(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """The text reaching the model comes off disk, so encoding is a retrieval concern."""
        text = f"Contexte avant — {CITATIONS_TEXT} — 引用 🌍"
        index_parent(vector_store, DOC_ONE, 0, text, child_texts=[CITATIONS_TEXT])

        assert vector_store.search(CITATIONS_TEXT, top_k_parents=1)[0].text == text

    def test_a_saved_parents_file_holds_utf8_rather_than_escapes(
        self, test_settings: Settings
    ) -> None:
        """Escaped ASCII would still load, but leaves the file unreadable to an operator."""
        parent = ParentChunk(
            id=f"{DOC_ONE}:p0",
            document_id=DOC_ONE,
            chunk_index=0,
            text="引用",
            char_count=2,
            start_index=0,
        )
        storage.save_parents(DOC_ONE, [parent], test_settings)

        raw = storage.parents_path(DOC_ONE, test_settings).read_bytes()

        assert "引用".encode() in raw

    def test_a_successful_save_leaves_no_staging_file_behind(
        self, test_settings: Settings, upload_dir: Path
    ) -> None:
        """The write-then-replace must not be observable once it has succeeded."""
        parents, _children = chunk_document(
            _long_text(), DOC_ONE, dict(DOC_METADATA), test_settings
        )
        storage.save_parents(DOC_ONE, parents, test_settings)

        assert list(upload_dir.glob("*.tmp")) == []


class TestChildOffsetsAndPages:
    """Citations are worth having only if the offset and page they carry are right.

    Children are where this can go wrong: their offsets are computed inside a parent and
    then rebased, so an error there is invisible to the parent-level assertions.
    """

    @staticmethod
    def _fine_settings(test_settings: Settings) -> Settings:
        """Small chunks, so a handful of paragraphs still spans several parents and pages."""
        return test_settings.model_copy(
            update={
                "parent_chunk_size": 200,
                "parent_chunk_overlap": 0,
                "child_chunk_size": 50,
                "child_chunk_overlap": 0,
            }
        )

    def test_child_offsets_locate_non_ascii_text_exactly(self, test_settings: Settings) -> None:
        """Offsets are code-point indices; a byte-based one drifts on the first accent.

        The drift is silent — every snippet stays plausible and every one is wrong — so it
        needs an exact slice assertion over text that is not ASCII.
        """
        text = "\n\n".join(
            f"Paragraphe {i} : les réponses citent leurs sources — 引用 🌍 ✅." for i in range(30)
        )

        _parents, children = chunk_document(text, DOC_ONE, dict(DOC_METADATA), test_settings)

        assert children
        for child in children:
            assert text[child.start_index : child.start_index + child.char_count] == child.text

    def test_parent_offsets_locate_non_ascii_text_exactly(self, test_settings: Settings) -> None:
        text = "\n\n".join(f"Paragraphe {i} — 引用 🌍 sur les sources citées." for i in range(30))

        parents, _children = chunk_document(text, DOC_ONE, dict(DOC_METADATA), test_settings)

        for parent in parents:
            assert text[parent.start_index : parent.start_index + parent.char_count] == parent.text

    def test_every_child_lies_inside_its_own_parent(self, test_settings: Settings) -> None:
        """A rebased offset that used the wrong parent would still slice valid-looking text."""
        text = _long_text()
        parents, children = chunk_document(text, DOC_ONE, dict(DOC_METADATA), test_settings)
        by_id = {parent.id: parent for parent in parents}

        for child in children:
            parent = by_id[child.parent_id]
            assert parent.start_index <= child.start_index
            assert child.start_index + child.char_count <= parent.start_index + parent.char_count

    def test_a_child_is_stamped_with_the_page_it_starts_on(self, test_settings: Settings) -> None:
        settings = self._fine_settings(test_settings)
        text = "\n\n".join(f"Paragraph {i} about grounded sources." for i in range(20))
        page_offsets = [0, 120, 300, 500]

        _parents, children = chunk_document(
            text, DOC_ONE, dict(DOC_METADATA), settings, page_offsets
        )

        assert children
        for child in children:
            assert child.metadata["page"] == page_for_offset(child.start_index, page_offsets)

    def test_children_of_one_parent_can_cite_different_pages(self, test_settings: Settings) -> None:
        """The reason children carry their own page: a large parent spans page breaks.

        Stamping every child with its parent's page would put a citation on the wrong page
        for any passage straddling a boundary — a confidently wrong reference.
        """
        settings = self._fine_settings(test_settings)
        text = "\n\n".join(f"Paragraph {i} about grounded sources." for i in range(20))
        page_offsets = [0, 60, 130, 220, 330]

        parents, children = chunk_document(
            text, DOC_ONE, dict(DOC_METADATA), settings, page_offsets
        )

        pages_per_parent: dict[str, set[object]] = {}
        for child in children:
            pages_per_parent.setdefault(child.parent_id, set()).add(child.metadata["page"])

        assert len(parents) > 1
        assert any(len(pages) > 1 for pages in pages_per_parent.values())

    def test_a_childs_page_is_never_taken_from_a_parent_local_offset(
        self, test_settings: Settings
    ) -> None:
        """Using the local offset would put every later parent's children back on page 1.

        That is the specific mistake `_split_parent` rebases to avoid, and it only shows up
        from the second parent onwards.
        """
        settings = self._fine_settings(test_settings)
        text = "\n\n".join(f"Paragraph {i} about grounded sources." for i in range(20))
        page_offsets = [0, 150, 320]

        parents, children = chunk_document(
            text, DOC_ONE, dict(DOC_METADATA), settings, page_offsets
        )
        last = parents[-1].id
        pages = {child.metadata["page"] for child in children if child.parent_id == last}

        assert pages
        assert 1 not in pages

    def test_a_format_without_pages_stamps_no_page_on_children(
        self, test_settings: Settings
    ) -> None:
        """A DOCX has no page information; defaulting to 1 would be a fabricated citation."""
        _parents, children = chunk_document(
            _long_text(), DOC_ONE, dict(DOC_METADATA), test_settings, None
        )

        assert children
        assert all("page" not in child.metadata for child in children)


class TestConcurrency:
    def test_concurrent_ingests_each_keep_their_own_parents_file(
        self, client: TestClient, sample_pdf_bytes: bytes, test_settings: Settings
    ) -> None:
        """Parents are per-document files written from a thread pool under real concurrency.

        The existing concurrency test counts vectors only; parents live on disk, where a
        race shows up as a missing or cross-contaminated file rather than a wrong count.
        """
        ingests = 5

        def ingest(index: int) -> dict[str, Any]:
            response = client.post(
                "/ingest", files={"file": (f"doc{index}.pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)}
            )
            assert response.status_code == 200, response.text
            return response.json()

        with ThreadPoolExecutor(max_workers=ingests) as pool:
            bodies = list(pool.map(ingest, range(ingests)))

        assert len({body["document"]["id"] for body in bodies}) == ingests
        for body in bodies:
            document_id = body["document"]["id"]
            parents = storage.load_parents(document_id, test_settings)
            assert len(parents) == body["parent_count"]
            assert all(parent.document_id == document_id for parent in parents.values())

    def test_concurrent_ingests_leave_no_staging_files(
        self, client: TestClient, sample_pdf_bytes: bytes, upload_dir: Path
    ) -> None:
        """Each document stages under its own name, so parallel saves cannot collide."""

        def ingest(index: int) -> int:
            return client.post(
                "/ingest", files={"file": (f"doc{index}.pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)}
            ).status_code

        with ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(pool.map(ingest, range(4)))

        assert statuses == [200] * 4
        assert list(upload_dir.glob("*.tmp")) == []

    def test_a_search_during_a_parents_rewrite_never_sees_a_partial_file(
        self,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
        test_settings: Settings,
    ) -> None:
        """This is what write-then-replace buys: a reader sees the old file or the new one.

        A plain `write_text` would let a concurrent query read a truncated file and drop
        the document from the answer — intermittently, and only under load.
        """
        text = f"{CITATIONS_TEXT} " * 40
        parent = index_parent(vector_store, DOC_ONE, 0, text)
        stop = threading.Event()
        torn: list[str] = []

        def rewrite() -> None:
            while not stop.is_set():
                try:
                    storage.save_parents(DOC_ONE, [parent], test_settings)
                except OSError:
                    # Windows can refuse the replace while a reader holds the file open.
                    # That is the writer's problem to retry, not a torn read.
                    continue

        writer = threading.Thread(target=rewrite)
        writer.start()
        try:
            for _ in range(60):
                results = vector_store.search(CITATIONS_TEXT, top_k_parents=1)
                if len(results) != 1 or results[0].text != text:
                    torn.append(repr(results))
        finally:
            stop.set()
            writer.join()

        assert torn == []


class TestIngestionRollback:
    """Half-ingested documents are the failure mode this phase makes possible.

    Text, parents, and vectors are three separate writes now. Any one of them surviving a
    failure of the others leaves retrieval able to match something it cannot answer from,
    or holding text nothing can ever match.
    """

    def _docx_upload(self, client: TestClient, body: bytes) -> Any:
        return client.post("/ingest", files={"file": ("long.docx", body, DOCX_CONTENT_TYPE)})

    def test_a_failing_parents_save_leaves_no_vectors(
        self,
        client: TestClient,
        long_docx_bytes: bytes,
        upload_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Vectors are written after parents, so a parents failure must abort before them.

        Vectors without parents are matchable and unanswerable: every query they win
        silently returns one fewer source.
        """

        def boom(document_id: str, parents: list[ParentChunk], settings: Settings) -> Path:
            raise OSError("disk full")

        monkeypatch.setattr("app.services.storage.save_parents", boom)

        with pytest.raises(OSError, match="disk full"):
            self._docx_upload(client, long_docx_bytes)

        assert client.get("/stats").json()["total_vectors"] == 0
        assert list(upload_dir.iterdir()) == []

    def test_a_failing_child_add_removes_the_parents_file(
        self,
        client: TestClient,
        long_docx_bytes: bytes,
        vector_store: VectorStoreService,
        upload_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Parents outliving a failed embed are unreachable text nothing can ever match."""

        def boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("chroma is down")

        monkeypatch.setattr(vector_store._store, "add_texts", boom)

        response = self._docx_upload(client, long_docx_bytes)

        assert response.status_code == 503
        assert list(upload_dir.glob("*.parents.json")) == []
        assert list(upload_dir.iterdir()) == []

    def test_a_rolled_back_document_is_not_retrievable(
        self,
        client: TestClient,
        long_docx_bytes: bytes,
        vector_store: VectorStoreService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The observable contract: a rejected ingest contributes nothing to any answer."""

        def boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("chroma is down")

        monkeypatch.setattr(vector_store._store, "add_texts", boom)
        self._docx_upload(client, long_docx_bytes)

        assert client.post("/search", json={"query": "word"}).json()["count"] == 0
        assert client.get("/stats").json()["total_vectors"] == 0

    def test_a_failed_ingest_after_a_good_one_leaves_the_good_one_intact(
        self,
        client: TestClient,
        sample_pdf_bytes: bytes,
        long_docx_bytes: bytes,
        vector_store: VectorStoreService,
        test_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rollback deletes by document id, so it must not take the rest of the corpus."""
        good_id = client.post(
            "/ingest", files={"file": ("sample.pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)}
        ).json()["document"]["id"]
        before = client.get("/stats").json()["total_vectors"]

        def boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("chroma is down")

        monkeypatch.setattr(vector_store._store, "add_texts", boom)
        self._docx_upload(client, long_docx_bytes)

        assert client.get("/stats").json()["total_vectors"] == before
        assert len(storage.load_parents(good_id, test_settings)) >= 1

    def test_a_failed_staging_write_leaves_no_temp_file(
        self, test_settings: Settings, upload_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash mid-write must leave the directory as it found it, not a stray `.tmp`."""
        real_write_text = Path.write_text

        def fail_midway(self: Path, data: str, **kwargs: Any) -> int:
            if self.suffix == ".tmp":
                real_write_text(self, data[:5], encoding="utf-8")
                raise OSError("disk full")
            return real_write_text(self, data, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_midway)

        with pytest.raises(OSError, match="disk full"):
            storage.save_parents(DOC_ONE, [], test_settings)

        assert list(upload_dir.iterdir()) == []
