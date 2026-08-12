"""Tests for embeddings, the vector store, and the search endpoints.

The vector store here is real ChromaDB in a temp directory; only the embedding model is
faked. That way id-overwrite, cosine scoring, and metadata filtering are verified against
Chroma itself rather than against a stand-in.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from app import main
from app.config import DOCX_CONTENT_TYPE, PDF_CONTENT_TYPE, Settings, get_settings
from app.main import app
from app.models.chunk import ChildChunk, ParentChunk
from app.services.chunking import chunk_document
from app.services.embedding import CHARS_PER_TOKEN_FALLBACK, EmbeddingService
from app.services.vector_store import VectorStoreService, _child_metadata
from tests.conftest import DOC_ONE, DOC_TWO, child_of

CITATIONS_TEXT = "grounded answers always cite their sources"
WEATHER_TEXT = "the weather forecast predicts rain tomorrow"


def _child(
    document_id: str,
    index: int,
    text: str,
    page_count: int | None = 3,
    filename: str = "report.pdf",
) -> ChildChunk:
    """One child, without indexing it — for testing the metadata flattening alone."""
    parent = ParentChunk(
        id=f"{document_id}:p{index}",
        document_id=document_id,
        chunk_index=index,
        text=text,
        char_count=len(text),
        start_index=index * 100,
        metadata={
            "filename": filename,
            "content_type": PDF_CONTENT_TYPE,
            "page_count": page_count,
        },
    )
    return child_of(parent)


class TestChildMetadata:
    def test_none_values_are_dropped(self) -> None:
        """A DOCX has no page count; Chroma metadata must simply omit the key."""
        metadata = _child_metadata(_child(DOC_ONE, 0, "text", page_count=None))

        assert "page_count" not in metadata
        assert metadata["document_id"] == DOC_ONE
        assert metadata["start_index"] == 0

    def test_present_values_are_kept_with_their_types(self) -> None:
        metadata = _child_metadata(_child(DOC_ONE, 2, "text", page_count=7))

        assert metadata["page_count"] == 7
        assert metadata["chunk_index"] == 0
        assert metadata["filename"] == "report.pdf"

    def test_the_parent_id_is_carried(self) -> None:
        """It is the only route back from a matched vector to answerable text."""
        assert _child_metadata(_child(DOC_ONE, 3, "text"))["parent_id"] == f"{DOC_ONE}:p3"

    def test_document_metadata_cannot_shadow_the_parent_id(self) -> None:
        """A document carrying its own `parent_id` must not redirect retrieval."""
        child = _child(DOC_ONE, 1, "text")
        child.metadata["parent_id"] = f"{DOC_TWO}:p9"

        assert _child_metadata(child)["parent_id"] == f"{DOC_ONE}:p1"


class TestAddChildren:
    def test_count_increases_by_the_number_added(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        assert vector_store.count() == 0

        for i in range(3):
            index_parent(vector_store, DOC_ONE, i, f"chunk {i}")

        assert vector_store.count() == 3

    def test_a_parent_with_several_children_stores_one_vector_each(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """The point of the split: many small vectors standing in for one passage."""
        index_parent(
            vector_store, DOC_ONE, 0, "alpha beta gamma", child_texts=["alpha", "beta", "gamma"]
        )

        assert vector_store.count() == 3

    def test_re_adding_the_same_ids_overwrites_rather_than_duplicates(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """Deterministic child ids are what make re-ingesting idempotent."""
        for _ in range(2):
            for i in range(3):
                index_parent(vector_store, DOC_ONE, i, f"chunk {i}")

        assert vector_store.count() == 3

    def test_adding_nothing_is_a_no_op(self, vector_store: VectorStoreService) -> None:
        assert vector_store.add_children([]) == 0
        assert vector_store.count() == 0

    def test_children_without_a_page_count_are_storable(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """Regression: a None metadata value must not break DOCX indexing."""
        index_parent(vector_store, DOC_TWO, 0, "text", page_count=None)

        assert vector_store.count() == 1


class TestMetadataRoundTrip:
    """Metadata is written into Chroma and read back out; the trip must preserve types.

    Citations use `filename` and `page`, and `chunk_index` orders passages, so a silently
    stringified int would only surface as a bug much later. Results are built from the
    parent, so this covers the parent store's round trip as well as Chroma's.
    """

    def test_every_field_survives_with_its_type(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        index_parent(vector_store, DOC_ONE, 2, CITATIONS_TEXT, page_count=7)

        metadata = vector_store.search(CITATIONS_TEXT, top_k_parents=1)[0].metadata

        assert metadata == {
            "start_index": 200,
            "filename": "report.pdf",
            "content_type": PDF_CONTENT_TYPE,
            "page_count": 7,
        }
        for field in ("start_index", "page_count"):
            assert isinstance(metadata[field], int), f"{field} came back as {type(metadata[field])}"
        for field in ("filename", "content_type"):
            assert isinstance(metadata[field], str)

    def test_zero_valued_ints_are_not_dropped_as_falsy(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """`_child_metadata` filters on `is not None`; 0 must not be swept up with it."""
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT, page_count=0)

        result = vector_store.search(CITATIONS_TEXT, top_k_parents=1)[0]

        assert result.chunk_index == 0
        assert result.metadata["start_index"] == 0
        assert result.metadata["page_count"] == 0


class _AxisEmbeddings(Embeddings):
    """Maps a few marker texts onto fixed unit vectors, so cosine scores are exact.

    `FakeEmbeddings` is deliberately hash-derived and only guarantees ordering; this one
    pins the actual arithmetic, which is what the `1 - distance` conversion rests on.
    """

    _TABLE = {"same": [1.0, 0.0], "orthogonal": [0.0, 1.0], "opposite": [-1.0, 0.0]}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._TABLE[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._TABLE[text]


@pytest.fixture
def axis_store(test_settings: Settings) -> VectorStoreService:
    """A real Chroma collection over vectors whose cosine similarities are known exactly."""
    return VectorStoreService(test_settings, _AxisEmbeddings())


class TestScoreRange:
    """`score = 1 - distance` is only meaningful across the full cosine range."""

    def test_the_three_reference_angles_score_exactly(
        self, axis_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        index_parent(axis_store, DOC_ONE, 0, "same")
        index_parent(axis_store, DOC_ONE, 1, "orthogonal")
        index_parent(axis_store, DOC_ONE, 2, "opposite")

        scores = {
            result.text: result.score for result in axis_store.search("same", top_k_parents=3)
        }

        assert scores["same"] == pytest.approx(1.0, abs=1e-6)
        assert scores["orthogonal"] == pytest.approx(0.0, abs=1e-6)
        # Negative similarity is real and must survive: SearchResult documents [-1, 1],
        # so clamping or a ge=0 bound here would silently distort ranking.
        assert scores["opposite"] == pytest.approx(-1.0, abs=1e-6)

    def test_a_negative_score_still_sorts_last(
        self, axis_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        index_parent(axis_store, DOC_ONE, 0, "opposite")
        index_parent(axis_store, DOC_ONE, 1, "orthogonal")

        results = axis_store.search("same", top_k_parents=2)

        assert [result.text for result in results] == ["orthogonal", "opposite"]

    def test_a_negative_score_serializes_through_the_endpoint(
        self,
        client: TestClient,
        axis_store: VectorStoreService,
        monkeypatch: pytest.MonkeyPatch,
        index_parent: Callable[..., ParentChunk],
    ) -> None:
        """A `ge=0` bound on the response model would 500 here instead of returning."""
        index_parent(axis_store, DOC_ONE, 0, "opposite")
        monkeypatch.setattr(client.app.state, "vector_store", axis_store)

        body = client.post("/search", json={"query": "same"}).json()

        assert body["results"][0]["score"] == pytest.approx(-1.0, abs=1e-6)


class TestSearch:
    def test_matching_chunk_ranks_first(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)
        index_parent(vector_store, DOC_ONE, 1, WEATHER_TEXT)

        results = vector_store.search(CITATIONS_TEXT, top_k_parents=2)

        assert results[0].text == CITATIONS_TEXT
        assert results[0].score > results[1].score

    def test_results_are_sorted_by_score_descending(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        for i in range(5):
            index_parent(vector_store, DOC_ONE, i, f"chunk number {i}")

        results = vector_store.search("chunk number 2", top_k_parents=5)

        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_caps_the_result_count(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        for i in range(6):
            index_parent(vector_store, DOC_ONE, i, f"chunk {i}")

        assert len(vector_store.search("chunk", top_k_parents=2)) == 2

    def test_identical_text_scores_near_one(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """Normalized vectors plus cosine distance means an exact match is 1.0."""
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)

        assert vector_store.search(CITATIONS_TEXT, top_k_parents=1)[0].score == pytest.approx(
            1.0, abs=1e-4
        )

    def test_document_id_filter_restricts_results(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)
        index_parent(vector_store, DOC_TWO, 0, CITATIONS_TEXT)

        results = vector_store.search(CITATIONS_TEXT, top_k_parents=5, document_id=DOC_TWO)

        assert [result.document_id for result in results] == [DOC_TWO]

    def test_an_empty_document_id_does_not_silently_widen_the_search(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """Scoping to a document that cannot exist must return nothing, not everything.

        Phase 4 answers from whatever `/search` returns, so a filter that quietly stops
        applying would ground an answer in documents the caller excluded.
        """
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)
        index_parent(vector_store, DOC_TWO, 0, WEATHER_TEXT)

        assert vector_store.search(CITATIONS_TEXT, top_k_parents=5, document_id="") == []

    def test_filter_for_an_unknown_document_returns_nothing(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)

        assert vector_store.search(CITATIONS_TEXT, top_k_parents=5, document_id="missing") == []

    def test_results_carry_chunk_identity_and_metadata(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        index_parent(vector_store, DOC_ONE, 4, CITATIONS_TEXT)

        result = vector_store.search(CITATIONS_TEXT, top_k_parents=1)[0]

        assert result.chunk_id == f"{DOC_ONE}:p4"
        assert result.document_id == DOC_ONE
        assert result.chunk_index == 4
        assert result.metadata["filename"] == "report.pdf"

    def test_search_on_an_empty_collection_returns_nothing(
        self, vector_store: VectorStoreService
    ) -> None:
        assert vector_store.search("anything", top_k_parents=5) == []

    def test_top_k_beyond_the_collection_size_returns_everything(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """Asking for more than exists must return what there is, not error or pad."""
        for i in range(3):
            index_parent(vector_store, DOC_ONE, i, f"chunk {i}")

        results = vector_store.search("chunk", top_k_parents=50)

        assert len(results) == 3
        assert len({result.chunk_id for result in results}) == 3

    def test_top_k_beyond_the_filtered_subset_returns_that_subset(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """The filter shrinks the candidate pool below k; Chroma must not backfill."""
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)
        for i in range(9):
            index_parent(vector_store, DOC_TWO, i, f"chunk {i}")

        results = vector_store.search("chunk", top_k_parents=10, document_id=DOC_ONE)

        assert [result.chunk_id for result in results] == [f"{DOC_ONE}:p0"]


class TestPersistence:
    def test_a_new_service_over_the_same_directory_sees_existing_vectors(
        self,
        test_settings: Settings,
        embedding_service: EmbeddingService,
        index_parent: Callable[..., ParentChunk],
    ) -> None:
        """The collection is persistent, so a restart must not lose the index."""
        first = VectorStoreService(test_settings, embedding_service.as_langchain())
        index_parent(first, DOC_ONE, 0, CITATIONS_TEXT)

        second = VectorStoreService(test_settings, embedding_service.as_langchain())

        assert second.count() == 1
        assert second.search(CITATIONS_TEXT, top_k_parents=1)[0].chunk_id == f"{DOC_ONE}:p0"


class TestDeleteDocument:
    def test_removes_only_that_document(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        index_parent(vector_store, DOC_ONE, 0, "a")
        index_parent(vector_store, DOC_ONE, 1, "b")
        index_parent(vector_store, DOC_TWO, 0, "c")

        removed = vector_store.delete_document(DOC_ONE)

        assert removed == 2
        assert vector_store.count() == 1
        assert vector_store.search("c", top_k_parents=5)[0].document_id == DOC_TWO

    def test_deleting_an_unknown_document_is_a_no_op(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        index_parent(vector_store, DOC_ONE, 0, "a")

        assert vector_store.delete_document("missing") == 0
        assert vector_store.count() == 1

    def test_removes_every_chunk_of_a_many_chunk_document(
        self, vector_store: VectorStoreService, index_parent: Callable[..., ParentChunk]
    ) -> None:
        """A real document is hundreds of chunks; `get` then `delete` must not page out.

        `delete_document` reads ids with an unlimited `get` and deletes them in one call,
        so this guards against a default result limit silently orphaning the tail.
        """
        for i in range(250):
            index_parent(vector_store, DOC_ONE, i, f"chunk number {i}")
        index_parent(vector_store, DOC_TWO, 0, CITATIONS_TEXT)

        removed = vector_store.delete_document(DOC_ONE)

        assert removed == 250
        assert vector_store.count() == 1
        assert vector_store.search("chunk number 7", top_k_parents=5, document_id=DOC_ONE) == []


class TestEmbeddingService:
    def test_count_tokens_grows_with_text_length(self, embedding_service: EmbeddingService) -> None:
        short = embedding_service.count_tokens("one two three")
        long = embedding_service.count_tokens("one two three " * 50)

        assert long > short

    def test_within_limit_accepts_a_normal_chunk(self, embedding_service: EmbeddingService) -> None:
        assert embedding_service.assert_within_limit("a modest chunk of text") is True

    def test_over_limit_is_flagged(self, test_settings: Settings) -> None:
        """A chunk beyond the model's window is silently truncated, so it must warn."""
        service = EmbeddingService(
            test_settings.model_copy(update={"embedding_max_tokens": 5}),
            embeddings=_FakeFromFixture(),
        )

        assert service.assert_within_limit("word " * 200) is False

    def test_embeds_queries_and_documents(self, embedding_service: EmbeddingService) -> None:
        query = embedding_service.embed_query("hello")
        documents = embedding_service.embed_texts(["hello", "world"])

        assert len(documents) == 2
        assert query == pytest.approx(documents[0])

    def test_count_tokens_falls_back_to_a_character_estimate(
        self, embedding_service: EmbeddingService
    ) -> None:
        """The fake exposes no tokenizer, so the documented chars-per-token rule applies."""
        text = "a" * 40

        assert embedding_service.count_tokens(text) == 40 // CHARS_PER_TOKEN_FALLBACK

    def test_count_tokens_uses_the_real_tokenizer_when_one_is_exposed(
        self, test_settings: Settings
    ) -> None:
        """The production path reads `embeddings.client.tokenizer`, not the estimate.

        The stub returns a count the fallback could not produce, so this fails if the
        attribute lookup ever stops finding the tokenizer and silently degrades.
        """
        service = EmbeddingService(test_settings, embeddings=_TokenizerEmbeddings())

        # "one two three" is 13 chars: the fallback would say 3, the tokenizer says 5.
        assert service.count_tokens("one two three") == 5

    def test_the_limit_check_honours_the_real_tokenizer(self, test_settings: Settings) -> None:
        """Whether a chunk is over the window must be decided by tokens, not characters."""
        service = EmbeddingService(
            test_settings.model_copy(update={"embedding_max_tokens": 4}),
            embeddings=_TokenizerEmbeddings(),
        )

        # 5 tokenizer tokens > 4; the character fallback would have said 3 and passed.
        assert service.assert_within_limit("one two three") is False

    def test_exactly_at_the_limit_is_accepted(self, test_settings: Settings) -> None:
        """The check is `>`, so a chunk landing on the boundary must not warn."""
        service = EmbeddingService(
            test_settings.model_copy(update={"embedding_max_tokens": 5}),
            embeddings=_TokenizerEmbeddings(),
        )

        assert service.assert_within_limit("one two three") is True

    def test_as_langchain_returns_the_injected_object(
        self, test_settings: Settings, fake_embeddings: Embeddings
    ) -> None:
        """The store must share the one loaded model rather than build its own."""
        service = EmbeddingService(test_settings, embeddings=fake_embeddings)

        assert service.as_langchain() is fake_embeddings


class _StubTokenizer:
    """Stands in for the sentence-transformers tokenizer: one token per word, plus CLS/SEP."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = list(range(len(text.split())))
        return [-1, *tokens, -2] if add_special_tokens else tokens


class _TokenizerEmbeddings:
    """Shaped like `HuggingFaceEmbeddings`: a `.client` carrying a `.tokenizer`."""

    def __init__(self) -> None:
        self.client = SimpleNamespace(tokenizer=_StubTokenizer())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _FakeFromFixture:
    """Minimal embeddings stand-in for constructing a service without the fixture."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class TestSearchEndpoint:
    def test_returns_a_valid_search_response(
        self,
        client: TestClient,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
    ) -> None:
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)
        index_parent(vector_store, DOC_ONE, 1, WEATHER_TEXT)

        response = client.post("/search", json={"query": CITATIONS_TEXT})

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["query"] == CITATIONS_TEXT
        assert body["count"] == len(body["results"])
        assert body["results"][0]["text"] == CITATIONS_TEXT
        assert body["results"][0]["chunk_id"] == f"{DOC_ONE}:p0"

    def test_top_k_is_respected(
        self,
        client: TestClient,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
    ) -> None:
        for i in range(6):
            index_parent(vector_store, DOC_ONE, i, f"chunk {i}")

        body = client.post("/search", json={"query": "chunk", "top_k": 2}).json()

        assert body["count"] == 2

    def test_default_top_k_comes_from_settings(
        self,
        client: TestClient,
        vector_store: VectorStoreService,
        test_settings: Settings,
        index_parent: Callable[..., ParentChunk],
    ) -> None:
        for i in range(20):
            index_parent(vector_store, DOC_ONE, i, f"chunk {i}")

        body = client.post("/search", json={"query": "chunk"}).json()

        assert body["count"] == test_settings.search_top_k

    def test_document_id_filter_is_applied(
        self,
        client: TestClient,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
    ) -> None:
        index_parent(vector_store, DOC_ONE, 0, CITATIONS_TEXT)
        index_parent(vector_store, DOC_TWO, 0, CITATIONS_TEXT)

        body = client.post("/search", json={"query": CITATIONS_TEXT, "document_id": DOC_TWO}).json()

        assert {result["document_id"] for result in body["results"]} == {DOC_TWO}

    def test_a_non_uuid_document_id_is_rejected(self, client: TestClient) -> None:
        """The id scheme is uuid everywhere else; /search must not be the soft spot."""
        response = client.post("/search", json={"query": "x", "document_id": "doc2"})

        assert response.status_code == 422

    def test_a_dict_document_id_cannot_smuggle_a_chroma_operator(self, client: TestClient) -> None:
        """A dict would reach Chroma's filter language as an operator like $ne."""
        response = client.post("/search", json={"query": "x", "document_id": {"$ne": "doc1"}})

        assert response.status_code == 422

    def test_unknown_fields_are_rejected(self, client: TestClient) -> None:
        """A misspelled `topk` silently using the default is worse than a 422."""
        response = client.post("/search", json={"query": "x", "topk": 3})

        assert response.status_code == 422

    def test_whitespace_only_query_is_rejected(self, client: TestClient) -> None:
        assert client.post("/search", json={"query": "   "}).status_code == 422

    def test_empty_query_is_rejected(self, client: TestClient) -> None:
        assert client.post("/search", json={"query": ""}).status_code == 422

    def test_overlong_query_is_rejected(self, client: TestClient) -> None:
        """The query is embedded, so its length is unbounded model work."""
        assert client.post("/search", json={"query": "x" * 5000}).status_code == 422

    @pytest.mark.parametrize("top_k", [0, -1, 51])
    def test_out_of_range_top_k_is_rejected(self, client: TestClient, top_k: int) -> None:
        assert client.post("/search", json={"query": "x", "top_k": top_k}).status_code == 422

    def test_missing_query_is_rejected(self, client: TestClient) -> None:
        assert client.post("/search", json={}).status_code == 422


class TestServiceWiring:
    """The spec's "loaded once at startup, never per request" acceptance criterion."""

    def test_the_model_is_built_once_and_shared_by_every_request(
        self,
        test_settings: Settings,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A per-request build would re-load ~90 MB of weights on every call."""
        builds: list[Settings] = []

        def counting_build(settings: Settings) -> EmbeddingService:
            builds.append(settings)
            return embedding_service

        monkeypatch.setattr(main, "get_settings", lambda: test_settings)
        monkeypatch.setattr(main, "build_embedding_service", counting_build)
        monkeypatch.setattr(main, "build_vector_store", lambda _settings, _embeddings: vector_store)
        app.dependency_overrides[get_settings] = lambda: test_settings
        try:
            with TestClient(app) as test_client:
                for _ in range(3):
                    assert test_client.post("/search", json={"query": "x"}).status_code == 200
                    assert test_client.get("/stats").status_code == 200
                assert app.state.embedding_service is embedding_service
                assert app.state.vector_store is vector_store
        finally:
            app.dependency_overrides.clear()

        assert len(builds) == 1, f"embedding model built {len(builds)} times, expected once"


class TestStatsEndpoint:
    def test_reports_zero_for_an_empty_collection(self, client: TestClient) -> None:
        body = client.get("/stats").json()

        assert body["total_vectors"] == 0
        assert body["collection"] == "documents"

    def test_reports_the_vector_count(
        self,
        client: TestClient,
        vector_store: VectorStoreService,
        index_parent: Callable[..., ParentChunk],
    ) -> None:
        for i in range(4):
            index_parent(vector_store, DOC_ONE, i, f"chunk {i}")

        assert client.get("/stats").json()["total_vectors"] == 4


class TestIngestionIndexesChunks:
    def test_ingesting_a_pdf_stores_vectors(
        self, client: TestClient, sample_pdf_bytes: bytes
    ) -> None:
        response = client.post(
            "/ingest", files={"file": ("sample.pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)}
        )

        assert response.status_code == 200, response.text
        chunk_count = response.json()["child_count"]
        assert client.get("/stats").json()["total_vectors"] == chunk_count

    def test_ingesting_a_docx_stores_vectors_despite_no_page_count(
        self, client: TestClient, sample_docx_bytes: bytes
    ) -> None:
        """The end-to-end version of the None-metadata regression."""
        response = client.post(
            "/ingest", files={"file": ("sample.docx", sample_docx_bytes, DOCX_CONTENT_TYPE)}
        )

        assert response.status_code == 200, response.text
        assert client.get("/stats").json()["total_vectors"] == response.json()["child_count"]

    def test_ingested_content_is_searchable(
        self, client: TestClient, sample_docx_bytes: bytes
    ) -> None:
        client.post(
            "/ingest", files={"file": ("sample.docx", sample_docx_bytes, DOCX_CONTENT_TYPE)}
        )

        body = client.post("/search", json={"query": "Grounded answers require citations."}).json()

        assert body["count"] > 0
        assert "citations" in body["results"][0]["text"].lower()

    def test_failed_ingest_leaves_no_vectors(
        self, client: TestClient, corrupt_pdf_bytes: bytes, upload_dir: Path
    ) -> None:
        response = client.post(
            "/ingest", files={"file": ("bad.pdf", corrupt_pdf_bytes, PDF_CONTENT_TYPE)}
        )

        assert response.status_code == 422
        assert client.get("/stats").json()["total_vectors"] == 0
        assert list(upload_dir.iterdir()) == []

    def test_re_indexing_the_same_chunks_does_not_duplicate(
        self,
        client: TestClient,
        sample_pdf_bytes: bytes,
        vector_store: VectorStoreService,
        test_settings: Settings,
    ) -> None:
        """Re-ingesting a document must overwrite its vectors, not stack new ones.

        The children are rebuilt from the stored text rather than reloaded: they live only
        in the collection, so splitting again is what a genuine re-ingest would do, and it
        is exactly what must produce the same deterministic ids.
        """
        response = client.post(
            "/ingest", files={"file": ("sample.pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)}
        )
        document_id = response.json()["document"]["id"]
        before = client.get("/stats").json()["total_vectors"]

        text = (test_settings.upload_dir / f"{document_id}.txt").read_text(encoding="utf-8")
        _parents, children = chunk_document(text, document_id, {}, test_settings)
        vector_store.add_children(children)

        assert client.get("/stats").json()["total_vectors"] == before

    def test_a_partially_successful_vector_add_is_rolled_back(
        self,
        client: TestClient,
        long_docx_bytes: bytes,
        vector_store: VectorStoreService,
        upload_dir: Path,
        test_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The rollback must undo writes that already landed, not just the failed call.

        Chroma writes in batches, so a mid-add failure can leave some vectors committed.
        Deleting only on a clean failure would leave `/stats` and `/search` reporting
        fragments of a document the client was told did not ingest.
        """
        real_add_texts = vector_store._store.add_texts
        committed: list[str] = []

        def half_then_fail(
            texts: list[str],
            metadatas: list[dict[str, Any]] | None = None,
            ids: list[str] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            assert ids is not None and metadatas is not None
            half = max(1, len(texts) // 2)
            assert len(texts) > 1, "fixture must chunk into more than one piece to be partial"
            committed.extend(ids[:half])
            real_add_texts(texts=texts[:half], metadatas=metadatas[:half], ids=ids[:half])
            raise RuntimeError("chroma write interrupted mid-batch")

        monkeypatch.setattr(vector_store._store, "add_texts", half_then_fail)

        # A store failure is now a clean 503 rather than an unhandled exception.
        response = client.post(
            "/ingest", files={"file": ("long.docx", long_docx_bytes, DOCX_CONTENT_TYPE)}
        )

        assert response.status_code == 503
        assert response.json()["error"] == "VectorStoreError"

        assert committed, "the partial write did not happen; the test proves nothing"
        assert client.get("/stats").json()["total_vectors"] == 0
        assert list(upload_dir.iterdir()) == []
        assert list(test_settings.upload_dir.glob("*.chunks.json")) == []

    def test_concurrent_ingests_all_land_in_the_one_collection(
        self, client: TestClient, sample_pdf_bytes: bytes
    ) -> None:
        """One process-wide store serves every request, so parallel writes must not race.

        Uvicorn runs handlers concurrently and the embed/store step is dispatched to a
        thread pool, so several documents really can be written at once.
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

        document_ids = {body["document"]["id"] for body in bodies}
        assert len(document_ids) == ingests, "uploads collided on a document id"
        assert client.get("/stats").json()["total_vectors"] == sum(
            body["child_count"] for body in bodies
        )
        for document_id in document_ids:
            body = client.post(
                "/search", json={"query": "Retrieval augmented", "document_id": document_id}
            ).json()
            assert body["count"] > 0, f"{document_id} is missing from the index"


@pytest.mark.integration
def test_real_model_ranks_a_semantic_match_first(
    tmp_path: Path, index_parent: Callable[..., ParentChunk]
) -> None:
    """Sanity-check the real embedding model: paraphrases should beat unrelated text.

    Deselected by default (see pyproject addopts). Run with: pytest -m integration
    """
    settings = Settings(chroma_persist_dir=tmp_path / "chroma")
    service = EmbeddingService(settings)
    store = VectorStoreService(settings, service.as_langchain())
    index_parent(
        store, DOC_ONE, 0, "Every generated answer must cite the source document it came from."
    )
    index_parent(store, DOC_ONE, 1, "Preheat the oven to 200 degrees and butter a cake tin.")

    results = store.search("How are sources attributed in the responses?", top_k_parents=2)

    assert "cite the source" in results[0].text
    assert results[0].score > results[1].score
