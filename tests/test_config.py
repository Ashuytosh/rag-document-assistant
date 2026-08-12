"""Tests for settings loading and the cached `get_settings` dependency."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import (
    CONTENT_TYPE_EXTENSIONS,
    DOCX_CONTENT_TYPE,
    MAX_TOP_K,
    PDF_CONTENT_TYPE,
    Settings,
    get_settings,
)


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """Keep the process-wide cache out of other tests' way."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_match_the_spec() -> None:
    settings = Settings()

    assert settings.app_name
    assert settings.app_version
    assert settings.upload_dir == Path("data/uploads")
    assert settings.max_upload_bytes == 20 * 1024 * 1024
    assert settings.allowed_content_types == {PDF_CONTENT_TYPE, DOCX_CONTENT_TYPE}


def test_every_allowed_content_type_has_a_storage_extension() -> None:
    """Otherwise a route-accepted upload would be rejected later by storage."""
    assert set(Settings().allowed_content_types) <= set(CONTENT_TYPE_EXTENSIONS)


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "4096")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "custom"))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = get_settings()

    assert settings.max_upload_bytes == 4096
    assert settings.upload_dir == tmp_path / "custom"
    assert settings.log_level == "DEBUG"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_cache_clear_rebuilds_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache is what makes the dependency cheap; clearing it must pick up new values."""
    first = get_settings()
    monkeypatch.setenv("APP_NAME", "Renamed Assistant")

    assert get_settings() is first, "cached value should survive an env change"

    get_settings.cache_clear()
    assert get_settings().app_name == "Renamed Assistant"


def test_unknown_environment_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_UNRELATED_SETTING", "value")

    assert get_settings().app_name


class TestChunkSettings:
    def test_defaults_match_the_spec(self) -> None:
        settings = Settings()

        assert settings.parent_chunk_size == 2000
        assert settings.parent_chunk_overlap == 200
        assert settings.child_chunk_size == 400
        assert settings.child_chunk_overlap == 80
        assert settings.chunk_separators == ["\n\n", "\n", ". ", " ", ""]

    @pytest.mark.parametrize("level", ["parent", "child"])
    def test_overlap_may_not_equal_or_exceed_chunk_size(self, level: str) -> None:
        """A window with no forward progress would loop or lose text."""
        for overlap in (200, 500):
            with pytest.raises(ValidationError, match=f"{level}_chunk_overlap"):
                Settings(**{f"{level}_chunk_size": 200, f"{level}_chunk_overlap": overlap})

    def test_overlap_up_to_half_the_chunk_size_is_accepted(self) -> None:
        settings = Settings(
            parent_chunk_size=200,
            parent_chunk_overlap=100,
            child_chunk_size=50,
            child_chunk_overlap=25,
        )

        assert settings.parent_chunk_overlap == 100
        assert settings.child_chunk_overlap == 25

    @pytest.mark.parametrize("level", ["parent", "child"])
    def test_overlap_above_half_is_refused(self, level: str) -> None:
        """Beyond half, the splitter cannot advance a full split and drops text."""
        with pytest.raises(ValidationError, match=f"{level}_chunk_overlap"):
            Settings(**{f"{level}_chunk_size": 200, f"{level}_chunk_overlap": 101})

    @pytest.mark.parametrize("level", ["parent", "child"])
    def test_non_positive_chunk_size_is_refused(self, level: str) -> None:
        """Reaches the splitter as a bare ValueError and 500s the request otherwise."""
        for size in (0, -100):
            with pytest.raises(ValidationError):
                Settings(**{f"{level}_chunk_size": size})

    def test_a_child_at_least_as_large_as_its_parent_is_refused(self) -> None:
        """Equal sizes collapse the two levels back into Phase 2's flat chunking."""
        for child in (2000, 3000):
            with pytest.raises(ValidationError, match="child_chunk_size"):
                Settings(child_chunk_size=child)

    def test_chunk_settings_are_environment_overridable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PARENT_CHUNK_SIZE", "1500")
        monkeypatch.setenv("CHILD_CHUNK_SIZE", "300")
        # The value is parsed as JSON, so the newline separator must be escaped as \\n —
        # a raw newline here is an invalid control character inside a JSON string.
        monkeypatch.setenv("CHUNK_SEPARATORS", '["\\n", " "]')

        settings = Settings()

        assert settings.parent_chunk_size == 1500
        assert settings.child_chunk_size == 300
        assert settings.chunk_separators == ["\n", " "]


class TestRetrievalSettings:
    """The embedding and vector-store knobs, which the scoring maths depends on."""

    def test_defaults_match_the_spec(self) -> None:
        settings = Settings()

        assert settings.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert settings.embedding_device == "cpu"
        assert settings.embedding_normalize is True
        assert settings.embedding_max_tokens == 256
        assert settings.chroma_persist_dir == Path("chroma_db")
        assert settings.chroma_collection == "documents"
        assert settings.chroma_distance == "cosine"
        assert settings.search_top_k == 5

    def test_a_non_cosine_distance_is_refused(self) -> None:
        """`score = 1 - distance` is only a similarity under cosine; l2 would be nonsense."""
        for space in ("l2", "ip", "COSINE"):
            with pytest.raises(ValidationError, match="chroma_distance"):
                Settings(chroma_distance=space)

    @pytest.mark.parametrize("name", ["ab", "-documents", "documents-", "my documents", "a" * 513])
    def test_a_collection_name_chroma_would_reject_fails_at_startup(self, name: str) -> None:
        """Chroma raises on a bad name at first use; catching it in config fails faster."""
        with pytest.raises(ValidationError, match="chroma_collection"):
            Settings(chroma_collection=name)

    @pytest.mark.parametrize("name", ["documents", "doc.v2", "doc_v2", "doc-v2", "a1b"])
    def test_a_valid_collection_name_is_accepted(self, name: str) -> None:
        assert Settings(chroma_collection=name).chroma_collection == name

    @pytest.mark.parametrize("top_k", [0, -1, MAX_TOP_K + 1])
    def test_out_of_range_default_top_k_is_refused(self, top_k: int) -> None:
        """The default feeds straight into the store when a request omits `top_k`."""
        with pytest.raises(ValidationError, match="search_top_k"):
            Settings(search_top_k=top_k)

    def test_the_default_top_k_may_sit_at_the_ceiling(self) -> None:
        assert Settings(search_top_k=MAX_TOP_K).search_top_k == MAX_TOP_K

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_non_positive_token_limit_is_refused(self, limit: int) -> None:
        """Zero would flag every chunk as oversized and drown the logs in warnings."""
        with pytest.raises(ValidationError, match="embedding_max_tokens"):
            Settings(embedding_max_tokens=limit)

    def test_retrieval_settings_are_environment_overridable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("SEARCH_TOP_K", "9")
        monkeypatch.setenv("CHROMA_COLLECTION", "phase3")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "vectors"))
        monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")

        settings = get_settings()

        assert settings.search_top_k == 9
        assert settings.chroma_collection == "phase3"
        assert settings.chroma_persist_dir == tmp_path / "vectors"
        assert settings.embedding_device == "cuda"
