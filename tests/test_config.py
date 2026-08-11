"""Tests for settings loading and the cached `get_settings` dependency."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import (
    CONTENT_TYPE_EXTENSIONS,
    DOCX_CONTENT_TYPE,
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

        assert settings.chunk_size == 800
        assert settings.chunk_overlap == 120
        assert settings.chunk_separators == ["\n\n", "\n", ". ", " ", ""]

    def test_overlap_may_not_equal_or_exceed_chunk_size(self) -> None:
        """A window with no forward progress would loop or lose text."""
        with pytest.raises(ValidationError, match="chunk_overlap"):
            Settings(chunk_size=200, chunk_overlap=200)

        with pytest.raises(ValidationError, match="chunk_overlap"):
            Settings(chunk_size=200, chunk_overlap=500)

    def test_overlap_up_to_half_the_chunk_size_is_accepted(self) -> None:
        assert Settings(chunk_size=200, chunk_overlap=100).chunk_overlap == 100

    def test_overlap_above_half_is_refused(self) -> None:
        """Beyond half, the splitter cannot advance a full split and drops text."""
        with pytest.raises(ValidationError, match="chunk_overlap"):
            Settings(chunk_size=200, chunk_overlap=101)

    def test_non_positive_chunk_size_is_refused(self) -> None:
        """Reaches the splitter as a bare ValueError and 500s the request otherwise."""
        for size in (0, -100):
            with pytest.raises(ValidationError):
                Settings(chunk_size=size)

    def test_chunk_settings_are_environment_overridable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHUNK_SIZE", "1500")
        # The value is parsed as JSON, so the newline separator must be escaped as \\n —
        # a raw newline here is an invalid control character inside a JSON string.
        monkeypatch.setenv("CHUNK_SEPARATORS", '["\\n", " "]')

        settings = Settings()

        assert settings.chunk_size == 1500
        assert settings.chunk_separators == ["\n", " "]
