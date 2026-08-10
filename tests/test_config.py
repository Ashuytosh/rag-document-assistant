"""Tests for settings loading and the cached `get_settings` dependency."""

from collections.abc import Iterator
from pathlib import Path

import pytest

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
