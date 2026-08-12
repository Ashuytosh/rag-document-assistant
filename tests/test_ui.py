"""Render smoke tests for the chat page.

The streaming, upload, and citation behaviour lives in the browser and is verified by
hand; what is worth asserting here is that the page is served at all, that it is wired to
the running settings, and that its script actually resolves.
"""

import re

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import MAX_GENERATION_TOP_K, Settings


def test_root_serves_the_chat_page(client: TestClient, test_settings: Settings) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert test_settings.app_name in response.text
    assert 'id="chat"' in response.text


def test_the_page_references_the_client_script(client: TestClient) -> None:
    assert "/static/app.js" in client.get("/").text


def test_the_model_dropdown_is_populated_from_settings(
    client: TestClient, test_settings: Settings
) -> None:
    body = client.get("/").text

    for model in test_settings.available_models:
        assert f'<option value="{model}"' in body


def test_the_configured_model_is_preselected(client: TestClient, test_settings: Settings) -> None:
    """Otherwise the dropdown would silently disagree with what the server would use."""
    body = client.get("/").text
    option = re.search(rf'<option value="{re.escape(test_settings.generation_model)}"[^>]*>', body)

    assert option is not None
    assert "selected" in option.group(0)


def test_the_top_k_input_is_bounded_by_the_generation_ceiling(
    client: TestClient, test_settings: Settings
) -> None:
    """A page offering more passages than /query accepts would 422 on submit."""
    body = client.get("/").text
    field = re.search(r'<input\s+id="topk-input".*?/>', body, re.DOTALL)

    assert field is not None
    assert f'max="{MAX_GENERATION_TOP_K}"' in field.group(0)
    assert f'value="{test_settings.generation_top_k}"' in field.group(0)


def test_the_static_mount_serves_the_client_script(client: TestClient) -> None:
    """Proves the mount resolves from the package, not the working directory."""
    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


def test_a_model_outside_the_allowlist_cannot_be_offered() -> None:
    """The dropdown must not be able to offer something /query would reject with a 422."""
    with pytest.raises(ValidationError, match="available_models"):
        Settings(available_models=["qwen2.5:7b", "not-installed:70b"])


def test_the_default_model_must_be_offered_in_the_dropdown() -> None:
    """The page preselects generation_model; omitting it would start on another model."""
    with pytest.raises(ValidationError, match="generation_model"):
        Settings(available_models=["gemma3:4b"], generation_model="qwen2.5:7b")


def test_an_empty_dropdown_is_refused() -> None:
    """With no options the client would send an empty model name and get a 422."""
    with pytest.raises(ValidationError):
        Settings(available_models=[])


def test_a_sub_megabyte_upload_limit_is_not_rendered_as_zero(client: TestClient) -> None:
    """The test settings cap uploads at 128 KB; "up to 0 MB" would be a lie to the user."""
    body = client.get("/").text

    assert "up to 0 MB" not in body
    assert "up to 0.1 MB" in body


def test_the_default_available_models_are_all_allowed() -> None:
    settings = Settings()

    assert set(settings.available_models) <= settings.allowed_generation_models
    assert settings.generation_model in settings.available_models
