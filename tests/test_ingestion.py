"""End-to-end tests for /health and /ingest."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import DOCX_CONTENT_TYPE, PDF_CONTENT_TYPE
from app.models.document import PREVIEW_CHARS
from tests.conftest import DOCX_PARAGRAPHS, PDF_SAMPLE_TEXT, TEST_MAX_UPLOAD_BYTES


def _stored_files(upload_dir: Path) -> list[str]:
    return sorted(path.name for path in upload_dir.iterdir())


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"]
    assert body["version"]


def test_ingest_pdf_returns_metadata(
    client: TestClient, sample_pdf_bytes: bytes, upload_dir: Path
) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("sample.pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)},
    )

    assert response.status_code == 200, response.text
    document = response.json()["document"]
    assert document["char_count"] > 0
    assert document["page_count"] >= 1
    assert document["filename"] == "sample.pdf"
    assert document["content_type"] == PDF_CONTENT_TYPE
    assert document["size_bytes"] == len(sample_pdf_bytes)
    assert PDF_SAMPLE_TEXT in document["text_preview"]

    # The upload, its extracted text, and its chunks are persisted under the returned id.
    assert _stored_files(upload_dir) == [
        f"{document['id']}.parents.json",
        f"{document['id']}.pdf",
        f"{document['id']}.txt",
    ]
    saved_text = (upload_dir / f"{document['id']}.txt").read_text(encoding="utf-8")
    assert saved_text.strip() == PDF_SAMPLE_TEXT


def test_ingest_docx_has_no_page_count(
    client: TestClient, sample_docx_bytes: bytes, upload_dir: Path
) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("sample.docx", sample_docx_bytes, DOCX_CONTENT_TYPE)},
    )

    assert response.status_code == 200, response.text
    document = response.json()["document"]
    assert document["char_count"] > 0
    assert document["page_count"] is None
    for paragraph in DOCX_PARAGRAPHS:
        assert paragraph in document["text_preview"]

    assert _stored_files(upload_dir) == [
        f"{document['id']}.docx",
        f"{document['id']}.parents.json",
        f"{document['id']}.txt",
    ]


def test_ingest_response_omits_full_text(client: TestClient, sample_pdf_bytes: bytes) -> None:
    """Only metadata and a preview come back — never the whole extracted document."""
    response = client.post(
        "/ingest",
        files={"file": ("sample.pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)},
    )

    document = response.json()["document"]
    assert "text" not in document
    assert len(document["text_preview"]) <= 500


def test_unsupported_content_type_returns_415(client: TestClient, upload_dir: Path) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("notes.txt", b"just some plain text", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["error"] == "UnsupportedFileTypeError"
    assert response.json()["detail"]
    assert _stored_files(upload_dir) == []  # rejected before anything was written


def test_oversized_upload_returns_413(client: TestClient, upload_dir: Path) -> None:
    oversized = b"%PDF-1.4\n" + b"x" * (TEST_MAX_UPLOAD_BYTES * 2)

    response = client.post(
        "/ingest",
        files={"file": ("big.pdf", oversized, PDF_CONTENT_TYPE)},
    )

    assert response.status_code == 413
    assert response.json()["error"] == "FileTooLargeError"
    assert _stored_files(upload_dir) == []  # the partial write was cleaned up


def test_corrupt_pdf_returns_422(
    client: TestClient, corrupt_pdf_bytes: bytes, upload_dir: Path
) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("corrupt.pdf", corrupt_pdf_bytes, PDF_CONTENT_TYPE)},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "ExtractionError"
    assert _stored_files(upload_dir) == []  # unusable upload was not kept


def test_empty_pdf_returns_422(client: TestClient, empty_pdf_bytes: bytes) -> None:
    """A parseable document with no text is a failed ingest, not an empty success."""
    response = client.post(
        "/ingest",
        files={"file": ("blank.pdf", empty_pdf_bytes, PDF_CONTENT_TYPE)},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "ExtractionError"


def test_blank_docx_returns_422(client: TestClient, empty_docx_bytes: bytes) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("blank.docx", empty_docx_bytes, DOCX_CONTENT_TYPE)},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "ExtractionError"


def test_mislabelled_pdf_as_docx_returns_422(
    client: TestClient, sample_pdf_bytes: bytes, upload_dir: Path
) -> None:
    """The declared content type drives extraction, so a lying client gets a clean 422."""
    response = client.post(
        "/ingest",
        files={"file": ("liar.docx", sample_pdf_bytes, DOCX_CONTENT_TYPE)},
    )

    assert response.status_code == 422
    assert _stored_files(upload_dir) == []


def test_missing_file_field_returns_422(client: TestClient) -> None:
    response = client.post("/ingest", data={"not_a_file": "x"})

    assert response.status_code == 422


def test_preview_is_truncated_but_full_text_is_persisted(
    client: TestClient, long_docx_bytes: bytes, upload_dir: Path
) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("long.docx", long_docx_bytes, DOCX_CONTENT_TYPE)},
    )

    assert response.status_code == 200, response.text
    document = response.json()["document"]
    assert document["char_count"] > PREVIEW_CHARS
    assert len(document["text_preview"]) == PREVIEW_CHARS

    saved_text = (upload_dir / f"{document['id']}.txt").read_text(encoding="utf-8")
    assert len(saved_text) == document["char_count"]
    assert saved_text.startswith(document["text_preview"])


def test_stored_filename_ignores_client_supplied_path(
    client: TestClient, sample_pdf_bytes: bytes, upload_dir: Path
) -> None:
    """A traversal-flavoured filename is echoed back but never used on disk."""
    response = client.post(
        "/ingest",
        files={"file": ("../../evil.pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)},
    )

    assert response.status_code == 200, response.text
    document = response.json()["document"]
    assert _stored_files(upload_dir) == [
        f"{document['id']}.parents.json",
        f"{document['id']}.pdf",
        f"{document['id']}.txt",
    ]


def test_each_ingest_gets_a_distinct_id(client: TestClient, sample_pdf_bytes: bytes) -> None:
    files = {"file": ("sample.pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)}
    first = client.post("/ingest", files=files).json()["document"]["id"]
    second = client.post("/ingest", files=files).json()["document"]["id"]

    assert first != second


def test_error_bodies_carry_no_stack_trace(client: TestClient) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("notes.txt", b"plain", "text/plain")},
    )

    assert set(response.json()) == {"error", "detail"}
    assert "Traceback" not in response.text


def test_ingest_does_not_touch_the_real_upload_dir(
    client: TestClient, sample_pdf_bytes: bytes, upload_dir: Path
) -> None:
    """Guards the fixtures themselves: everything must land in the tmp_path directory."""
    response = client.post(
        "/ingest",
        files={"file": ("sample.pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)},
    )

    document = response.json()["document"]
    assert (upload_dir / f"{document['id']}.pdf").exists()
    assert not (Path("data/uploads") / f"{document['id']}.pdf").exists()
