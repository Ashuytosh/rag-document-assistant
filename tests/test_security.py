"""Tests for the hardening applied after the Phase 1 security review.

Each test here corresponds to a specific reported finding, so a regression names the
vulnerability it reintroduces.
"""

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import DOCX_CONTENT_TYPE, PDF_CONTENT_TYPE, Settings
from app.core.exceptions import ExtractionError
from app.routers.ingestion import _safe_filename
from app.services.extraction import extract_text


def _zip_bomb(uncompressed_bytes: int) -> bytes:
    """A structurally valid DOCX whose members expand enormously.

    python-docx materializes each member in memory, so without a guard this is an
    out-of-memory kill triggered by a small upload.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "A" * uncompressed_bytes)
    return buffer.getvalue()


class TestDecompressionBomb:
    def test_highly_compressible_docx_is_rejected(
        self, test_settings: Settings, upload_dir: Path
    ) -> None:
        path = upload_dir / "bomb.docx"
        path.write_bytes(_zip_bomb(50 * 1024 * 1024))

        with pytest.raises(ExtractionError, match="structural validation"):
            extract_text(path, DOCX_CONTENT_TYPE, test_settings)

    def test_bomb_is_rejected_over_http_without_exhausting_memory(
        self, client: TestClient, upload_dir: Path
    ) -> None:
        response = client.post(
            "/ingest",
            files={"file": ("bomb.docx", _zip_bomb(50 * 1024 * 1024), DOCX_CONTENT_TYPE)},
        )

        assert response.status_code == 422
        assert response.json()["error"] == "ExtractionError"
        assert sorted(p.name for p in upload_dir.iterdir()) == []

    def test_ordinary_docx_still_passes_the_guard(
        self, client: TestClient, sample_docx_bytes: bytes
    ) -> None:
        """The guard must not reject legitimately compressible real documents."""
        response = client.post(
            "/ingest",
            files={"file": ("sample.docx", sample_docx_bytes, DOCX_CONTENT_TYPE)},
        )

        assert response.status_code == 200, response.text


class TestErrorsDoNotLeakPaths:
    def test_corrupt_docx_response_hides_the_server_path(
        self, client: TestClient, upload_dir: Path
    ) -> None:
        """python-docx embeds the absolute file path in its exception message."""
        response = client.post(
            "/ingest",
            files={"file": ("bad.docx", b"definitely not a zip", DOCX_CONTENT_TYPE)},
        )

        assert response.status_code == 422
        body = response.text
        assert str(upload_dir) not in body
        assert "Package not found" not in body

    def test_corrupt_pdf_response_hides_the_server_path(
        self, client: TestClient, corrupt_pdf_bytes: bytes, upload_dir: Path
    ) -> None:
        response = client.post(
            "/ingest",
            files={"file": ("bad.pdf", corrupt_pdf_bytes, PDF_CONTENT_TYPE)},
        )

        assert response.status_code == 422
        assert str(upload_dir) not in response.text


class TestFilenameHandling:
    def test_overlong_filename_is_truncated(
        self, client: TestClient, sample_pdf_bytes: bytes
    ) -> None:
        """400 chars gets through the multipart parser; the response must still be capped."""
        response = client.post(
            "/ingest",
            files={"file": ("a" * 400 + ".pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)},
        )

        assert response.status_code == 200, response.text
        assert len(response.json()["document"]["filename"]) == 255

    @pytest.mark.parametrize(
        "supplied",
        ["../../evil.pdf", r"..\..\evil.pdf", "/etc/passwd.pdf", "sub/dir/report.pdf"],
    )
    def test_directory_components_are_stripped_from_the_echoed_name(
        self, client: TestClient, sample_pdf_bytes: bytes, supplied: str
    ) -> None:
        response = client.post(
            "/ingest",
            files={"file": (supplied, sample_pdf_bytes, PDF_CONTENT_TYPE)},
        )

        filename = response.json()["document"]["filename"]
        assert "/" not in filename
        assert "\\" not in filename
        assert ".." not in filename

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("re\x00port\x1f.pdf", "report.pdf"),
            ("../../evil.pdf", "evil.pdf"),
            (r"..\..\evil.pdf", "evil.pdf"),
            ("  spaced.pdf  ", "spaced.pdf"),
            ("a" * 400 + ".pdf", "a" * 255),
            ("", "upload"),
            (None, "upload"),
            ("\x00\x01", "upload"),
        ],
    )
    def test_safe_filename_reduces_to_a_bounded_basename(
        self, raw: str | None, expected: str
    ) -> None:
        """Unit-level: an HTTP client percent-encodes control bytes, so test directly."""
        assert _safe_filename(raw, fallback="upload") == expected
