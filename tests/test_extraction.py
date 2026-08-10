"""Unit tests for the extraction service.

These exercise `extract_text` and its whitespace normalization directly, without going
through the HTTP layer, so failures point at the parser rather than the route.
"""

from pathlib import Path

import pytest

from app.config import DOCX_CONTENT_TYPE, PDF_CONTENT_TYPE
from app.core.exceptions import ExtractionError, UnsupportedFileTypeError
from app.services.extraction import _normalize, extract_text
from tests.conftest import DOCX_PARAGRAPHS, PDF_SAMPLE_TEXT


def _write(directory: Path, name: str, payload: bytes) -> Path:
    path = directory / name
    path.write_bytes(payload)
    return path


class TestNormalize:
    """`_normalize` must tidy whitespace without flattening paragraph structure."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            pytest.param("  hello  ", "hello", id="strips-outer-whitespace"),
            pytest.param("a\nb", "a\nb", id="keeps-single-newline"),
            pytest.param("a\n\nb", "a\n\nb", id="keeps-paragraph-break"),
            pytest.param("a\n\n\nb", "a\n\nb", id="collapses-three-newlines"),
            pytest.param("a\n\n\n\n\n\n\nb", "a\n\nb", id="collapses-long-blank-run"),
            pytest.param("a   \nb\t\t\nc", "a\nb\nc", id="drops-trailing-spaces-per-line"),
            pytest.param("a\r\nb", "a\nb", id="normalizes-crlf"),
            pytest.param("a\rb", "a\nb", id="normalizes-lone-cr"),
            pytest.param("a \r\n\r\n\r\n\r\nb", "a\n\nb", id="crlf-blank-run-collapses"),
            pytest.param("word  spaced", "word  spaced", id="preserves-intra-line-spacing"),
            pytest.param("", "", id="empty-stays-empty"),
            pytest.param("  \n\t\n \n ", "", id="whitespace-only-becomes-empty"),
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert _normalize(raw) == expected

    def test_normalize_is_idempotent(self) -> None:
        raw = "  Heading \r\n\r\n\r\nBody line one\nBody line two\t\n\n\n\n"
        once = _normalize(raw)
        assert _normalize(once) == once

    def test_normalize_preserves_multi_paragraph_document(self) -> None:
        raw = "Para one.\n\n\n\nPara two.\nStill two.\n\n\nPara three.\n\n\n"
        assert _normalize(raw) == "Para one.\n\nPara two.\nStill two.\n\nPara three."


class TestExtractText:
    def test_pdf_returns_text_and_page_count(self, tmp_path: Path, sample_pdf_bytes: bytes) -> None:
        result = extract_text(_write(tmp_path, "s.pdf", sample_pdf_bytes), PDF_CONTENT_TYPE)

        assert PDF_SAMPLE_TEXT in result.text
        assert result.page_count == 1
        assert result.char_count == len(result.text)

    def test_docx_returns_text_without_page_count(
        self, tmp_path: Path, sample_docx_bytes: bytes
    ) -> None:
        result = extract_text(_write(tmp_path, "s.docx", sample_docx_bytes), DOCX_CONTENT_TYPE)

        assert result.page_count is None
        for paragraph in DOCX_PARAGRAPHS:
            assert paragraph in result.text
        assert result.char_count == len(result.text)

    def test_extracted_text_is_normalized(self, tmp_path: Path, sample_pdf_bytes: bytes) -> None:
        """The returned text carries no leading/trailing whitespace or blank-line runs."""
        result = extract_text(_write(tmp_path, "s.pdf", sample_pdf_bytes), PDF_CONTENT_TYPE)

        assert result.text == result.text.strip()
        assert "\n\n\n" not in result.text

    def test_empty_pdf_raises_extraction_error(
        self, tmp_path: Path, empty_pdf_bytes: bytes
    ) -> None:
        """A parseable PDF with no recoverable text is useless downstream."""
        path = _write(tmp_path, "empty.pdf", empty_pdf_bytes)

        with pytest.raises(ExtractionError, match="No text"):
            extract_text(path, PDF_CONTENT_TYPE)

    def test_blank_docx_raises_extraction_error(
        self, tmp_path: Path, empty_docx_bytes: bytes
    ) -> None:
        """Paragraphs made only of whitespace normalize away to nothing."""
        path = _write(tmp_path, "empty.docx", empty_docx_bytes)

        with pytest.raises(ExtractionError, match="No text"):
            extract_text(path, DOCX_CONTENT_TYPE)

    def test_corrupt_pdf_raises_extraction_error(
        self, tmp_path: Path, corrupt_pdf_bytes: bytes
    ) -> None:
        with pytest.raises(ExtractionError, match="could not be parsed"):
            extract_text(_write(tmp_path, "bad.pdf", corrupt_pdf_bytes), PDF_CONTENT_TYPE)

    def test_pdf_bytes_declared_as_docx_raise_extraction_error(
        self, tmp_path: Path, sample_pdf_bytes: bytes
    ) -> None:
        """A content type that lies about the payload fails in the DOCX reader, not silently."""
        with pytest.raises(ExtractionError, match="could not be parsed"):
            extract_text(_write(tmp_path, "liar.docx", sample_pdf_bytes), DOCX_CONTENT_TYPE)

    def test_missing_file_raises_extraction_error(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractionError):
            extract_text(tmp_path / "does-not-exist.pdf", PDF_CONTENT_TYPE)

    @pytest.mark.parametrize("content_type", ["text/plain", "image/png", ""])
    def test_unknown_content_type_is_rejected(self, tmp_path: Path, content_type: str) -> None:
        path = _write(tmp_path, "notes.txt", b"plain text")

        with pytest.raises(UnsupportedFileTypeError):
            extract_text(path, content_type)

    def test_unknown_content_type_message_avoids_empty_value(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "notes.txt", b"plain text")

        with pytest.raises(UnsupportedFileTypeError, match="unknown"):
            extract_text(path, "")
