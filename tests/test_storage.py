"""Unit tests for the storage service.

`save_upload` is the only code path that writes attacker-influenced data to disk, so it
gets the closest scrutiny: where the name comes from, and what is left behind on failure.
"""

import asyncio
import io
import uuid
from pathlib import Path

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from app.config import DOCX_CONTENT_TYPE, PDF_CONTENT_TYPE, Settings
from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.services import storage


def _upload(payload: bytes, content_type: str | None, filename: str = "doc.pdf") -> UploadFile:
    """Build an UploadFile the way Starlette does when parsing multipart form data."""
    headers = Headers({"content-type": content_type}) if content_type is not None else Headers({})
    return UploadFile(file=io.BytesIO(payload), filename=filename, headers=headers)


def _save_full(upload: UploadFile, settings: Settings) -> tuple[str, Path, int]:
    return asyncio.run(storage.save_upload(upload, settings))


def _save(upload: UploadFile, settings: Settings) -> tuple[str, Path]:
    """Run `save_upload` and drop the byte count, which most tests don't assert on."""
    document_id, path, _size = _save_full(upload, settings)
    return document_id, path


class TestSaveUpload:
    def test_stores_file_under_generated_uuid(
        self, test_settings: Settings, upload_dir: Path
    ) -> None:
        document_id, path = _save(_upload(b"%PDF-1.4\ndata", PDF_CONTENT_TYPE), test_settings)

        assert uuid.UUID(document_id).version == 4
        assert path == upload_dir / f"{document_id}.pdf"
        assert path.read_bytes() == b"%PDF-1.4\ndata"

    def test_ids_are_unique_across_calls(self, test_settings: Settings, upload_dir: Path) -> None:
        first, _ = _save(_upload(b"a", PDF_CONTENT_TYPE), test_settings)
        second, _ = _save(_upload(b"b", PDF_CONTENT_TYPE), test_settings)

        assert first != second
        assert len(list(upload_dir.iterdir())) == 2

    @pytest.mark.parametrize(
        ("content_type", "expected_suffix"),
        [(PDF_CONTENT_TYPE, ".pdf"), (DOCX_CONTENT_TYPE, ".docx")],
    )
    def test_extension_comes_from_content_type(
        self, test_settings: Settings, content_type: str, expected_suffix: str
    ) -> None:
        _, path = _save(_upload(b"payload", content_type, filename="whatever.bin"), test_settings)

        assert path.suffix == expected_suffix

    def test_client_filename_never_reaches_the_path(self, test_settings: Settings) -> None:
        """A misleading double extension must not survive onto disk."""
        _, path = _save(
            _upload(b"payload", PDF_CONTENT_TYPE, filename="invoice.pdf.exe"), test_settings
        )

        assert path.suffix == ".pdf"
        assert "invoice" not in path.name

    @pytest.mark.parametrize(
        "filename",
        ["../../escape.pdf", "..\\..\\escape.pdf", "/etc/passwd", "sub/dir/nested.pdf"],
    )
    def test_traversal_filenames_stay_inside_upload_dir(
        self, test_settings: Settings, upload_dir: Path, filename: str
    ) -> None:
        _, path = _save(_upload(b"payload", PDF_CONTENT_TYPE, filename=filename), test_settings)

        assert path.parent == upload_dir
        assert path.resolve().is_relative_to(upload_dir.resolve())

    def test_creates_upload_dir_when_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "deeply" / "nested" / "uploads"
        settings = Settings(upload_dir=target, max_upload_bytes=1024)

        _, path = _save(_upload(b"payload", PDF_CONTENT_TYPE), settings)

        assert path.exists()
        assert path.parent == target

    @pytest.mark.parametrize("content_type", ["text/plain", "image/png", "", None])
    def test_unsupported_content_type_writes_nothing(
        self, test_settings: Settings, upload_dir: Path, content_type: str | None
    ) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            _save(_upload(b"payload", content_type), test_settings)

        assert list(upload_dir.iterdir()) == []

    def test_upload_exactly_at_the_limit_is_accepted(self, upload_dir: Path) -> None:
        settings = Settings(upload_dir=upload_dir, max_upload_bytes=64)

        _, path = _save(_upload(b"x" * 64, PDF_CONTENT_TYPE), settings)

        assert path.stat().st_size == 64

    def test_one_byte_over_the_limit_is_rejected(self, upload_dir: Path) -> None:
        settings = Settings(upload_dir=upload_dir, max_upload_bytes=64)

        with pytest.raises(FileTooLargeError, match="64"):
            _save(_upload(b"x" * 65, PDF_CONTENT_TYPE), settings)

        assert list(upload_dir.iterdir()) == [], "partial file must be cleaned up"

    def test_oversized_upload_leaves_no_partial_file(self, upload_dir: Path) -> None:
        """The cleanup must also hold when the overflow spans several read chunks."""
        settings = Settings(upload_dir=upload_dir, max_upload_bytes=storage.CHUNK_BYTES)
        oversized = b"y" * (storage.CHUNK_BYTES * 2 + 7)

        with pytest.raises(FileTooLargeError):
            _save(_upload(oversized, PDF_CONTENT_TYPE), settings)

        assert list(upload_dir.iterdir()) == []

    def test_multi_chunk_upload_is_written_whole(self, upload_dir: Path) -> None:
        settings = Settings(upload_dir=upload_dir, max_upload_bytes=storage.CHUNK_BYTES * 4)
        payload = bytes(range(256)) * (storage.CHUNK_BYTES // 128)

        _, path = _save(_upload(payload, PDF_CONTENT_TYPE), settings)

        assert path.read_bytes() == payload

    def test_read_failure_cleans_up_and_propagates(
        self, test_settings: Settings, upload_dir: Path
    ) -> None:
        """A mid-stream I/O error must not leave an orphaned file behind."""

        class ExplodingUpload(UploadFile):
            async def read(self, size: int = -1) -> bytes:
                raise OSError("disk gone")

        upload = ExplodingUpload(
            file=io.BytesIO(b"payload"),
            filename="doc.pdf",
            headers=Headers({"content-type": PDF_CONTENT_TYPE}),
        )

        with pytest.raises(OSError, match="disk gone"):
            _save(upload, test_settings)

        assert list(upload_dir.iterdir()) == []


class TestSaveText:
    def test_writes_utf8_text_next_to_the_upload(
        self, test_settings: Settings, upload_dir: Path
    ) -> None:
        path = storage.save_text("abc-123", "grounded answers — café", test_settings)

        assert path == upload_dir / "abc-123.txt"
        assert path.read_text(encoding="utf-8") == "grounded answers — café"

    def test_creates_missing_directory(self, tmp_path: Path) -> None:
        settings = Settings(upload_dir=tmp_path / "fresh" / "uploads")

        path = storage.save_text("abc-123", "text", settings)

        assert path.read_text(encoding="utf-8") == "text"

    def test_overwrites_previous_text_for_the_same_id(self, test_settings: Settings) -> None:
        storage.save_text("abc-123", "first", test_settings)
        path = storage.save_text("abc-123", "second", test_settings)

        assert path.read_text(encoding="utf-8") == "second"
