"""Tests for the chunking service, its persistence, and the chunks endpoint."""

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import DOCX_CONTENT_TYPE, PDF_CONTENT_TYPE, Settings
from app.core.exceptions import DocumentNotFoundError
from app.models.chunk import Chunk
from app.services import storage
from app.services.chunking import chunk_document
from tests.conftest import _build_docx

DOC_ID = "11111111-2222-3333-4444-555555555555"
DOC_METADATA = {"filename": "report.pdf", "content_type": PDF_CONTENT_TYPE, "page_count": 3}

#: Japanese, accented Latin, and an astral-plane emoji: one code point each in Python,
#: but 2-4 bytes in UTF-8. Chunk offsets are code-point offsets and must stay that way.
UNICODE_LINE = "段落{i}。検索拡張生成のテストです — naïve café 🚀 grounded answers cite sources."


def _long_text(paragraphs: int = 40) -> str:
    """Several paragraphs of prose, comfortably longer than one chunk."""
    return "\n\n".join(
        f"Paragraph {i} discusses retrieval augmented generation. "
        f"It explains how grounded answers cite their sources carefully."
        for i in range(paragraphs)
    )


def _unicode_text(lines: int = 30) -> str:
    return "\n\n".join(UNICODE_LINE.format(i=i) for i in range(lines))


def _settings(upload_dir: Path, **overrides: object) -> Settings:
    """Settings pinned to the per-test upload dir, with chunking knobs overridden."""
    return Settings(upload_dir=upload_dir, **overrides)


def _covered_characters(text: str, chunks: list[Chunk]) -> set[int]:
    """Character positions of ``text`` that at least one chunk spans."""
    covered: set[int] = set()
    for chunk in chunks:
        covered.update(range(chunk.start_index, chunk.start_index + chunk.char_count))
    return covered


@pytest.fixture
def chunks(test_settings: Settings) -> list[Chunk]:
    return chunk_document(_long_text(), DOC_ID, dict(DOC_METADATA), test_settings)


class TestChunkDocument:
    def test_long_text_produces_multiple_chunks(self, chunks: list[Chunk]) -> None:
        assert len(chunks) > 1

    def test_short_text_produces_exactly_one_chunk(self, test_settings: Settings) -> None:
        result = chunk_document("A short sentence.", DOC_ID, dict(DOC_METADATA), test_settings)

        assert len(result) == 1
        assert result[0].text == "A short sentence."
        assert result[0].chunk_index == 0

    def test_chunk_indexes_are_sequential_from_zero(self, chunks: list[Chunk]) -> None:
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_ids_are_deterministic(self, chunks: list[Chunk]) -> None:
        assert [c.id for c in chunks] == [f"{DOC_ID}:{i}" for i in range(len(chunks))]

    def test_consecutive_chunks_overlap(self, chunks: list[Chunk], test_settings: Settings) -> None:
        """The tail of each chunk must reappear at the head of the next."""
        assert len(chunks) > 1
        overlapping = 0
        for current, following in zip(chunks, chunks[1:], strict=False):
            tail = current.text[-test_settings.chunk_overlap :]
            # Compare on a word, since the splitter breaks on natural boundaries.
            if tail.split() and tail.split()[-1] in following.text:
                overlapping += 1
        assert overlapping == len(chunks) - 1

    def test_start_index_is_present_and_non_decreasing(self, chunks: list[Chunk]) -> None:
        offsets = [c.start_index for c in chunks]

        assert offsets[0] == 0
        assert offsets == sorted(offsets)

    def test_start_index_locates_the_chunk_in_the_source(self, test_settings: Settings) -> None:
        """The offset must be usable for citations — it should point at the real text."""
        text = _long_text()
        result = chunk_document(text, DOC_ID, dict(DOC_METADATA), test_settings)

        for chunk in result:
            assert text[chunk.start_index : chunk.start_index + chunk.char_count] == chunk.text

    def test_document_metadata_is_propagated_to_every_chunk(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            assert chunk.metadata["filename"] == "report.pdf"
            assert chunk.metadata["content_type"] == PDF_CONTENT_TYPE
            assert chunk.metadata["page_count"] == 3

    def test_start_index_is_not_duplicated_inside_metadata(self, chunks: list[Chunk]) -> None:
        """It belongs on the typed field, not smuggled into the propagated dict."""
        for chunk in chunks:
            assert "start_index" not in chunk.metadata

    def test_chunks_respect_the_configured_size(
        self, chunks: list[Chunk], test_settings: Settings
    ) -> None:
        for chunk in chunks:
            assert chunk.char_count <= test_settings.chunk_size

    def test_counts_are_consistent(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            assert chunk.char_count == len(chunk.text)
            assert chunk.token_estimate == chunk.char_count // 4

    def test_document_id_is_carried_on_each_chunk(self, chunks: list[Chunk]) -> None:
        assert {c.document_id for c in chunks} == {DOC_ID}


class TestChunkPersistence:
    def test_round_trips_through_disk(self, chunks: list[Chunk], test_settings: Settings) -> None:
        storage.save_chunks(DOC_ID, chunks, test_settings)

        assert storage.load_chunks(DOC_ID, test_settings) == (len(chunks), chunks)

    def test_saves_utf8_json_next_to_the_upload(
        self, chunks: list[Chunk], test_settings: Settings, upload_dir: Path
    ) -> None:
        path = storage.save_chunks(DOC_ID, chunks, test_settings)

        assert path == upload_dir / f"{DOC_ID}.chunks.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload) == len(chunks)
        assert payload[0]["id"] == f"{DOC_ID}:0"

    def test_missing_file_raises_not_found(self, test_settings: Settings) -> None:
        with pytest.raises(DocumentNotFoundError):
            storage.load_chunks(DOC_ID, test_settings)

    def test_corrupt_file_raises_not_found_rather_than_crashing(
        self, test_settings: Settings, upload_dir: Path
    ) -> None:
        (upload_dir / f"{DOC_ID}.chunks.json").write_text("{not json", encoding="utf-8")

        with pytest.raises(DocumentNotFoundError):
            storage.load_chunks(DOC_ID, test_settings)

    def test_schema_violating_file_raises_not_found(
        self, test_settings: Settings, upload_dir: Path
    ) -> None:
        (upload_dir / f"{DOC_ID}.chunks.json").write_text('[{"id": 1}]', encoding="utf-8")

        with pytest.raises(DocumentNotFoundError):
            storage.load_chunks(DOC_ID, test_settings)


class TestIngestReportsChunks:
    def test_ingest_response_includes_chunk_count(
        self, client: TestClient, sample_pdf_bytes: bytes
    ) -> None:
        response = client.post(
            "/ingest", files={"file": ("sample.pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)}
        )

        assert response.status_code == 200, response.text
        assert response.json()["chunk_count"] > 0

    def test_chunk_count_matches_what_was_persisted(
        self, client: TestClient, long_docx_bytes: bytes, upload_dir: Path
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
        persisted = json.loads(
            (upload_dir / f"{body['document']['id']}.chunks.json").read_text(encoding="utf-8")
        )
        assert body["chunk_count"] == len(persisted)

    def test_failed_ingest_leaves_no_chunk_file(
        self, client: TestClient, corrupt_pdf_bytes: bytes, upload_dir: Path
    ) -> None:
        client.post("/ingest", files={"file": ("bad.pdf", corrupt_pdf_bytes, PDF_CONTENT_TYPE)})

        assert list(upload_dir.iterdir()) == []


class TestGetChunksEndpoint:
    def _ingest(self, client: TestClient, payload: bytes) -> str:
        response = client.post("/ingest", files={"file": ("sample.pdf", payload, PDF_CONTENT_TYPE)})
        return response.json()["document"]["id"]

    def test_returns_the_persisted_chunks(
        self, client: TestClient, sample_pdf_bytes: bytes
    ) -> None:
        document_id = self._ingest(client, sample_pdf_bytes)

        response = client.get(f"/documents/{document_id}/chunks")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["document_id"] == document_id
        assert body["total_chunks"] == len(body["chunks"])
        assert body["chunks"][0]["id"] == f"{document_id}:0"
        assert body["chunks"][0]["text"]

    def test_limit_caps_the_list_but_not_the_total(
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
        document_id = response.json()["document"]["id"]
        total = response.json()["chunk_count"]
        assert total > 1, "fixture must produce enough chunks to exercise the limit"

        limited = client.get(f"/documents/{document_id}/chunks?limit=1").json()

        assert len(limited["chunks"]) == 1
        assert limited["total_chunks"] == total

    def test_unknown_but_valid_id_returns_404(self, client: TestClient) -> None:
        response = client.get(f"/documents/{uuid.uuid4()}/chunks")

        assert response.status_code == 404
        assert response.json()["error"] == "DocumentNotFoundError"

    @pytest.mark.parametrize("limit", [0, -1, 501])
    def test_out_of_range_limit_is_rejected(self, client: TestClient, limit: int) -> None:
        response = client.get(f"/documents/{uuid.uuid4()}/chunks?limit={limit}")

        assert response.status_code == 422


class TestGetChunksPathTraversal:
    """`document_id` is the first client-controlled value to reach a file path."""

    @pytest.mark.parametrize(
        "document_id",
        [
            r"..\..\secrets",
            "../../etc/passwd",
            r"..\..\..\..\Windows\win.ini",
            "not-a-uuid",
            "",
            "%2e%2e%2fsecrets",
        ],
    )
    def test_non_uuid_ids_are_refused(self, client: TestClient, document_id: str) -> None:
        response = client.get(f"/documents/{document_id}/chunks")

        assert response.status_code in (404, 422)
        assert "win.ini" not in response.text
        assert "root:" not in response.text

    def test_traversal_cannot_read_a_file_outside_the_upload_dir(
        self, client: TestClient, upload_dir: Path, tmp_path: Path
    ) -> None:
        """Plant a real file one level up; a traversal id must not reach it."""
        outside = tmp_path / "outside.chunks.json"
        outside.write_text('[{"id": "leaked:0"}]', encoding="utf-8")

        response = client.get(r"/documents/..\outside/chunks")

        assert response.status_code in (404, 422)
        assert "leaked" not in response.text

    def test_error_body_does_not_reflect_the_supplied_id(self, client: TestClient) -> None:
        """The id is attacker-controlled; echoing it back invites reflected content."""
        response = client.get("/documents/<script>alert(1)</script>/chunks")

        assert "<script>" not in response.text


class TestChunkSizeBoundary:
    """Text sitting exactly at the size limit is where an off-by-one would show."""

    @pytest.mark.parametrize("length", [1, 99, 100])
    def test_text_at_or_below_chunk_size_stays_one_chunk(
        self, upload_dir: Path, length: int
    ) -> None:
        settings = _settings(upload_dir, chunk_size=100, chunk_overlap=20)
        text = "a" * length

        result = chunk_document(text, DOC_ID, dict(DOC_METADATA), settings)

        assert len(result) == 1
        assert result[0].text == text
        assert result[0].start_index == 0

    def test_one_character_over_chunk_size_splits(self, upload_dir: Path) -> None:
        settings = _settings(upload_dir, chunk_size=100, chunk_overlap=20)
        text = "a" * 101

        result = chunk_document(text, DOC_ID, dict(DOC_METADATA), settings)

        assert len(result) > 1
        assert all(chunk.char_count <= 100 for chunk in result)
        assert _covered_characters(text, result) == set(range(len(text)))

    def test_a_word_boundary_run_at_the_limit_is_not_split(self, upload_dir: Path) -> None:
        """Exactly `chunk_size` characters of real prose must still be a single chunk."""
        settings = _settings(upload_dir, chunk_size=100, chunk_overlap=20)
        text = ("word " * 20).strip()  # 99 characters, splittable but under the limit
        assert len(text) == 99

        assert len(chunk_document(text, DOC_ID, dict(DOC_METADATA), settings)) == 1


class TestTextWithNoSeparatorMatch:
    """One unbroken token: every separator misses until the "" hard cut."""

    @pytest.fixture
    def unbroken(self) -> str:
        return "A" * 5000

    def test_unbroken_text_is_still_split_at_the_size_limit(
        self, unbroken: str, test_settings: Settings
    ) -> None:
        result = chunk_document(unbroken, DOC_ID, dict(DOC_METADATA), test_settings)

        assert len(result) > 1
        assert all(chunk.char_count <= test_settings.chunk_size for chunk in result)

    def test_unbroken_text_loses_no_characters(
        self, unbroken: str, test_settings: Settings
    ) -> None:
        result = chunk_document(unbroken, DOC_ID, dict(DOC_METADATA), test_settings)

        assert _covered_characters(unbroken, result) == set(range(len(unbroken)))

    def test_hard_cut_offsets_advance_by_the_configured_stride(self, upload_dir: Path) -> None:
        """With no separator to snap to, chunks step by `chunk_size - chunk_overlap`."""
        settings = _settings(upload_dir, chunk_size=100, chunk_overlap=20)
        text = "A" * 350

        result = chunk_document(text, DOC_ID, dict(DOC_METADATA), settings)

        assert [chunk.start_index for chunk in result] == [0, 80, 160, 240, 320]

    def test_the_empty_separator_is_what_enforces_the_size_cap(self, upload_dir: Path) -> None:
        """Drop the terminal "" and an unbroken run exceeds `chunk_size` — that is why
        the configured separator list ends with it."""
        without_hard_cut = _settings(
            upload_dir, chunk_size=100, chunk_overlap=20, chunk_separators=["\n\n"]
        )
        text = "A" * 350

        oversized = chunk_document(text, DOC_ID, dict(DOC_METADATA), without_hard_cut)

        assert len(oversized) == 1
        assert oversized[0].char_count == 350

    def test_whitespace_only_text_yields_no_chunks_rather_than_raising(
        self, test_settings: Settings
    ) -> None:
        """Phase 1 rejects this upstream; chunking must still degrade quietly."""
        assert chunk_document("   \n\n\t ", DOC_ID, dict(DOC_METADATA), test_settings) == []


class TestUnicodeText:
    """Offsets are character offsets, not byte offsets — multi-byte text must not skew them."""

    @pytest.fixture
    def unicode_chunks(self, test_settings: Settings) -> list[Chunk]:
        return chunk_document(_unicode_text(), DOC_ID, dict(DOC_METADATA), test_settings)

    def test_offsets_locate_the_chunk_in_the_source(self, unicode_chunks: list[Chunk]) -> None:
        text = _unicode_text()

        for chunk in unicode_chunks:
            assert text[chunk.start_index : chunk.start_index + chunk.char_count] == chunk.text

    def test_char_count_counts_code_points_not_bytes(self, unicode_chunks: list[Chunk]) -> None:
        for chunk in unicode_chunks:
            assert chunk.char_count == len(chunk.text)
            assert chunk.char_count < len(chunk.text.encode("utf-8"))

    def test_astral_characters_survive_intact(self, unicode_chunks: list[Chunk]) -> None:
        """A chunk boundary must never leave half a surrogate pair or a mojibake tail."""
        joined = "".join(chunk.text for chunk in unicode_chunks)

        assert "🚀" in joined
        assert "café" in joined
        assert "�" not in joined
        for chunk in unicode_chunks:
            chunk.text.encode("utf-8")  # raises on a lone surrogate

    def test_unicode_survives_the_json_round_trip(
        self, unicode_chunks: list[Chunk], test_settings: Settings
    ) -> None:
        path = storage.save_chunks(DOC_ID, unicode_chunks, test_settings)

        assert storage.load_chunks(DOC_ID, test_settings) == (len(unicode_chunks), unicode_chunks)
        # ensure_ascii=False: the file holds real UTF-8, not \u escapes.
        assert "検索" in path.read_text(encoding="utf-8")

    def test_unicode_document_ingests_and_reads_back_intact(
        self, client: TestClient, upload_dir: Path
    ) -> None:
        payload = _build_docx(tuple(UNICODE_LINE.format(i=i) for i in range(30)))

        response = client.post(
            "/ingest", files={"file": ("unicode.docx", payload, DOCX_CONTENT_TYPE)}
        )

        assert response.status_code == 200, response.text
        document_id = response.json()["document"]["id"]
        body = client.get(f"/documents/{document_id}/chunks").json()
        assert body["total_chunks"] == response.json()["chunk_count"]
        assert "🚀" in "".join(chunk["text"] for chunk in body["chunks"])

        # Offsets must still address the text as it was persisted.
        stored = (upload_dir / f"{document_id}.txt").read_text(encoding="utf-8")
        for chunk in body["chunks"]:
            start = chunk["start_index"]
            assert stored[start : start + chunk["char_count"]] == chunk["text"]


class TestOverlapRelativeToChunkSize:
    def test_large_overlap_still_covers_the_whole_text(self, upload_dir: Path) -> None:
        """The maximum permitted overlap (50%) may duplicate text, never drop it."""
        settings = _settings(upload_dir, chunk_size=800, chunk_overlap=400)
        text = _long_text()

        result = chunk_document(text, DOC_ID, dict(DOC_METADATA), settings)

        assert _covered_characters(text, result) == set(range(len(text)))
        assert all(chunk.char_count <= 800 for chunk in result)
        assert [c.start_index for c in result] == sorted(c.start_index for c in result)

    def test_zero_overlap_is_allowed_and_produces_disjoint_chunks(self, upload_dir: Path) -> None:
        settings = _settings(upload_dir, chunk_size=100, chunk_overlap=0)
        text = "A" * 350

        result = chunk_document(text, DOC_ID, dict(DOC_METADATA), settings)

        assert [chunk.start_index for chunk in result] == [0, 100, 200, 300]

    def test_overlap_that_would_drop_text_is_refused_by_config(self, upload_dir: Path) -> None:
        """Regression: chunk_size=100/overlap=99 silently dropped most of a document.

        The splitter could not advance a full split, so it emitted the same chunk 81
        times covering only the first 99 characters. The remaining 400 were lost with no
        error — in Phase 3 that is document text that never gets indexed. The config
        validator now refuses the window outright.
        """
        with pytest.raises(ValidationError, match="chunk_overlap"):
            _settings(upload_dir, chunk_size=100, chunk_overlap=99)

    def test_maximum_permitted_overlap_covers_all_text(self, upload_dir: Path) -> None:
        """The boundary the validator does allow must still lose nothing."""
        settings = _settings(upload_dir, chunk_size=100, chunk_overlap=50)
        text = ("word " * 100).strip()

        result = chunk_document(text, DOC_ID, dict(DOC_METADATA), settings)

        assert _covered_characters(text, result) == set(range(len(text)))
        assert all(chunk.char_count <= 100 for chunk in result)


class TestSaveChunksOverwrite:
    """Re-ingesting under a known id must replace the chunk file, not merge into it."""

    def test_rewriting_replaces_the_previous_chunks(
        self, chunks: list[Chunk], test_settings: Settings
    ) -> None:
        storage.save_chunks(DOC_ID, chunks, test_settings)
        assert len(chunks) > 1

        replacement = chunk_document(
            "A single short replacement.", DOC_ID, dict(DOC_METADATA), test_settings
        )
        storage.save_chunks(DOC_ID, replacement, test_settings)

        _total, reloaded = storage.load_chunks(DOC_ID, test_settings)
        assert reloaded == replacement
        assert len(reloaded) == 1

    def test_rewriting_leaves_no_trailing_bytes_from_the_longer_file(
        self, chunks: list[Chunk], test_settings: Settings, upload_dir: Path
    ) -> None:
        """A truncating write, not an in-place one: stale JSON would fail to parse."""
        path = storage.save_chunks(DOC_ID, chunks, test_settings)
        long_size = path.stat().st_size

        storage.save_chunks(
            DOC_ID,
            chunk_document("Short.", DOC_ID, dict(DOC_METADATA), test_settings),
            test_settings,
        )

        assert path.stat().st_size < long_size
        assert len(json.loads(path.read_text(encoding="utf-8"))) == 1
        assert [p.name for p in upload_dir.iterdir()] == [f"{DOC_ID}.chunks.json"]

    def test_saving_an_empty_list_round_trips(self, test_settings: Settings) -> None:
        storage.save_chunks(DOC_ID, [], test_settings)

        assert storage.load_chunks(DOC_ID, test_settings) == (0, [])

    @pytest.mark.parametrize("body", ["null", '{"chunks": []}', "[]extra", '["not a chunk"]'])
    def test_malformed_files_are_reported_as_missing(
        self, test_settings: Settings, upload_dir: Path, body: str
    ) -> None:
        (upload_dir / f"{DOC_ID}.chunks.json").write_text(body, encoding="utf-8")

        with pytest.raises(DocumentNotFoundError):
            storage.load_chunks(DOC_ID, test_settings)


class TestRepeatedIngestion:
    def _chunk_files(self, upload_dir: Path) -> list[Path]:
        return sorted(upload_dir.glob("*.chunks.json"))

    def test_repeated_ingests_write_independent_chunk_files(
        self, client: TestClient, sample_pdf_bytes: bytes, upload_dir: Path
    ) -> None:
        ids = []
        for _ in range(3):
            response = client.post(
                "/ingest", files={"file": ("sample.pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)}
            )
            assert response.status_code == 200, response.text
            ids.append(response.json()["document"]["id"])

        assert len(set(ids)) == 3
        assert len(self._chunk_files(upload_dir)) == 3
        for document_id in ids:
            body = client.get(f"/documents/{document_id}/chunks").json()
            assert body["chunks"][0]["id"] == f"{document_id}:0"
            assert body["chunks"][0]["document_id"] == document_id

    def test_concurrent_ingests_do_not_clobber_each_other(
        self, client: TestClient, long_docx_bytes: bytes, upload_dir: Path
    ) -> None:
        """Ids are generated per request; parallel uploads must stay fully separate."""

        def ingest() -> dict:
            return client.post(
                "/ingest", files={"file": ("long.docx", long_docx_bytes, DOCX_CONTENT_TYPE)}
            ).json()

        with ThreadPoolExecutor(max_workers=4) as pool:
            bodies = [future.result() for future in [pool.submit(ingest) for _ in range(4)]]

        ids = [body["document"]["id"] for body in bodies]
        assert len(set(ids)) == 4
        assert len(self._chunk_files(upload_dir)) == 4
        for body in bodies:
            document_id = body["document"]["id"]
            persisted = json.loads(
                (upload_dir / f"{document_id}.chunks.json").read_text(encoding="utf-8")
            )
            assert len(persisted) == body["chunk_count"]
            assert {entry["document_id"] for entry in persisted} == {document_id}


class TestIngestCleanupWhenChunkingFails:
    """Half-ingested state is worse than none: Phase 3 would index text with no chunks."""

    def test_chunking_failure_removes_upload_and_text(
        self,
        client: TestClient,
        sample_pdf_bytes: bytes,
        upload_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(*_args: object, **_kwargs: object) -> list[Chunk]:
            raise RuntimeError("splitter exploded")

        monkeypatch.setattr("app.routers.ingestion.chunk_document", boom)

        with pytest.raises(RuntimeError):
            client.post(
                "/ingest", files={"file": ("sample.pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)}
            )

        assert list(upload_dir.iterdir()) == []

    def test_chunk_persistence_failure_removes_every_artifact(
        self,
        client: TestClient,
        sample_pdf_bytes: bytes,
        upload_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        real_save = storage.save_chunks

        def save_then_fail(document_id: str, chunk_list: list[Chunk], settings: Settings) -> Path:
            real_save(document_id, chunk_list, settings)
            raise OSError("disk full")

        monkeypatch.setattr("app.services.storage.save_chunks", save_then_fail)

        with pytest.raises(OSError, match="disk full"):
            client.post(
                "/ingest", files={"file": ("sample.pdf", sample_pdf_bytes, PDF_CONTENT_TYPE)}
            )

        assert list(upload_dir.iterdir()) == []


class TestIdCanonicalization:
    """`uuid.UUID` is lenient; only the canonical spelling may reach the filesystem."""

    @pytest.mark.parametrize(
        "variant",
        [
            "urn:uuid:11111111-2222-3333-4444-555555555555",
            "{11111111-2222-3333-4444-555555555555}",
            "11111111222233334444555555555555",
            "11111111-2222-3333-4444-555555555555 ",
            "11111111_2222_3333_4444_555555555555",
            "11111111-2222-3333-4444-555555555555５",
        ],
    )
    def test_non_canonical_spellings_are_refused(
        self, client: TestClient, chunks: list[Chunk], test_settings: Settings, variant: str
    ) -> None:
        storage.save_chunks(DOC_ID, chunks, test_settings)

        response = client.get(f"/documents/{variant}/chunks")

        assert response.status_code in (404, 422)
        assert "chunks" not in response.json()

    def test_canonical_spelling_is_served(
        self, client: TestClient, chunks: list[Chunk], test_settings: Settings
    ) -> None:
        storage.save_chunks(DOC_ID, chunks, test_settings)

        response = client.get(f"/documents/{DOC_ID}/chunks")

        assert response.status_code == 200, response.text
        assert response.json()["total_chunks"] == len(chunks)

    def test_uppercase_is_refused_so_one_document_has_one_id(
        self, client: TestClient, chunks: list[Chunk], test_settings: Settings
    ) -> None:
        """Two spellings of one id would break any future cache or dedup keyed on it."""
        hex_id = "abcdef01-2345-4678-9abc-def012345678"
        storage.save_chunks(hex_id, chunks, test_settings)
        assert hex_id.upper() != hex_id

        assert client.get(f"/documents/{hex_id}/chunks").status_code == 200
        assert client.get(f"/documents/{hex_id.upper()}/chunks").status_code == 404


class TestChunkFileReadRobustness:
    def test_non_utf8_file_is_a_404_not_a_500(
        self, client: TestClient, test_settings: Settings, upload_dir: Path
    ) -> None:
        """A truncated write leaves invalid UTF-8; read_text raises UnicodeDecodeError."""
        (upload_dir / f"{DOC_ID}.chunks.json").write_bytes(b'[{"id": "\xff\xfe\x00garbage"}]')

        with pytest.raises(DocumentNotFoundError):
            storage.load_chunks(DOC_ID, test_settings)

        assert client.get(f"/documents/{DOC_ID}/chunks").status_code == 404

    def test_oversize_chunk_file_is_refused(
        self, test_settings: Settings, upload_dir: Path
    ) -> None:
        """Serving a huge chunk file repeatedly is a memory-amplification vector."""
        settings = _settings(upload_dir, max_chunks_file_bytes=64)
        (upload_dir / f"{DOC_ID}.chunks.json").write_text("[" + '{"x":1},' * 50 + "]")

        with pytest.raises(DocumentNotFoundError):
            storage.load_chunks(DOC_ID, settings)

    def test_limit_avoids_validating_chunks_it_will_not_return(
        self, chunks: list[Chunk], test_settings: Settings
    ) -> None:
        """total reflects the whole file; only the returned window is built into models."""
        storage.save_chunks(DOC_ID, chunks, test_settings)

        total, returned = storage.load_chunks(DOC_ID, test_settings, limit=1)

        assert total == len(chunks)
        assert len(returned) == 1

    def test_error_body_never_reflects_a_valid_looking_id(self, client: TestClient) -> None:
        missing = "99999999-8888-7777-6666-555555555555"

        response = client.get(f"/documents/{missing}/chunks")

        assert response.status_code == 404
        assert missing not in response.text
