# Phase 1 — Backend Skeleton & Document Ingestion

## Objective
Stand up the FastAPI application skeleton and a document ingestion endpoint that
accepts PDF and DOCX uploads, extracts clean text, persists the file, and returns
structured metadata. This is the foundation the rest of the RAG pipeline builds on.

**Not in this phase:** chunking, embeddings, vector store, retrieval, generation, UI.

## Dependencies to add this phase
Install, then add to `requirements.txt` (re-freeze after):
- `pypdf` — PDF text extraction
- `python-docx` — DOCX text extraction

Already installed: fastapi, uvicorn, pydantic, pydantic-settings, python-dotenv,
structlog, jinja2, python-multipart.

## Project structure to create
```
app/
├── __init__.py
├── main.py                 # FastAPI app, startup, router + handler registration
├── config.py               # Settings via pydantic-settings
├── core/
│   ├── __init__.py
│   ├── logging.py          # structlog JSON logging config
│   └── exceptions.py       # custom exceptions + FastAPI handlers
├── models/
│   ├── __init__.py
│   └── document.py         # Pydantic schemas
├── routers/
│   ├── __init__.py
│   ├── health.py           # GET /health
│   └── ingestion.py        # POST /ingest
└── services/
    ├── __init__.py
    ├── extraction.py       # PDF/DOCX text extraction
    └── storage.py          # save uploaded files
tests/
├── __init__.py
├── conftest.py
├── fixtures/               # tiny sample.pdf, sample.docx, corrupt.pdf
└── test_ingestion.py
data/
└── uploads/                # created at runtime; gitignored
```

## Configuration — app/config.py
Use `pydantic-settings` (`BaseSettings`), loaded from environment / `.env`:
- `app_name: str = "RAG Document Assistant"`
- `app_version: str = "0.1.0"`
- `log_level: str = "INFO"`
- `upload_dir: Path = Path("data/uploads")`
- `max_upload_bytes: int = 20 * 1024 * 1024`  # 20 MB
- `allowed_content_types: set[str] = {"application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}`

Expose a cached `get_settings()` for use as a FastAPI dependency.

## Logging — app/core/logging.py
Configure structlog for JSON output (reuse the Project 1 pattern). Provide
`configure_logging()` (called at startup) and `get_logger()`. Bind a per-request
`request_id` in the ingestion route.

## Exceptions — app/core/exceptions.py
Custom exceptions and matching FastAPI handlers that return clean JSON (no stack traces):
- `UnsupportedFileTypeError` → HTTP 415
- `FileTooLargeError` → HTTP 413
- `ExtractionError` → HTTP 422
Each handler returns `{ "error": <type>, "detail": <message> }`.

## Models — app/models/document.py
Pydantic v2 models:
- `DocumentMetadata`: `id: str` (uuid4), `filename: str`, `content_type: str`,
  `size_bytes: int`, `page_count: int | None`, `char_count: int`,
  `text_preview: str` (first ~500 chars of extracted text)
- `IngestResponse`: `status: str`, `document: DocumentMetadata`

The full extracted text is NOT returned in the response — only metadata + preview.
Persist the full text alongside the file (e.g. `{id}.txt`) for later phases.

## Extraction service — app/services/extraction.py
- Define a small result type (dataclass) `ExtractedDocument(text: str,
  page_count: int | None, char_count: int)`.
- `extract_text(file_path: Path, content_type: str) -> ExtractedDocument`:
  - PDF (`application/pdf`): use `pypdf`; iterate pages; join extracted page text;
    set `page_count` to the number of pages.
  - DOCX: use `python-docx`; iterate paragraphs; join with newlines; `page_count = None`.
  - Lightly normalize whitespace (strip, collapse 3+ blank lines to one) but preserve
    paragraph breaks.
  - Raise `ExtractionError` on parse failure OR if the extracted text is empty.

## Storage service — app/services/storage.py
- `save_upload(upload: UploadFile, settings) -> tuple[str, Path]`:
  - Generate a uuid4 `id`.
  - Stream the file to `upload_dir/{id}{ext}` in chunks (e.g. 1 MB), tracking bytes
    written; if it exceeds `max_upload_bytes`, delete the partial file and raise
    `FileTooLargeError`.
  - Return `(id, saved_path)`.
- `save_text(id: str, text: str, settings) -> Path`: write extracted text to
  `upload_dir/{id}.txt`. (Full dedup/versioning is deferred to Phase 7.)

## Routers
### app/routers/health.py
- `GET /health` → `{ "status": "ok", "app": <app_name>, "version": <app_version> }`

### app/routers/ingestion.py
- `POST /ingest`, `multipart/form-data`, `file: UploadFile`:
  1. Validate `file.content_type` against `allowed_content_types`; else
     `UnsupportedFileTypeError`.
  2. `storage.save_upload(...)` (enforces max size).
  3. `extraction.extract_text(...)` on the saved file.
  4. `storage.save_text(...)` with the full text.
  5. Build `DocumentMetadata` (preview = first 500 chars) and return `IngestResponse`.
  - Bind `request_id`; log start, success (with char_count, page_count), and failures.

## main.py
- Create `FastAPI(title=app_name, version=app_version)`.
- Startup: `configure_logging()`; ensure `upload_dir` exists.
- Register the exception handlers and both routers.
- Leave the default `/docs` (Swagger) enabled for manual testing.

## Tests — tests/
pytest + FastAPI `TestClient`:
- `GET /health` returns 200 and `status == "ok"`.
- Ingesting a small sample PDF returns 200, `char_count > 0`, `page_count >= 1`.
- Ingesting a small sample DOCX returns 200, `char_count > 0`, `page_count is None`.
- Unsupported type (e.g. a `.txt`/image) returns 415.
- Oversized upload returns 413.
- Corrupt PDF returns 422.
Provide tiny fixture files under `tests/fixtures/` (generate them in `conftest.py` if
easier). Extraction is local, so no mocking needed. Tests must be fast and deterministic.

## Acceptance criteria
- `uvicorn app.main:app --reload` starts with no errors.
- `GET /health` → 200 `{ "status": "ok", ... }`.
- `POST /ingest` with a real PDF and a real DOCX each return 200 with sensible metadata
  and a non-empty preview.
- Unsupported → 415, oversized → 413, corrupt → 422, all as clean JSON error bodies.
- The uploaded file and extracted `.txt` are persisted under `data/uploads/`
  (which is gitignored).
- All tests pass; `ruff check` and `ruff format` are clean.

## Out of scope (later phases)
Chunking (P2); embeddings + ChromaDB (P3); retrieval + generation (P4); contextual /
parent-document retrieval (P5); abstention/grounding (P6); idempotent dedup, embedding
cache, tracing (P7); evaluation harness + README (P8); frontend UI.