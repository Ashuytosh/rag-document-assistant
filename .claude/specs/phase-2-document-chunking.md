# Phase 2 — Document Chunking

## Objective
Take the clean text produced in Phase 1 and split each document into well-formed,
overlapping chunks with propagated metadata, then persist them for later embedding.
This is the last text-only phase before vectors.

**Not in this phase:** embeddings, vector store, retrieval, generation (Phase 3+);
parent-document / semantic / contextual chunking (Phase 5); token-accurate sizing via
the real tokenizer (Phase 3).

## Dependencies to add this phase
Install, then add to `requirements.txt` (re-freeze after):
- `langchain-text-splitters` — provides `RecursiveCharacterTextSplitter`

This is the project's first LangChain component.

## Project structure changes
Add:
```
app/
├── models/
│   └── chunk.py            # Chunk, ChunksResponse
├── services/
│   └── chunking.py         # ChunkingService (RecursiveCharacterTextSplitter)
```
Modify: `app/config.py`, `app/services/storage.py`, `app/routers/ingestion.py`,
`app/models/document.py` (add `chunk_count` to the ingest response). Add tests in
`tests/test_chunking.py`.

## Configuration — app/config.py (additions)
- `chunk_size: int = 800`          # characters; sized to stay within the embedding
                                    # model's 256-token (~1000 char) limit, with margin
- `chunk_overlap: int = 120`       # characters shared between consecutive chunks
- `chunk_separators: list[str] = ["\n\n", "\n", ". ", " ", ""]`
                                    # split priority: paragraph → line → sentence → word

Add a validator asserting `chunk_overlap < chunk_size`.

## Models — app/models/chunk.py
Pydantic v2:
- `Chunk`:
  - `id: str`               # deterministic: f"{document_id}:{chunk_index}"
  - `document_id: str`
  - `chunk_index: int`      # sequential, starting at 0
  - `text: str`
  - `char_count: int`
  - `token_estimate: int`   # char_count // 4 (rough; replaced by real tokens in P3)
  - `start_index: int`      # char offset in the source text (citation groundwork)
  - `metadata: dict`        # {filename, content_type, page_count}
- `ChunksResponse`:
  - `document_id: str`
  - `total_chunks: int`
  - `chunks: list[Chunk]`   # capped by the endpoint's limit

Add `chunk_count: int` to `IngestResponse` in `app/models/document.py`.

## Chunking service — app/services/chunking.py
- `chunk_document(text: str, document_id: str, doc_metadata: dict, settings) -> list[Chunk]`:
  - Build a `RecursiveCharacterTextSplitter` with `chunk_size`, `chunk_overlap`,
    `separators=settings.chunk_separators`, `add_start_index=True`, and
    `length_function=len` (character-based for now).
  - Use `splitter.create_documents([text], metadatas=[doc_metadata])` so each split
    carries the document metadata and a `start_index`.
  - Map each split to a `Chunk`: deterministic `id`, sequential `chunk_index`,
    `char_count`, `token_estimate = char_count // 4`, `start_index` from metadata,
    and the propagated `metadata`.
  - Return the list. If `text` is short enough it yields a single chunk — that's fine.
  - Raise nothing new here; empty text was already rejected in Phase 1 extraction.

## Storage additions — app/services/storage.py
- `save_chunks(document_id: str, chunks: list[Chunk], settings) -> Path`:
  write `upload_dir/{document_id}.chunks.json` (UTF-8) as a JSON list of chunk dicts.
- `load_chunks(document_id: str, settings) -> list[Chunk]`:
  read and parse that file; raise a clear 404-mapped error if it doesn't exist.

(This JSON persistence is interim — Phase 3 moves chunks into the vector store.)

## Router changes — app/routers/ingestion.py
- Extend `POST /ingest`: after `save_text`, call `chunk_document`, then `save_chunks`,
  and set `chunk_count` on the response. Log the chunk count.
- Add `GET /documents/{document_id}/chunks?limit=50`:
  - `load_chunks`, return a `ChunksResponse` with `total_chunks` and up to `limit` chunks.
  - If no chunks exist for the id, return 404 with a clean JSON error.

## Tests — tests/test_chunking.py
pytest:
- A long text (several paragraphs) produces more than one chunk.
- Consecutive chunks overlap: the end of chunk *i* shares text with the start of chunk *i+1*.
- `chunk_index` is sequential from 0; `id` matches `f"{document_id}:{i}"`.
- `start_index` is present and non-decreasing across chunks.
- Document metadata (filename, content_type) is present on every chunk.
- A very short text yields exactly one chunk.
- `POST /ingest` response now includes `chunk_count > 0`.
- `GET /documents/{id}/chunks` returns the persisted chunks and respects `limit`.
- Requesting chunks for an unknown id returns 404.
Keep tests fast and deterministic; no network, no models.

## Acceptance criteria
- Installing `langchain-text-splitters` and re-freezing `requirements.txt` succeeds.
- Ingesting a multi-page PDF chunks the text; `chunk_count` in the response equals the
  number of persisted chunks.
- Each chunk carries text, sequential index, `start_index`, and propagated metadata.
- `GET /documents/{id}/chunks` works and respects `limit`; unknown id → 404.
- Chunk sizes stay at or below `chunk_size` (recursive splitting may occasionally land
  slightly under on natural boundaries — that's expected).
- All tests pass; `ruff check` and `ruff format` are clean.

## Out of scope (later phases)
Embeddings + ChromaDB (P3); retrieval + generation (P4); parent-document, semantic, and
contextual chunking (P5); grounding/abstention (P6); dedup + tracing (P7);
evaluation + README (P8); frontend UI.