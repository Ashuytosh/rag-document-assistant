# Phase 6 — Parent-Document Retrieval (Small-to-Big)

## Objective
Decouple the unit you *match* from the unit you *answer with*: embed small "child" chunks
for sharp retrieval, but feed the larger "parent" chunk to the LLM for full context. This
improves answer grounding without changing the UI or the generation prompt.

**Not in this phase:** contextual retrieval (Phase 7), abstention (Phase 8),
dedup/observability (Phase 9), evaluation (Phase 10). No LLM calls at ingestion.

## Dependencies to add this phase
None (reuse `langchain-text-splitters`).

## Concept to build toward
- Split each document into large PARENT chunks, then split each parent into small CHILD
  chunks that carry a `parent_id`.
- Embed and store ONLY children in Chroma (with `parent_id` + document metadata).
- Persist parents to a parent store keyed by id.
- Retrieval: search children (over-retrieve, e.g. top 15) → collect their unique
  `parent_id`s → load those parents → return up to `max_parents` parents (deduplicated,
  scored by each parent's best child) to generation. The LLM answers from parents;
  citations reference parents.

## Project structure changes
Modify:
- `app/services/chunking.py` — produce parents + children.
- `app/services/storage.py` — `save_parents` / `load_parents` / `load_parent`.
- `app/services/vector_store.py` — `add_children`; `search` resolves children → parents.
- `app/services/generation.py` — build context + sources from parents.
- `app/models/chunk.py` — `ParentChunk`, `ChildChunk`.
- `app/models/document.py` — report `parent_count` and `child_count` in the ingest response.
- `app/config.py` — parent/child sizes, retrieval knobs.
- `app/routers/ingestion.py` — new ingest flow.
Add `tests/test_parent_retrieval.py`.

## Configuration — app/config.py (additions)
- `parent_chunk_size: int = 2000`
- `parent_chunk_overlap: int = 200`
- `child_chunk_size: int = 400`
- `child_chunk_overlap: int = 80`
- `retrieval_child_k: int = 15`     # children fetched before collapsing to parents
- `max_parents: int = 4`            # parents passed to the LLM after dedup

## Chunking — app/services/chunking.py
- `chunk_document(text, document_id, doc_metadata, settings)
  -> tuple[list[ParentChunk], list[ChildChunk]]`:
  1. Parent splitter (`RecursiveCharacterTextSplitter`, parent size/overlap,
     `add_start_index=True`) over the document → parents; `id = f"{document_id}:p{idx}"`;
     carry `start_index` and metadata.
  2. For each parent, a child splitter (child size/overlap, `add_start_index=True`) over
     that parent's text → children; `id = f"{parent_id}:c{idx}"`; set
     `child.parent_id = parent_id`; compute an ABSOLUTE `start_index`
     (`parent.start_index + local_offset`) for citations; inherit metadata.
  - Compute `token_estimate` on children (they are what gets embedded — keep within the
    256-token embedding limit; children at ~400 chars are comfortably inside).

## Models — app/models/chunk.py
- `ParentChunk`: `id`, `document_id`, `text`, `start_index`, `metadata`
- `ChildChunk`: `id`, `parent_id`, `document_id`, `chunk_index`, `text`, `char_count`,
  `token_estimate`, `start_index`, `metadata`
- Add `parent_count: int` and `child_count: int` to `IngestResponse`.

## Storage — app/services/storage.py
- `save_parents(document_id, parents, settings) -> Path`  # `{document_id}.parents.json`
- `load_parents(document_id, settings) -> dict[str, ParentChunk]`  # keyed by parent id
- `load_parent(parent_id, settings) -> ParentChunk`  # derive document_id from the id prefix
(Replaces the flat `{id}.chunks.json` persistence from Phase 2 — children now live in
Chroma, parents in the parent store.)

## Vector store — app/services/vector_store.py
- `add_children(children: list[ChildChunk]) -> int`: embed + store children with metadata
  `{parent_id, document_id, filename, page, start_index}`; `ids = child.id` (deterministic
  → re-ingest overwrites rather than duplicates).
- `search(query, top_k_parents=None, document_id=None) -> list[SearchResult]`:
  1. `similarity_search_with_score` over children with `k = retrieval_child_k` (apply the
     `document_id` filter when provided).
  2. Convert distance → similarity; group matched children by `parent_id`, keeping each
     parent's best child score.
  3. Load those parents; return up to `max_parents` (or `top_k_parents`) parents, sorted
     by best child score, as `SearchResult` (text = PARENT text, score = best child score,
     plus parent metadata).
- `delete_document(document_id)`: delete children by `document_id` filter AND remove the
  parents file.

## Generation — app/services/generation.py
- Build the context and `Source` list from the returned PARENT chunks (prompt and
  `[Source N]` labels unchanged). Snippet + citation fields come from the parent
  (filename, page, start_index).

## Ingestion — app/routers/ingestion.py
- Flow: extract → `chunk_document` (parents + children) → `save_parents` →
  `add_children` (embed). Response reports `parent_count` and `child_count`.
- Log both counts.

## Tests — tests/test_parent_retrieval.py (injected fake embeddings)
- Chunking yields parents and children; every `child.parent_id` resolves to an existing
  parent; a long parent yields multiple children (many children → one parent).
- Search collapses children to unique parents: a query matching a child returns that
  child's PARENT text as the result.
- Two matched children sharing a parent → the parent is returned once, scored by its best
  child.
- The `document_id` filter restricts results to that document.
- Generation context contains parent text, not raw child text.
- Re-ingesting a document does not duplicate children and overwrites the parents file.

## Acceptance criteria
- Ingesting a document creates persisted parents and embedded children; the response
  reports both counts.
- Queries return parent-sized context; answers are better grounded on questions whose
  answer spans more than one small chunk.
- Citations reference parents with correct filename / page / start_index.
- Re-ingest is idempotent; delete removes both the children and the parents file.
- All tests pass; `ruff check` / `ruff format` clean.
- **Migration:** the vector layout changed — clear `chroma_db/` (and any old
  `*.chunks.json`) and re-ingest documents once after this phase.

## Out of scope (later phases)
Contextual retrieval (Phase 7); abstention (Phase 8); dedup + observability (Phase 9);
evaluation + README (Phase 10). Hybrid search and reranking are reserved for Project 3.