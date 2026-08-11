# Phase 3 — Embeddings & Vector Store

## Objective
Embed each chunk with `all-MiniLM-L6-v2`, store the vectors and metadata in a persistent
ChromaDB collection, and expose a similarity-search endpoint that returns the top-k chunks
for a query with scores. This completes the indexing pipeline and the retrieval mechanics —
without any LLM generation yet.

**Not in this phase:** LLM generation, prompt augmentation, streaming, citations (Phase 4);
contextual / parent-document retrieval (Phase 5); grounding/abstention (Phase 6);
hybrid search + reranking (reserved for Project 3).

## Dependencies to add this phase
Install, then add to `requirements.txt` (re-freeze after). This is the large install
(pulls in torch); the embedding model (~90 MB) downloads to cache on first use.
- `sentence-transformers` — the embedding model runtime
- `chromadb` — vector database
- `langchain-huggingface` — `HuggingFaceEmbeddings` wrapper
- `langchain-chroma` — `Chroma` vector store integration

## Project structure changes
Add:
```
app/
├── models/
│   └── search.py           # SearchRequest, SearchResult, SearchResponse, StatsResponse
├── services/
│   ├── embedding.py        # EmbeddingService (load model once, embed, token_count)
│   └── vector_store.py     # VectorStoreService (Chroma wrapper)
├── routers/
│   └── search.py           # POST /search, GET /stats
```
Modify: `app/config.py`, `app/main.py` (lifespan loads the model + store into app state),
`app/routers/ingestion.py` (embed + store after chunking). Add
`tests/test_search.py`.

## Configuration — app/config.py (additions)
- `embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"`
- `embedding_device: str = "cpu"`          # keep embeddings off the GPU (VRAM for Ollama)
- `embedding_normalize: bool = True`        # normalized vectors → cosine distance
- `chroma_persist_dir: Path = Path("chroma_db")`
- `chroma_collection: str = "documents"`
- `chroma_distance: str = "cosine"`         # HNSW space; must match normalized embeddings
- `search_top_k: int = 5`
- `embedding_max_tokens: int = 256`         # model's max sequence length

## Embedding service — app/services/embedding.py
- `EmbeddingService`:
  - On init, build `HuggingFaceEmbeddings(model_name=..., model_kwargs={"device": ...},
    encode_kwargs={"normalize_embeddings": ...})`. Load once; reuse.
  - Expose the underlying embeddings object for the vector store (`as_langchain()`).
  - `embed_query(text: str) -> list[float]` and `embed_texts(texts) -> list[list[float]]`
    convenience wrappers.
  - `count_tokens(text: str) -> int` using the model's tokenizer, and
    `assert_within_limit(text)` that logs a warning if a chunk exceeds
    `embedding_max_tokens`.

## Vector store service — app/services/vector_store.py
- `VectorStoreService`:
  - Init with a persistent `Chroma(collection_name=..., persist_directory=...,
    embedding_function=<from EmbeddingService>, collection_metadata={"hnsw:space":
    settings.chroma_distance})`.
  - `add_chunks(chunks: list[Chunk]) -> int`: add texts with `ids=[chunk.id ...]`,
    `metadatas=[{document_id, chunk_index, filename, content_type, page_count,
    start_index} ...]`. Using deterministic `chunk.id` makes re-adding a document
    overwrite rather than duplicate.
  - `search(query: str, top_k: int, document_id: str | None = None)
    -> list[SearchResult]`: run `similarity_search_with_score`; if `document_id` is
    given, pass a `where={"document_id": document_id}` filter. Convert Chroma distance
    to a similarity score: `score = 1 - distance` (cosine). Return chunk text +
    metadata + score, sorted by score descending.
  - `delete_document(document_id: str) -> int`: delete all vectors where
    `document_id` matches.
  - `count() -> int`: number of vectors in the collection.
- Design note: accept the embedding function via injection so tests can pass a fast,
  deterministic fake instead of the real model.

## Models — app/models/search.py
- `SearchRequest`: `query: str`, `top_k: int | None = None`,
  `document_id: str | None = None`
- `SearchResult`: `chunk_id: str`, `document_id: str`, `chunk_index: int`,
  `text: str`, `score: float`, `metadata: dict`
- `SearchResponse`: `query: str`, `count: int`, `results: list[SearchResult]`
- `StatsResponse`: `total_vectors: int`, `collection: str`

## Routers — app/routers/search.py
- `POST /search` (body: `SearchRequest`):
  - Use `settings.search_top_k` when `top_k` is not provided.
  - Call `VectorStoreService.search(...)`; return `SearchResponse`.
  - Bind a `request_id`; log query, result count, and top score.
- `GET /stats` → `StatsResponse` with the collection's vector count.

## Ingestion changes — app/routers/ingestion.py
- After chunking + `save_chunks`, call `embedding_service.assert_within_limit` on each
  chunk (log warnings), then `vector_store.add_chunks(chunks)`.
- Keep `chunk_count` in the response; log the number of vectors added.

## main.py
- In the lifespan startup: construct `EmbeddingService` (loads the model once) and
  `VectorStoreService`, and store both on `app.state` (or a simple dependency provider).
- Routers and the ingestion flow get these via FastAPI dependencies — no per-request
  model loading.
- Register the new `search` router.

## Tests — tests/test_search.py
pytest, using an injected fake embedding function (deterministic, tiny vectors) so tests
run offline and fast:
- Adding chunks increases `count()` by the number added.
- Re-adding the same document (same ids) does NOT increase the count (overwrite, not
  duplicate).
- `search` returns results sorted by score descending; a query matching a specific chunk
  ranks that chunk first.
- The `document_id` filter restricts results to that document.
- `POST /search` returns a valid `SearchResponse`; `top_k` is respected.
- `GET /stats` reports the correct vector count.
- One `@pytest.mark.integration` test may use the real model to sanity-check a semantic
  query (skippable so the default suite stays fast).
- A token-count check confirms `count_tokens` works and flags an over-limit string.

## Acceptance criteria
- The new dependencies install and `requirements.txt` re-freezes cleanly.
- Server startup loads the embedding model once (visible in logs); no per-request loads.
- Ingesting a document embeds and stores its chunks; `GET /stats` reflects the vector count.
- `POST /search` with a natural-language query returns relevant chunks with sensible
  scores, top result first. Filtering by `document_id` works.
- Re-ingesting a document does not duplicate its vectors.
- All non-integration tests pass; `ruff check` and `ruff format` are clean.

## Out of scope (later phases)
LLM generation + augmentation + streaming + citations (P4); contextual & parent-document
retrieval (P5); grounding/abstention (P6); dedup + tracing (P7); evaluation + README (P8);
frontend UI. Hybrid search and reranking are reserved for Project 3.