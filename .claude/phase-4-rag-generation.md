# Phase 4 — Retrieval-Augmented Generation (Grounded Answers)

## Objective
Turn retrieved chunks into grounded, cited, streamed answers using a local Ollama model.
This is the phase where the system becomes a usable Q&A product:
query → retrieve → augment prompt → generate → stream answer + sources.

**Not in this phase:** confidence-based abstention (Phase 6 — grounding here is prompt-level
only); contextual / parent-document retrieval (Phase 5); conversation memory (later);
chat UI (dedicated next phase); hybrid search + reranking (Project 3).

## Dependencies to add this phase
Install, then add to `requirements.txt` (re-freeze after):
- `langchain-ollama` — `ChatOllama` client with streaming

Prerequisite: Ollama running locally with the generation model pulled:
`ollama pull qwen2.5:7b`

## Project structure changes
Add:
```
app/
├── prompts.py              # system prompt + context/answer templates
├── models/
│   └── query.py            # QueryRequest, Source, QueryResponse
├── services/
│   └── generation.py       # GenerationService (context build, prompt, stream/non-stream)
├── routers/
│   └── query.py            # POST /query (SSE stream + JSON)
```
Modify: `app/config.py`, `app/main.py` (build GenerationService in lifespan, register
router). Add `tests/test_query.py`.

## Configuration — app/config.py (additions)
- `ollama_base_url: str = "http://localhost:11434"`
- `generation_model: str = "qwen2.5:7b"`
- `generation_temperature: float = 0.2`     # low → faithful, factual, repeatable
- `generation_top_k: int = 5`               # chunks retrieved for context
- `generation_num_ctx: int = 8192`          # Ollama context window
- `request_timeout_s: float = 120.0`

## Prompts — app/prompts.py
- `SYSTEM_PROMPT`: instruct the model to answer using ONLY the provided context; if the
  context does not contain the answer, say it doesn't know rather than guessing; cite the
  sources it uses by their `[Source N]` labels; be concise and factual.
- `format_context(results: list[SearchResult]) -> str`: number each chunk as
  `"[Source N] (<filename>, page <page or n/a>):\n<text>"`, joined by blank lines.
- `build_user_prompt(context: str, question: str) -> str`.

## Models — app/models/query.py
- `QueryRequest`: `query: str`, `top_k: int | None = None`,
  `document_id: str | None = None`, `model: str | None = None`, `stream: bool = True`
- `Source`: `source_num: int`, `chunk_id: str`, `document_id: str`, `filename: str`,
  `page: int | None`, `start_index: int`, `score: float`, `snippet: str` (first ~200 chars)
- `QueryResponse`: `query: str`, `answer: str`, `sources: list[Source]`, `model: str`

## Generation service — app/services/generation.py
- `GenerationService` holds the vector store, embedding service, and generation config;
  constructs `ChatOllama(base_url=..., model=..., temperature=..., num_ctx=...)`
  (allow a per-request `model` override).
- `retrieve_and_build(query, top_k, document_id) -> tuple[str, list[Source]]`:
  call `VectorStoreService.search`, format the context, and build the `Source` list
  (source_num aligned with the `[Source N]` labels).
- `generate(request) -> QueryResponse` (non-streaming): build context, invoke ChatOllama,
  return answer + sources.
- `astream(request) -> AsyncIterator[str]` (SSE): yield, in order —
  1. a `sources` event: `{"type": "sources", "sources": [...]}`
  2. `token` events as ChatOllama streams: `{"type": "token", "text": "..."}`
  3. a final `{"type": "done"}` event.
  Each yielded as an SSE line `data: <json>\n\n`.
- If retrieval returns no chunks, proceed with empty context (the model will say it
  doesn't know). Robust score-threshold abstention is Phase 6.

## Router — app/routers/query.py
- `POST /query` (body: `QueryRequest`):
  - `stream=True` → `StreamingResponse(generation.astream(...),
    media_type="text/event-stream")`.
  - `stream=False` → `QueryResponse` JSON from `generation.generate(...)`.
  - Bind `request_id`; log query, model, retrieved count, and total latency.

## main.py
- Lifespan builds `GenerationService` (wraps the existing embedding + vector store
  services) and stores it on `app.state`; inject via a FastAPI dependency.
- Register the `query` router.

## Tests — tests/test_query.py
Inject a fake LLM (returns a fixed answer, no network) and fake vector-store results so
tests run offline and fast:
- Non-stream `/query` returns an answer plus a `sources` list mapped to the retrieved
  chunks, with `source_num` matching the `[Source N]` numbering.
- The `document_id` filter is passed through to search.
- The streaming generator yields a `sources` event first, then `token` events, then a
  `done` event, in that order.
- Empty retrieval → still returns a valid response, no crash.
- `top_k` and `model` overrides are respected.

## Acceptance criteria
- Ollama is running with `qwen2.5:7b`; `POST /query` (stream) returns a grounded answer
  streamed token-by-token, preceded by its sources.
- The answer draws only on the retrieved context; a question the documents don't cover
  produces an "I don't know"-style answer (prompt-level grounding).
- Each source lists filename, page (or n/a), score, and a snippet.
- Non-stream mode returns clean `QueryResponse` JSON.
- All tests pass with the mocked LLM; `ruff check` and `ruff format` are clean.

## Out of scope (later phases)
Confidence-based abstention (P6); contextual & parent-document retrieval (P5);
conversation memory; chat UI (dedicated next phase); dedup + tracing (P7);
evaluation + README (P8). Hybrid search and reranking are reserved for Project 3.