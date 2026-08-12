# Phase 5 — Chat UI

## Objective
A server-rendered chat interface so the RAG system is demo-able in the browser: upload
documents, ask questions, and watch grounded answers stream in token-by-token with their
sources shown underneath. Reuses the existing `/ingest` and `/query` endpoints; adds no
new backend logic beyond serving the page.

**Not in this phase:** multi-turn conversation memory (later), authentication, chat
history / sessions, retrieval-quality upgrades (Phase 6).

## Dependencies to add this phase
None. `jinja2` is already installed (Phase 1); `StaticFiles` ships with FastAPI/Starlette.

## Project structure changes
Add:
```
app/
├── templates/
│   └── chat.html           # the chat page (Jinja2)
├── static/
│   └── app.js              # client logic: upload, SSE streaming, rendering
└── routers/
    └── ui.py               # GET / serves chat.html
```
Modify: `app/main.py` (mount StaticFiles, configure Jinja2Templates, register ui router),
`app/config.py` (expose app_name + available model list). Follow the `frontend-design`
skill for all styling and interaction. Add `tests/test_ui.py`.

## Configuration — app/config.py (additions)
- `available_models: list[str] = ["qwen2.5:7b", "gemma3:4b", "phi4-mini",
  "qwen2.5-coder:7b", "mistral:7b"]`   # populates the model dropdown; default is
  `generation_model`

## UI router — app/routers/ui.py
- `GET /` → `TemplateResponse("chat.html", {request, app_name, available_models,
  default_model: generation_model})`

## Template — app/templates/chat.html (per the frontend-design skill)
- Dark theme (zinc-950 background, zinc-800 borders, zinc-100 text, one accent color).
- Load Tailwind (CDN) and Lucide (CDN); no build step.
- Layout:
  - Header with the app name.
  - Upload row: file input (`accept=".pdf,.docx"`), an Upload button, and status text
    that shows the filename + chunk_count on success or a clear error on failure.
  - Controls: model dropdown (from `available_models`), an optional `top_k` input, and an
    optional "scope to last uploaded document" toggle.
  - Chat container: a scrollable list of user and assistant message bubbles.
  - Input row: a textarea + send button; Enter sends, Shift+Enter inserts a newline.
- Reference `/static/app.js`.

## Client logic — app/static/app.js
- **Upload:** POST the file to `/ingest` as multipart `FormData`; on success show the
  filename + chunk_count and store the returned document id (for optional scoping); on
  error show an inline message.
- **Ask:** POST to `/query` with `{query, model, top_k, stream: true, document_id?}`.
  - Consume the SSE stream from the POST response using `fetch` +
    `response.body.getReader()` + `TextDecoder` (NOT `EventSource`, which is GET-only).
    Buffer partial text and parse complete `data:` lines.
  - On the `sources` event: render source badges immediately under a new assistant
    message (filename, page, score; snippet expandable on click).
  - On each `token` event: append the text to the assistant message; show a blinking cursor.
  - On `done`: remove the cursor and finalize the message.
  - Before the first token arrives: show animated "thinking" dots.
  - Disable the send button while a request is in flight; re-enable on completion or error.
  - Handle network / Ollama-down errors gracefully with an inline error message.
- Keep all listeners in this one file (no inline `onclick`). **HTML-escape** all rendered
  text — document content and model output are untrusted — to prevent injection.

## main.py
- `app.mount("/static", StaticFiles(directory="app/static"), name="static")`
- `templates = Jinja2Templates(directory="app/templates")`
- Register the `ui` router.

## Tests — tests/test_ui.py
- `GET /` returns 200 and HTML containing the app name and the chat container element.
- The rendered page includes the model dropdown populated from `available_models`.
(Interactive streaming behavior is verified manually; keep automated tests to render
smoke checks.)

## Acceptance criteria
- `GET /` serves a dark, responsive, ChatGPT-like chat page (Tailwind + Lucide from CDN).
- Uploading a PDF/DOCX through the UI shows success and the chunk count.
- Asking a question streams the answer token-by-token with a blinking cursor and shows its
  sources underneath.
- A question the documents don't cover renders the "I don't know" response cleanly.
- All rendered document/model text is HTML-escaped (no injection).
- The page works at mobile width. `GET /` test passes; `ruff check`/`ruff format` clean.

## Out of scope (later phases)
Multi-turn conversation memory, authentication, chat history/sessions; contextual &
parent-document retrieval (Phase 6); abstention hardening (Phase 7); dedup + observability
(Phase 8); evaluation + README (Phase 9). Hybrid search and reranking are reserved for
Project 3.