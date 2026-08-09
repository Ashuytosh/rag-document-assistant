# RAG Document Assistant

A production-grade document Q&A system. Users upload documents and ask questions; the
system retrieves relevant passages and generates grounded, cited answers using local models.

## Tech stack
- Python 3.12, FastAPI, Uvicorn
- LangChain (orchestration), ChromaDB (vector store), sentence-transformers
  `all-MiniLM-L6-v2` (embeddings)
- Ollama for local LLM generation
- Jinja2 + Tailwind (CDN) + vanilla JS frontend (see the `frontend-design` skill)
- structlog for structured JSON logging
- Pydantic v2 + pydantic-settings for validation and config

## Architecture principles
- Two-phase RAG: indexing (extract → chunk → embed → store) and query
  (embed → retrieve → augment → generate).
- Grounded answers only: answer strictly from retrieved context; abstain when retrieval
  confidence is low rather than hallucinating.
- Every answer carries citations (source file + page/section).
- Dependencies are added per phase, not all upfront — `requirements.txt` grows as
  features land.

## Available Ollama models (8GB VRAM budget)
- `gemma3:4b` — fast, simple queries
- `phi4-mini` — logic and reasoning
- `qwen2.5:7b` — complex analysis
- `qwen2.5-coder:7b` — code and technical
- `mistral:7b` — conversational / creative

## Code rules
- Type hints everywhere (`str | None`, not `Optional`).
- Async I/O: use httpx in async mode; never block the event loop.
- Validate all external input with Pydantic; enforce upload size and type limits.
- Custom exceptions and graceful degradation; no bare `except`.
- Never read or commit `.env`; use `.env.example` for variable names.
- Keep functions small and documented; write tests alongside features.

## Workflow
- Specs-driven: each phase has a spec in `.claude/specs/`, implemented phase by phase.
- After writing code, invoke the `code-reviewer` and `test-engineer` subagents to verify it;
  invoke `security-reviewer` before merging upload/input/config changes.
- Use the `/daily-push` command to branch, commit, merge into main, delete the branch, and push.

## Project structure
Established in Phase 1 — application code under `app/`, tests under `tests/`,
uploaded files under `data/uploads/` (gitignored), vector store in `chroma_db/` (gitignored).