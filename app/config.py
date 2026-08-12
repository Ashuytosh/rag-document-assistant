"""Application configuration, loaded from the environment or a local `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Upper bound on how many results a single search may request, so one call cannot ask
#: the store for an unbounded number of vectors.
MAX_TOP_K = 50
#: Tighter bound for generation, and the ceiling on a request's `top_k`. It counts
#: PARENTS now, not the 800-character chunks it was originally sized for: each is ~2000
#: characters of prompt text, so 6 already approaches half of an 8192-token window once
#: the system prompt and question are added. Too many pushes the grounding rules out of
#: the window entirely — which is exactly what a caller could do while this said 12.
MAX_GENERATION_TOP_K = 6

PDF_CONTENT_TYPE = "application/pdf"
DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

#: Extension to use on disk for each accepted content type. Uploads are stored under a
#: generated uuid plus one of these extensions, so the client-supplied filename never
#: reaches the filesystem and cannot be used for path traversal.
CONTENT_TYPE_EXTENSIONS: dict[str, str] = {
    PDF_CONTENT_TYPE: ".pdf",
    DOCX_CONTENT_TYPE: ".docx",
}


class Settings(BaseSettings):
    """Runtime settings. Every field has a usable default for local development."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG Document Assistant"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    upload_dir: Path = Path("data/uploads")
    max_upload_bytes: int = 20 * 1024 * 1024
    allowed_content_types: set[str] = set(CONTENT_TYPE_EXTENSIONS)

    #: A DOCX is a zip container, and python-docx reads members fully into memory. These
    #: bound decompression so a small upload cannot expand into an out-of-memory kill.
    max_uncompressed_bytes: int = 100 * 1024 * 1024
    max_compression_ratio: int = 100

    #: Parent-document retrieval splits twice. Parents are the unit an answer is written
    #: from: large enough to hold evidence that spans several sentences, and never
    #: embedded, so the embedding model's 256-token window does not constrain them.
    parent_chunk_size: int = Field(default=2000, gt=0)
    parent_chunk_overlap: int = Field(default=200, ge=0)
    #: Children are the unit that is matched: small enough that a vector describes one
    #: idea sharply, and comfortably inside the ~256-token window (400 chars ≈ 100 tokens).
    child_chunk_size: int = Field(default=400, gt=0)
    child_chunk_overlap: int = Field(default=80, ge=0)
    #: Split priority: paragraph, then line, then sentence, then word, then hard cut.
    #: Shared by both splitters — the boundaries worth respecting do not change with size.
    chunk_separators: list[str] = ["\n\n", "\n", ". ", " ", ""]
    #: Ceiling on a parents file before a read refuses it, so serving the parents of a
    #: huge document cannot be used to amplify memory use on repeated requests.
    max_chunks_file_bytes: int = 32 * 1024 * 1024

    #: Embeddings run on the CPU by default: the 8 GB VRAM budget is reserved for Ollama.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    #: Pin the exact model snapshot. Without it, a moved upstream `main` would be
    #: downloaded and loaded silently on the next restart.
    embedding_model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    #: Load from the local cache only. Startup otherwise makes ~20 calls to
    #: huggingface.co to revalidate an already-complete cache, which both slows boot and
    #: makes the service unable to start when that host is unreachable. Set false for
    #: the first run, which populates the cache.
    embedding_local_files_only: bool = False
    embedding_device: str = "cpu"
    #: Normalized vectors make cosine distance equivalent to the dot product, which is
    #: what lets `score = 1 - distance` be a meaningful similarity.
    embedding_normalize: bool = True
    embedding_max_tokens: int = Field(default=256, gt=0)

    chroma_persist_dir: Path = Path("chroma_db")
    #: Chroma requires 3-512 chars from [a-zA-Z0-9._-], starting and ending alphanumeric.
    chroma_collection: str = Field(
        default="documents",
        min_length=3,
        max_length=512,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$",
    )
    #: The HNSW space. `score = 1 - distance` only holds for cosine, so this is a Literal
    #: rather than a free string — a typo would silently produce meaningless scores.
    chroma_distance: Literal["cosine"] = "cosine"

    search_top_k: int = Field(default=5, gt=0, le=MAX_TOP_K)

    #: Generation runs on Ollama locally. Temperature is deliberately low: the answer
    #: must be faithful to the retrieved context, not creative.
    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    generation_model: str = "qwen2.5:7b"
    #: Closed set, not a pattern: `model` is client-supplied, and any installed name
    #: would otherwise let a caller force repeated multi-GB evict/reload cycles on the
    #: 8 GB VRAM budget, and enumerate what is installed from the response codes.
    allowed_generation_models: set[str] = {
        "qwen2.5:7b",
        "qwen2.5-coder:7b",
        "phi4-mini",
        "gemma3:4b",
        "mistral:7b",
    }
    #: Names offered in the UI model dropdown, in the order they appear. Ordering is why
    #: this is a list and not derived from the allowlist set above; the validator below
    #: keeps the two from drifting apart.
    available_models: list[str] = Field(
        min_length=1,
        default=[
            "qwen2.5:7b",
            "gemma3:4b",
            "phi4-mini",
            "qwen2.5-coder:7b",
            "mistral:7b",
        ],
    )
    generation_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    #: Children fetched from the collection before they are collapsed to parents. Larger
    #: than max_parents on purpose: several matches usually share a parent, so the search
    #: must over-retrieve to have max_parents *distinct* parents to choose from.
    retrieval_child_k: int = Field(default=15, gt=0, le=MAX_TOP_K)
    #: Parents passed to the LLM after deduplication. Bounded well below MAX_TOP_K: each
    #: parent is ~2000 characters of prompt text, and Ollama truncates from the front
    #: once num_ctx is exceeded — so an unbounded count pushes the grounding rules out of
    #: the window entirely.
    max_parents: int = Field(default=4, gt=0, le=MAX_GENERATION_TOP_K)
    generation_num_ctx: int = Field(default=8192, gt=0)
    #: Hard cap on generated tokens. Without it, output length is bounded only by the
    #: context window and one request can occupy the GPU for minutes.
    generation_num_predict: int = Field(default=1024, gt=0)
    #: Concurrent generations. One GPU with one resident model: queueing beyond this
    #: adds latency and memory pressure without adding throughput.
    generation_concurrency: int = Field(default=2, gt=0)
    #: Total wall-clock budget for one generation.
    request_timeout_s: float = Field(default=120.0, gt=0)

    @model_validator(mode="after")
    def _check_generation_model_allowed(self) -> "Settings":
        """The configured default must itself be in the allowlist."""
        if self.generation_model not in self.allowed_generation_models:
            raise ValueError(
                f"generation_model ({self.generation_model}) must be one of "
                f"{sorted(self.allowed_generation_models)}."
            )
        return self

    @model_validator(mode="after")
    def _check_available_models_allowed(self) -> "Settings":
        """Every model offered in the UI must be one ``/query`` will actually accept.

        Without this, a name added to the dropdown but not to the allowlist would render
        fine and then fail with a 422 only once a user selected it.
        """
        unknown = [
            model for model in self.available_models if model not in self.allowed_generation_models
        ]
        if unknown:
            raise ValueError(
                f"available_models {unknown} must all be in "
                f"{sorted(self.allowed_generation_models)}."
            )
        if self.generation_model not in self.available_models:
            # The dropdown preselects generation_model; if it isn't offered, the page
            # would silently start on a different model than the server's default.
            raise ValueError(
                f"available_models must include generation_model ({self.generation_model})."
            )
        return self

    @model_validator(mode="after")
    def _check_chunk_bounds(self) -> "Settings":
        """Fail at startup rather than per-request when the chunk window is nonsensical.

        The window must advance by at least half a chunk each step. Merely requiring
        ``overlap < size`` is not enough: at 99/100 the splitter makes so little forward
        progress that it emits many identical chunks and silently drops the rest of the
        document — text that would then never be indexed. Both levels are checked: the
        parent splitter losing text loses it for retrieval entirely.
        """
        for size_name, overlap_name in (
            ("parent_chunk_size", "parent_chunk_overlap"),
            ("child_chunk_size", "child_chunk_overlap"),
        ):
            size = getattr(self, size_name)
            overlap = getattr(self, overlap_name)
            max_overlap = size // 2
            if overlap > max_overlap:
                raise ValueError(
                    f"{overlap_name} ({overlap}) must be at most half of "
                    f"{size_name} ({size}), i.e. <= {max_overlap}."
                )
        if self.child_chunk_size >= self.parent_chunk_size:
            # Equal sizes make every parent yield exactly one identical child, which is
            # Phase 2's flat chunking wearing a new name — no sharper matching, no wider
            # context, and nothing in the behaviour to reveal the misconfiguration.
            raise ValueError(
                f"child_chunk_size ({self.child_chunk_size}) must be smaller than "
                f"parent_chunk_size ({self.parent_chunk_size})."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, cached for use as a FastAPI dependency."""
    return Settings()
