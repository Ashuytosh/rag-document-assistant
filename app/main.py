"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.routers import health, ingestion, query, search
from app.services.embedding import EmbeddingService
from app.services.generation import GenerationService
from app.services.vector_store import VectorStoreService


def build_embedding_service(settings: Settings) -> EmbeddingService:
    """Construct the embedding service, loading the model.

    A module-level factory rather than an inline call so tests can substitute a fake and
    keep the model — and torch — out of the suite entirely.
    """
    return EmbeddingService(settings)


def build_vector_store(settings: Settings, embeddings: EmbeddingService) -> VectorStoreService:
    """Construct the vector store over the embedding service's model."""
    return VectorStoreService(settings, embeddings.as_langchain())


def build_generation_service(
    settings: Settings, vector_store: VectorStoreService
) -> GenerationService:
    """Construct the generation service over the retrieval stack.

    A module-level factory for the same reason as the others: tests substitute a fake
    chat model here so nothing in the suite reaches Ollama.
    """
    return GenerationService(settings, vector_store)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare logging, storage, and the retrieval services before serving.

    The embedding model is loaded exactly once here; request handlers reach it through
    ``app.state`` so nothing reloads it per request.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    log = get_logger(__name__)
    log.info(
        "app.startup",
        app=settings.app_name,
        version=settings.app_version,
        upload_dir=str(settings.upload_dir),
    )

    embedding_service = build_embedding_service(settings)
    app.state.embedding_service = embedding_service
    vector_store = build_vector_store(settings, embedding_service)
    app.state.vector_store = vector_store
    app.state.generation_service = build_generation_service(settings, vector_store)
    log.info(
        "app.ready",
        collection=settings.chroma_collection,
        vectors=vector_store.count(),
        generation_model=settings.generation_model,
    )
    yield


def create_app() -> FastAPI:
    """Build the FastAPI application with its routers and error handlers registered."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(ingestion.router)
    app.include_router(search.router)
    app.include_router(query.router)
    return app


app = create_app()
