"""FastAPI dependency providers for the services built at startup."""

from fastapi import Request

from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService


def get_embedding_service(request: Request) -> EmbeddingService:
    """Return the process-wide embedding service built during startup.

    Reading it off ``app.state`` rather than constructing it keeps the model load out of
    the request path entirely.
    """
    return request.app.state.embedding_service


def get_vector_store(request: Request) -> VectorStoreService:
    """Return the process-wide vector store built during startup."""
    return request.app.state.vector_store
