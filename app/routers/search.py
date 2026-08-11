"""Similarity search over stored chunks, and collection statistics."""

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.dependencies import get_vector_store
from app.models.search import SearchRequest, SearchResponse, StatsResponse
from app.services.vector_store import VectorStoreService

router = APIRouter(tags=["search"])
log = get_logger(__name__)


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[VectorStoreService, Depends(get_vector_store)],
) -> SearchResponse:
    """Return the chunks most similar to ``query``, best match first."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=str(uuid.uuid4()))
    top_k = request.top_k or settings.search_top_k

    document_id = str(request.document_id) if request.document_id else None
    # Embedding and the HNSW lookup are blocking CPU work — keep them off the loop.
    results = await run_in_threadpool(store.search, request.query, top_k, document_id)

    log.info(
        "search.complete",
        top_k=top_k,
        document_id=document_id,
        result_count=len(results),
        top_score=round(results[0].score, 4) if results else None,
    )
    return SearchResponse(query=request.query, count=len(results), results=results)


@router.get("/stats", response_model=StatsResponse)
async def stats(store: Annotated[VectorStoreService, Depends(get_vector_store)]) -> StatsResponse:
    """Report how many vectors the collection currently holds."""
    total = await run_in_threadpool(store.count)
    return StatsResponse(total_vectors=total, collection=store.collection_name)
