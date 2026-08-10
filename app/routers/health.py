"""Liveness endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, str]:
    """Report that the service is up, along with its name and version."""
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}
