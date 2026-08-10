"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.routers import health, ingestion


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and ensure the upload directory exists before serving."""
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    get_logger(__name__).info(
        "app.startup",
        app=settings.app_name,
        version=settings.app_version,
        upload_dir=str(settings.upload_dir),
    )
    yield


def create_app() -> FastAPI:
    """Build the FastAPI application with its routers and error handlers registered."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(ingestion.router)
    return app


app = create_app()
