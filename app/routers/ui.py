"""Server-rendered chat page.

The only route here renders a template; every piece of behaviour it drives — ingestion,
retrieval, generation — already exists as an API endpoint the page calls from the browser.
"""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import MAX_GENERATION_TOP_K, Settings, get_settings

#: Resolved from this module rather than the working directory: uvicorn is not always
#: started from the repository root, and a relative "app/templates" silently 500s when it
#: isn't.
PACKAGE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
#: Exported so the application factory can mount it without re-deriving the path.
STATIC_DIR = PACKAGE_DIR / "static"


def _format_megabytes(size_bytes: int) -> str:
    """Render a byte count as megabytes, dropping a trailing ``.0``."""
    megabytes = round(size_bytes / (1024 * 1024), 1)
    return str(int(megabytes)) if megabytes == int(megabytes) else str(megabytes)


router = APIRouter(tags=["ui"], include_in_schema=False)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse)
async def chat_page(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> HTMLResponse:
    """Serve the chat interface, configured from the running settings."""
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "available_models": settings.available_models,
            "default_model": settings.generation_model,
            "default_top_k": settings.max_parents,
            "max_top_k": MAX_GENERATION_TOP_K,
            # Rounded, not floored: a limit under 1 MiB — which the test settings use —
            # would otherwise advertise itself to the user as "up to 0 MB". Whole values
            # render without the decimal, so the common case still reads "20 MB".
            "max_upload_mb": _format_megabytes(settings.max_upload_bytes),
        },
    )
