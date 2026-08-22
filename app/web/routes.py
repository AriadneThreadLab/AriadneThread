"""UI routes.

Serving the demo page does not touch PostgreSQL, Ollama, Overpass or BGE-M3.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_INDEX_HTML = _TEMPLATES_DIR / "index.html"

#: Restrictive CSP compatible with same-origin assets + MapLibre/OpenFreeMap CDN.
_CSP = (
    "default-src 'self'; "
    "script-src 'self' https://unpkg.com; "
    "style-src 'self' 'unsafe-inline' https://unpkg.com https://fonts.bunny.net; "
    "font-src 'self' https://fonts.bunny.net; "
    "img-src 'self' data: blob: https://*.openfreemap.org https://*.tile.openstreetmap.org; "
    "connect-src 'self' https://*.openfreemap.org; "
    "worker-src 'self' blob:; "
    "child-src blob:; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


@router.get("/", response_class=HTMLResponse, summary="Ariadne Thread web UI")
async def ui_index() -> HTMLResponse:
    """Return the English demo UI. No agent or database work happens here."""
    html = _INDEX_HTML.read_text(encoding="utf-8")
    return HTMLResponse(
        content=html,
        headers={
            "Content-Security-Policy": _CSP,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        },
    )
