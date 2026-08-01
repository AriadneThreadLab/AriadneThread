"""FastAPI application factory and lifespan.

Long-lived resources are created once at startup, attached to ``app.state`` and
released at shutdown. Nothing connects to PostgreSQL, Ollama or Hugging Face at
import time.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

from app import __version__
from app.api.health import router as health_router
from app.core.config import Settings, get_settings
from app.core.errors import GeoAgentError
from app.core.logging import configure_logging
from app.db.session import Database, DatabaseConfig
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def build_tool_registry() -> ToolRegistry:
    """Create the registry of tools the agent may call.

    Empty until the retrieval and Overpass implementations land; the agent can
    only ever call what is registered here.
    """
    return ToolRegistry()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Accepts settings so tests can override them."""
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved
        app.state.database = Database(DatabaseConfig(url=resolved.database_url))
        app.state.tool_registry = build_tool_registry()
        logger.info(
            "OSM GeoAgent starting (env=%s, model=%s)",
            resolved.app_env,
            resolved.ollama_model,
        )
        try:
            yield
        finally:
            await app.state.database.dispose()
            logger.info("OSM GeoAgent stopped")

    app = FastAPI(
        title="OSM GeoAgent",
        version=__version__,
        summary="Natural-language GIS requests turned into controlled OpenStreetMap operations.",
        lifespan=lifespan,
    )
    app.include_router(health_router)

    @app.exception_handler(GeoAgentError)
    async def handle_geoagent_error(_: Request, exc: GeoAgentError) -> JSONResponse:
        logger.warning("request failed: %s (%s)", exc.message, exc.code)
        return JSONResponse(status_code=400, content={"code": exc.code, "detail": exc.message})

    return app


app = create_app()
