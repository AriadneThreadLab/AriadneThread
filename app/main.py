"""FastAPI application factory and lifespan.

Long-lived resources are created once at startup via the composition root
(:mod:`app.bootstrap`), attached to ``app.state``, and released at shutdown.
Nothing connects to PostgreSQL, Ollama or Hugging Face at import time. BGE-M3
is constructed lazily and only loaded on first embedding use.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.agent import router as agent_router
from app.api.errors import register_exception_handlers
from app.api.health import geoloadst_health
from app.api.health import router as health_router
from app.api.middleware import request_context_middleware
from app.bootstrap import ApplicationServices, build_application_services
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.web.routes import router as ui_router

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"


def create_app(
    settings: Settings | None = None,
    *,
    services: ApplicationServices | None = None,
) -> FastAPI:
    """Build the application.

    ``services`` may be supplied by tests to inject fakes without touching
    PostgreSQL, Ollama or Overpass.
    """
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = services or build_application_services(resolved)
        app.state.settings = owned.settings
        app.state.services = owned
        app.state.database = owned.database
        app.state.embedding_provider = owned.embedding_provider
        app.state.query_osm_tool = owned.query_osm_tool
        app.state.llm_provider = owned.llm_provider
        app.state.tool_registry = owned.tool_registry
        app.state.geo_agent = owned.geo_agent
        # Selection only. No training pipeline is imported or started here.
        app.state.active_learning = owned.active_learning
        energy = geoloadst_health()
        logger.info(
            "OSM GeoAgent starting (env=%s, provider=%s, model=%s, llm_host=%s, tools=%s)",
            owned.settings.app_env,
            owned.settings.llm_provider,
            owned.settings.llm_model,
            owned.settings.llm_endpoint_host,
            ",".join(owned.tool_registry.names) or "none",
        )
        logger.info(
            "geoloadst_health available=%s version=%s capabilities=%s",
            energy.geoloadst_available,
            energy.version or "none",
            ",".join(energy.capabilities) or "none",
        )
        try:
            yield
        finally:
            # Only dispose resources this lifespan created.
            if services is None:
                await owned.aclose()
            logger.info("OSM GeoAgent stopped")

    app = FastAPI(
        title="Ariadne Thread",
        version=__version__,
        summary="Knowledge-grounded OpenStreetMap search and analysis.",
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(agent_router)
    app.include_router(ui_router)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    app.middleware("http")(request_context_middleware)
    return app


app = create_app()
