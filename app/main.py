"""FastAPI application factory and lifespan.

Long-lived resources are created once at startup, attached to ``app.state`` and
released at shutdown. Nothing connects to PostgreSQL, Ollama or Hugging Face at
import time. The BGE-M3 model is constructed lazily and only loaded on first use.

Phase 5 wires the planner-executor agent into application state. The production
``/agent/query`` HTTP endpoint is intentionally deferred to Phase 6.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

from app import __version__
from app.agent.factory import build_geo_agent
from app.api.health import router as health_router
from app.core.config import Settings, get_settings
from app.core.errors import GeoAgentError
from app.core.logging import configure_logging
from app.db.session import Database, DatabaseConfig
from app.embeddings.factory import build_bge_m3_provider
from app.llm.factory import build_ollama_provider
from app.osm.factory import build_query_osm_tool
from app.rag.retriever import SessionBoundKnowledgeRetriever
from app.tools.factory import build_tool_registry

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Accepts settings so tests can override them."""
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = Database(DatabaseConfig(url=resolved.database_url))
        embedder = build_bge_m3_provider(resolved)
        retriever = SessionBoundKnowledgeRetriever(database, embedder)
        query_osm_tool = build_query_osm_tool(resolved)
        llm = build_ollama_provider(resolved)
        registry = build_tool_registry(
            knowledge_retriever=retriever,
            query_osm_tool=query_osm_tool,
            rag_top_k=resolved.rag_top_k,
        )
        agent = build_geo_agent(llm, registry, resolved)
        app.state.settings = resolved
        app.state.database = database
        app.state.embedding_provider = embedder
        app.state.query_osm_tool = query_osm_tool
        app.state.llm_provider = llm
        app.state.tool_registry = registry
        app.state.geo_agent = agent
        logger.info(
            "OSM GeoAgent starting (env=%s, model=%s, tools=%s)",
            resolved.app_env,
            resolved.ollama_model,
            ",".join(registry.names) or "none",
        )
        try:
            yield
        finally:
            await llm.aclose()
            await query_osm_tool.aclose()
            await embedder.aclose()
            await database.dispose()
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
