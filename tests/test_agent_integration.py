"""Optional real-Ollama agent check.

Skipped unless ``RUN_INTEGRATION=1``. Requires a running Ollama with the
configured ``OLLAMA_MODEL``. Uses real tools only when the DB/embeddings are
available; otherwise skips.
"""

from __future__ import annotations

import os

import pytest
from app.agent.contracts import GeoAgentRequest
from app.agent.factory import build_geo_agent
from app.core.config import get_settings
from app.db.session import Database, DatabaseConfig
from app.embeddings.cache import is_model_cached
from app.embeddings.factory import build_bge_m3_provider
from app.llm.factory import build_ollama_provider
from app.osm.factory import build_query_osm_tool
from app.rag.retriever import SessionBoundKnowledgeRetriever
from app.tools.factory import build_tool_registry

pytestmark = pytest.mark.integration


def _integration_enabled() -> bool:
    return os.environ.get("RUN_INTEGRATION") == "1"


@pytest.mark.asyncio
async def test_live_ollama_documentation_question_does_not_require_overpass():
    if not _integration_enabled():
        pytest.skip("Set RUN_INTEGRATION=1 to run a real Ollama agent check")

    get_settings.cache_clear()
    settings = get_settings()
    if not is_model_cached(settings.bge_model_name):
        pytest.skip(f"{settings.bge_model_name} is not present in the local HF cache")

    database = Database(DatabaseConfig(url=settings.database_url))
    embedder = build_bge_m3_provider(settings)
    retriever = SessionBoundKnowledgeRetriever(database, embedder)
    query_osm_tool = build_query_osm_tool(settings)
    llm = build_ollama_provider(settings)
    registry = build_tool_registry(
        knowledge_retriever=retriever,
        query_osm_tool=query_osm_tool,
        rag_top_k=settings.rag_top_k,
    )
    agent = build_geo_agent(llm, registry, settings)
    try:
        result = await agent.run(
            GeoAgentRequest(
                message="What is the difference between leisure=park and landuse=grass?"
            )
        )
    finally:
        await llm.aclose()
        await query_osm_tool.aclose()
        await embedder.aclose()
        await database.dispose()
        get_settings.cache_clear()

    assert result.answer
    assert "<think>" not in result.answer
    assert all("<think>" not in event.message for event in result.trace)
    # Documentation question should not require live features.
    assert result.stop_reason in {"final_answer", "max_tool_rounds", "max_tool_calls", "llm_error"}
