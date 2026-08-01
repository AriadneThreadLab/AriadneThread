"""Optional real-model retrieval checks.

These tests require:

* the ``embeddings`` extra installed
* BAAI/bge-m3 present in the local Hugging Face cache
* a migrated ``osm_geoagent`` database with indexed chunks

They are skipped unless ``RUN_INTEGRATION=1`` is set in the environment.
"""

from __future__ import annotations

import os

import pytest
from app.core.config import get_settings
from app.db.session import Database, DatabaseConfig
from app.embeddings.cache import is_model_cached
from app.embeddings.factory import build_bge_m3_provider
from app.rag.retriever import SessionBoundKnowledgeRetriever

pytestmark = pytest.mark.integration


def _integration_enabled() -> bool:
    return os.environ.get("RUN_INTEGRATION") == "1"


@pytest.mark.asyncio
async def test_persian_query_ranks_leisure_park_documentation():
    if not _integration_enabled():
        pytest.skip("Set RUN_INTEGRATION=1 to run real BGE-M3 + database checks")
    if not is_model_cached("BAAI/bge-m3"):
        pytest.skip("BAAI/bge-m3 is not present in the local Hugging Face cache")

    get_settings.cache_clear()
    settings = get_settings()
    database = Database(DatabaseConfig(url=settings.database_url))
    embedder = build_bge_m3_provider(settings)
    retriever = SessionBoundKnowledgeRetriever(database, embedder)
    try:
        passages = await retriever.search(
            "پارک عمومی در OSM با چه تگی مشخص می‌شود؟",
            top_k=5,
        )
    finally:
        await embedder.aclose()
        await database.dispose()
        get_settings.cache_clear()

    assert passages, "expected indexed OSM documentation passages"
    joined = " ".join(
        f"{passage.document_title} {passage.section or ''} {passage.content}"
        for passage in passages[:3]
    ).lower()
    assert "leisure" in joined and "park" in joined
