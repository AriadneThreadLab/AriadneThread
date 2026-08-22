"""Shared fixtures.

No test in this suite may contact PostgreSQL, Ollama, Hugging Face or Overpass.
"""

from __future__ import annotations

import pytest
from app.analytics.datasets import DatasetRegistry
from app.core.config import Settings
from app.places.contracts import PlaceRegistry
from app.tools.context import AnalysisRunState, GroundingState, ToolContext


@pytest.fixture
def settings() -> Settings:
    """Settings with explicit values, independent of any local .env file."""
    return Settings(
        app_env="test",
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:5433/osm_geoagent_test",
        ollama_model="deepseek-r1:7b",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def tool_context() -> ToolContext:
    """Empty request-scoped tool context for offline tool tests."""
    return ToolContext(
        datasets=DatasetRegistry(),
        analysis=AnalysisRunState(),
        user_message="test message 35.7 51.4",
        places=PlaceRegistry(),
        grounding=GroundingState(),
    )
