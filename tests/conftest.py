"""Shared fixtures.

No test in this suite may contact PostgreSQL, Ollama, Hugging Face or Overpass.
"""

from __future__ import annotations

import pytest
from app.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Settings with explicit values, independent of any local .env file."""
    return Settings(
        app_env="test",
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:5433/osm_geoagent_test",
        ollama_model="deepseek-r1:7b",
        _env_file=None,  # type: ignore[call-arg]
    )
