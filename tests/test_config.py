"""Settings behaviour."""

from __future__ import annotations

import pytest
from app.core.config import BGE_M3_EMBEDDING_DIM, Settings, get_settings
from pydantic import ValidationError


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "database_url": "postgresql+asyncpg://user:pass@127.0.0.1:5433/osm_geoagent",
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_defaults_match_the_documented_environment():
    settings = _settings()
    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "deepseek-r1:7b"
    assert settings.ollama_num_ctx == 4096
    assert settings.ollama_tool_mode == "prompted"
    assert settings.embedding_dim == BGE_M3_EMBEDDING_DIM == 1024
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.llm_endpoint_host == "127.0.0.1:11434"
    assert settings.avalai_base_url == "https://api.avalai.ir/v1"
    assert settings.avalai_model == "gemini-3.6-flash"
    assert settings.avalai_tool_mode == "native"
    assert settings.agent_max_tool_rounds >= 1
    assert settings.agent_max_tool_calls >= 1
    assert settings.agent_request_timeout_seconds > 0
    assert settings.ready_check_timeout_seconds > 0


def test_database_url_must_use_the_async_driver():
    with pytest.raises(ValidationError, match="asyncpg"):
        _settings(database_url="postgresql://user:pass@127.0.0.1:5433/osm_geoagent")


def test_settings_are_immutable():
    settings = _settings()
    with pytest.raises(ValidationError):
        settings.app_port = 9999  # type: ignore[misc]


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "some-other-model:latest")
    monkeypatch.setenv("RAG_TOP_K", "9")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@127.0.0.1:5433/osm_geoagent")
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.ollama_model == "some-other-model:latest"
        assert settings.rag_top_k == 9
        assert settings.database_url.startswith("postgresql+asyncpg://")
    finally:
        get_settings.cache_clear()


def test_avalai_provider_requires_an_api_key():
    with pytest.raises(ValidationError, match="AVALAI_API_KEY"):
        _settings(llm_provider="avalai")


def test_avalai_model_is_exact_when_configured():
    settings = _settings(llm_provider="avalai", avalai_api_key="placeholder-not-a-real-key")
    assert settings.llm_provider == "avalai"
    assert settings.llm_model == "gemini-3.6-flash"
    assert settings.avalai_model == "gemini-3.6-flash"
    assert settings.llm_endpoint_host == "api.avalai.ir"


def test_get_settings_is_cached():
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


def test_database_url_is_required_without_a_password_default():
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]
