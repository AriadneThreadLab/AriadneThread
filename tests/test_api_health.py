"""The health endpoint must work without PostgreSQL, Ollama or Overpass."""

from __future__ import annotations

import httpx
import pytest
from app.core.config import Settings
from app.main import create_app


@pytest.fixture
async def client(settings: Settings):
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        yield http


async def test_health_reports_configuration(client: httpx.AsyncClient):
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_env"] == "test"
    assert body["llm_model"] == "deepseek-r1:7b"
    assert body["llm_provider"] == "ollama"
    assert body["embedding_model"] == "BAAI/bge-m3"
    assert body["embedding_dim"] == 1024


async def test_health_reports_the_avalai_model_when_selected():
    settings = Settings(
        app_env="test",
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:5433/osm_geoagent_test",
        llm_provider="avalai",
        avalai_api_key="placeholder-not-a-real-key",
        avalai_model="gemini-3.6-flash",
        _env_file=None,  # type: ignore[call-arg]
    )
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        response = await http.get("/health")
    assert response.status_code == 200
    assert response.json()["llm_model"] == "gemini-3.6-flash"
    assert response.json()["llm_provider"] == "avalai"


async def test_openapi_documents_the_health_endpoint(client: httpx.AsyncClient):
    schema = (await client.get("/openapi.json")).json()
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]
    assert "/api/v1/agent/query" in schema["paths"]
    assert "/" in schema["paths"]
    assert schema["info"]["title"] == "Ariadne Thread"
    assert "post" in schema["paths"]["/api/v1/agent/query"]
    components = schema.get("components", {}).get("schemas", {})
    assert "AgentQueryRequest" in components
    assert "AgentQueryResponse" in components


async def test_application_state_exposes_shared_resources(settings: Settings):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        assert app.state.settings is settings
        # Registry.names is sorted alphabetically.
        assert app.state.tool_registry.names == (
            "analyze_features",
            "query_osm",
            "resolve_place",
            "search_osm_knowledge",
        )
        assert app.state.query_osm_tool is not None
        assert app.state.llm_provider is not None
        assert app.state.geo_agent is not None
        assert app.state.services is not None
        assert app.state.database is not None
        assert app.state.embedding_provider is not None
        assert not app.state.embedding_provider.is_loaded
