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
    assert body["embedding_model"] == "BAAI/bge-m3"
    assert body["embedding_dim"] == 1024


async def test_openapi_documents_the_health_endpoint(client: httpx.AsyncClient):
    schema = (await client.get("/openapi.json")).json()
    assert "/health" in schema["paths"]
    assert schema["info"]["title"] == "OSM GeoAgent"


async def test_application_state_exposes_shared_resources(settings: Settings):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        assert app.state.settings is settings
        assert app.state.tool_registry.names == ()
        assert app.state.database is not None
