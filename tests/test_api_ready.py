"""Readiness and liveness behaviour."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from app.agent.contracts import GeoAgentResponse, TraceEvent
from app.api.dependencies import get_database
from app.bootstrap import ApplicationServices
from app.core.config import Settings
from app.main import create_app


class _IdleAgent:
    async def run(self, request):  # type: ignore[no-untyped-def]
        return GeoAgentResponse(
            answer="unused",
            trace=[TraceEvent(kind="final_answer", message="unused")],
            stop_reason="final_answer",
            model="fake",
        )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_env="test",
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:5433/osm_geoagent_test",
        ollama_model="deepseek-r1:7b",
        ready_check_timeout_seconds=0.2,
        _env_file=None,  # type: ignore[call-arg]
    )


def _services(settings: Settings) -> ApplicationServices:
    registry = MagicMock()
    registry.names = ()
    return ApplicationServices(
        settings=settings,
        database=MagicMock(),
        embedding_provider=MagicMock(is_loaded=False),
        query_osm_tool=MagicMock(),
        resolve_place_tool=MagicMock(),
        analyze_features_tool=MagicMock(),
        llm_provider=MagicMock(),
        tool_registry=registry,
        geo_agent=_IdleAgent(),  # type: ignore[arg-type]
    )


@pytest.fixture
async def client(settings: Settings):
    app = create_app(settings, services=_services(settings))
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        yield http, app


async def test_liveness_does_not_require_dependencies(client):
    http, app = client
    # Even if DB/Ollama probes would fail, /health stays ok and does not use them.
    response = await http.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert app.state.embedding_provider.is_loaded is False


async def test_readiness_uses_bounded_checks(client, monkeypatch):
    http, app = client

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def execute(self, *_args, **_kwargs):
            return None

    db = MagicMock()
    db.session = MagicMock(return_value=FakeSession())
    app.state.database = db
    app.dependency_overrides[get_database] = lambda: db

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"models": [{"name": "deepseek-r1:7b"}]}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.timeout = kwargs.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url: str):
            assert url.endswith("/api/tags")
            return FakeResponse()

    monkeypatch.setattr("app.api.health.httpx.AsyncClient", FakeAsyncClient)
    response = await http.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert {check["name"] for check in body["checks"]} == {"database", "ollama"}


async def test_readiness_checks_avalai_instead_of_ollama_when_selected(monkeypatch):
    settings = Settings(
        app_env="test",
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:5433/osm_geoagent_test",
        llm_provider="avalai",
        avalai_api_key="placeholder-not-a-real-key",
        avalai_model="gemini-3.6-flash",
        ready_check_timeout_seconds=0.2,
        _env_file=None,  # type: ignore[call-arg]
    )
    app = create_app(settings, services=_services(settings))
    transport = httpx.ASGITransport(app=app)

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def execute(self, *_args, **_kwargs):
            return None

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "gemini-3.6-flash"}]}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url: str, headers=None):
            assert url.endswith("/models")
            assert headers is not None
            assert str(headers.get("Authorization", "")).startswith("Bearer ")
            assert "placeholder-not-a-real-key" in str(headers.get("Authorization"))
            return FakeResponse()

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        db = MagicMock()
        db.session = MagicMock(return_value=FakeSession())
        app.state.database = db
        app.dependency_overrides[get_database] = lambda: db
        monkeypatch.setattr("app.api.health.httpx.AsyncClient", FakeAsyncClient)
        response = await http.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert {check["name"] for check in body["checks"]} == {"database", "avalai"}
    assert "ollama" not in {check["name"] for check in body["checks"]}


async def test_readiness_not_ready_when_database_down(client, monkeypatch):
    http, app = client

    class BoomSession:
        async def __aenter__(self):
            raise ConnectionError("db down")

        async def __aexit__(self, *args):
            return None

    db = MagicMock()
    db.session = MagicMock(return_value=BoomSession())
    app.state.database = db
    app.dependency_overrides[get_database] = lambda: db

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url: str):
            raise httpx.ConnectError("ollama down", request=None)

    monkeypatch.setattr("app.api.health.httpx.AsyncClient", FakeAsyncClient)
    response = await http.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
