"""Agent HTTP API tests (offline; fake orchestrator / dependencies)."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from app.agent.contracts import (
    GeoAgentRequest,
    GeoAgentResponse,
    SourceReference,
    TraceEvent,
)
from app.api.dependencies import get_geo_agent
from app.bootstrap import ApplicationServices
from app.core.config import Settings
from app.core.errors import (
    EmbeddingError,
    LLMError,
    LLMProtocolError,
    LLMTimeoutError,
    OverpassError,
    OverpassRateLimitedError,
    OverpassTimeoutError,
    ToolTimeoutError,
)
from app.main import create_app
from app.osm.contracts import OSM_ATTRIBUTION
from app.rag.contracts import RetrievedPassage

_PASSAGE = RetrievedPassage(
    content="Public parks use leisure=park.",
    document_title="Tag:leisure=park",
    section="Description",
    source_url="https://wiki.openstreetmap.org/wiki/Tag:leisure%3Dpark",
    score=0.91,
)

_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [13.4, 52.5]},
            "properties": {
                "osm_type": "node",
                "osm_id": 1,
                "tags": {"leisure": "park"},
                "attribution": OSM_ATTRIBUTION,
            },
        }
    ],
}


class FakeAgent:
    def __init__(self, result: GeoAgentResponse | BaseException) -> None:
        self._result = result
        self.calls: list[GeoAgentRequest] = []

    async def run(self, request: GeoAgentRequest) -> GeoAgentResponse:
        self.calls.append(request)
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


def _response(**overrides: Any) -> GeoAgentResponse:
    base = GeoAgentResponse(
        answer="ok",
        sources=[],
        trace=[TraceEvent(kind="request_received", message="received request")],
        stop_reason="final_answer",
        model="fake-model",
    )
    return base.model_copy(update=overrides)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_env="test",
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:5433/osm_geoagent_test",
        ollama_model="deepseek-r1:7b",
        agent_request_timeout_seconds=5.0,
        ready_check_timeout_seconds=0.2,
        _env_file=None,  # type: ignore[call-arg]
    )


def _fake_services(settings: Settings, agent: FakeAgent) -> ApplicationServices:
    registry = MagicMock()
    registry.names = ("query_osm", "search_osm_knowledge")
    return ApplicationServices(
        settings=settings,
        database=MagicMock(),
        embedding_provider=MagicMock(is_loaded=False),
        query_osm_tool=MagicMock(),
        resolve_place_tool=MagicMock(),
        analyze_features_tool=MagicMock(),
        llm_provider=MagicMock(),
        tool_registry=registry,
        geo_agent=agent,  # type: ignore[arg-type]
    )


@pytest.fixture
async def api_client(settings: Settings):
    """Client with a default successful fake agent; tests may replace it."""
    agent = FakeAgent(_response(answer="default"))
    app = create_app(settings, services=_fake_services(settings, agent))
    app.state.fake_agent = agent  # type: ignore[attr-defined]
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        yield http, app


def _set_agent(app: Any, agent: FakeAgent) -> None:
    app.state.geo_agent = agent
    app.dependency_overrides[get_geo_agent] = lambda: agent


async def test_documentation_only_response(api_client):
    http, app = api_client
    _set_agent(
        app,
        FakeAgent(
            _response(
                answer="leisure=park is for parks; landuse=grass is managed grass.",
                passages=[_PASSAGE],
                sources=[
                    SourceReference(
                        kind="osm_documentation",
                        title="Tag:leisure=park",
                        url=_PASSAGE.source_url,
                    )
                ],
                trace=[
                    TraceEvent(kind="request_received", message="received"),
                    TraceEvent(
                        kind="tool_call",
                        message="search",
                        tool_name="search_osm_knowledge",
                        round_index=1,
                    ),
                    TraceEvent(
                        kind="tool_result",
                        message="received 1 documentation passage(s)",
                        tool_name="search_osm_knowledge",
                        round_index=1,
                    ),
                    TraceEvent(kind="final_answer", message="done", round_index=2),
                ],
            )
        ),
    )
    response = await http.post(
        "/api/v1/agent/query",
        json={"message": "What is the difference between leisure=park and landuse=grass?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["geojson"] is None
    assert body["knowledge_sources"][0]["title"] == "Tag:leisure=park"
    assert body["knowledge_sources"][0]["section"] == "Description"
    assert body["attribution"] is None
    assert body["stop_reason"] == "final_answer"


async def test_live_osm_response_with_geojson(api_client):
    http, app = api_client
    _set_agent(
        app,
        FakeAgent(
            _response(
                answer="Found parks in Berlin.",
                geojson=_GEOJSON,
                feature_count=1,
                overpass_query='area["name"="Berlin"]; out geom;',
                sources=[
                    SourceReference(kind="osm_features", title="OpenStreetMap features (Overpass)")
                ],
            )
        ),
    )
    response = await http.post(
        "/api/v1/agent/query",
        json={"message": "Find leisure=park features in Berlin."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["geojson"]["type"] == "FeatureCollection"
    assert body["feature_count"] == 1
    assert body["overpass_query"]
    assert body["attribution"] == OSM_ATTRIBUTION
    assert body["knowledge_sources"] == []


async def test_rag_then_overpass_sequence(api_client):
    http, app = api_client
    _set_agent(
        app,
        FakeAgent(
            _response(
                answer="Public parks are leisure=park; found live features.",
                passages=[_PASSAGE],
                geojson=_GEOJSON,
                feature_count=1,
                overpass_query="out geom;",
                trace=[
                    TraceEvent(kind="request_received", message="received"),
                    TraceEvent(
                        kind="tool_call",
                        message="rag",
                        tool_name="search_osm_knowledge",
                        round_index=1,
                    ),
                    TraceEvent(
                        kind="tool_result",
                        message="docs",
                        tool_name="search_osm_knowledge",
                        round_index=1,
                    ),
                    TraceEvent(
                        kind="tool_call",
                        message="osm",
                        tool_name="query_osm",
                        round_index=2,
                    ),
                    TraceEvent(
                        kind="tool_result",
                        message="features",
                        tool_name="query_osm",
                        round_index=2,
                    ),
                    TraceEvent(kind="final_answer", message="done", round_index=3),
                ],
            )
        ),
    )
    response = await http.post(
        "/api/v1/agent/query",
        json={"message": "Find public parks in Berlin."},
    )
    assert response.status_code == 200
    body = response.json()
    tools = [step["tool"] for step in body["execution_trace"] if step["event"] == "tool_call"]
    assert tools == ["search_osm_knowledge", "query_osm"]
    assert body["knowledge_sources"]
    assert body["geojson"] is not None


async def test_zero_feature_response(api_client):
    http, app = api_client
    empty = {"type": "FeatureCollection", "features": []}
    _set_agent(
        app,
        FakeAgent(
            _response(
                answer="Zero features matched; I will not invent parks.",
                geojson=empty,
                feature_count=0,
                overpass_query="out geom;",
            )
        ),
    )
    response = await http.post(
        "/api/v1/agent/query",
        json={"message": "Find leisure=park in Berlin"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["feature_count"] == 0
    assert body["geojson"]["features"] == []
    assert "invent" in body["answer"].lower()


async def test_persian_request(api_client):
    http, app = api_client
    # Persian text intentionally exercises Unicode request handling.
    answer = "\u067e\u0627\u0631\u06a9\u200c\u0647\u0627 \u0628\u0627 leisure=park."
    message = (
        "\u067e\u0627\u0631\u06a9\u200c\u0647\u0627\u06cc \u0639\u0645\u0648\u0645\u06cc "
        "\u0628\u0631\u0644\u06cc\u0646 \u0631\u0627 \u0627\u0632 OpenStreetMap "
        "\u067e\u06cc\u062f\u0627 \u06a9\u0646"
    )
    agent = FakeAgent(_response(answer=answer))
    _set_agent(app, agent)
    response = await http.post("/api/v1/agent/query", json={"message": message})
    assert response.status_code == 200
    assert agent.calls[0].message == message


@pytest.mark.parametrize("payload", [{"message": "   "}, {"message": "\n\t"}])
async def test_whitespace_only_message_rejected(api_client, payload):
    http, _ = api_client
    response = await http.post("/api/v1/agent/query", json=payload)
    assert response.status_code == 422


async def test_oversized_message_rejected(api_client):
    http, _ = api_client
    response = await http.post("/api/v1/agent/query", json={"message": "x" * 2001})
    assert response.status_code == 422


async def test_unknown_input_field_rejected(api_client):
    http, _ = api_client
    response = await http.post(
        "/api/v1/agent/query",
        json={"message": "Find parks", "overpass_ql": "node;out;"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMError("Ollama unreachable"), 503, "llm_error"),
        (LLMTimeoutError("Ollama timed out"), 504, "llm_timeout"),
        (LLMProtocolError("bad protocol"), 503, "llm_protocol_error"),
        (EmbeddingError("model missing"), 503, "embedding_error"),
        (ToolTimeoutError("Overpass timed out"), 504, "tool_timeout"),
        (
            OverpassRateLimitedError(
                "Overpass rate limited the request (HTTP 429)",
                upstream_status=429,
            ),
            429,
            "overpass_rate_limited",
        ),
        (
            OverpassTimeoutError(
                "The configured Overpass service returned HTTP 504",
                upstream_status=504,
            ),
            504,
            "overpass_timeout",
        ),
        (OverpassError("Overpass server error HTTP 503: boom"), 503, "overpass_error"),
    ],
)
async def test_upstream_errors_map_to_http_status(api_client, error, status, code):
    http, app = api_client
    _set_agent(app, FakeAgent(error))
    response = await http.post("/api/v1/agent/query", json={"message": "Find parks"})
    assert response.status_code == status
    body = response.json()
    assert body["code"] == code
    assert "password" not in body["detail"].lower()
    assert "traceback" not in body["detail"].lower()
    assert "request_id" in body


async def test_database_retrieval_failure_via_llm_error_result(api_client):
    http, app = api_client
    _set_agent(
        app,
        FakeAgent(
            _response(
                answer="Knowledge search unavailable.",
                stop_reason="llm_error",
                errors=["embedding_error: knowledge search unavailable: db down"],
            )
        ),
    )
    response = await http.post("/api/v1/agent/query", json={"message": "What tag is a park?"})
    assert response.status_code == 503
    assert response.json()["stop_reason"] == "llm_error"


async def test_max_rounds_and_max_calls_are_successful_stops(api_client):
    http, app = api_client
    for reason in ("max_tool_rounds", "max_tool_calls"):
        _set_agent(
            app,
            FakeAgent(
                _response(
                    answer=f"Stopped because the tool budget was reached ({reason}).",
                    stop_reason=reason,  # type: ignore[arg-type]
                )
            ),
        )
        response = await http.post("/api/v1/agent/query", json={"message": "Find parks"})
        assert response.status_code == 200
        assert response.json()["stop_reason"] == reason


async def test_unexpected_internal_error_is_sanitized(api_client):
    http, app = api_client

    class BoomAgent:
        async def run(self, request: GeoAgentRequest) -> GeoAgentResponse:
            raise RuntimeError("secret /var/lib/data password=supersecret")

    app.state.geo_agent = BoomAgent()
    app.dependency_overrides[get_geo_agent] = lambda: BoomAgent()
    response = await http.post("/api/v1/agent/query", json={"message": "Find parks"})
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "internal_error"
    assert "supersecret" not in body["detail"]
    assert "/var/lib" not in body["detail"]


async def test_response_contains_no_think_blocks(api_client):
    http, app = api_client
    _set_agent(
        app,
        FakeAgent(
            _response(
                answer="Visible answer",
                trace=[TraceEvent(kind="final_answer", message="Visible answer")],
            )
        ),
    )
    response = await http.post("/api/v1/agent/query", json={"message": "Hello there"})
    raw = response.text.lower()
    assert "<think>" not in raw
    assert "</think>" not in raw
    assert "chain-of-thought" not in raw


async def test_knowledge_sources_distinct_and_only_query_osm_sets_geojson(api_client):
    http, app = api_client
    _set_agent(
        app,
        FakeAgent(
            _response(
                answer="Both.",
                passages=[_PASSAGE],
                geojson=_GEOJSON,
                feature_count=1,
            )
        ),
    )
    body = (await http.post("/api/v1/agent/query", json={"message": "parks"})).json()
    assert body["knowledge_sources"][0]["url"].startswith("https://wiki.openstreetmap.org")
    assert body["geojson"]["type"] == "FeatureCollection"


async def test_agent_logs_do_not_include_full_geojson(api_client, caplog):
    http, app = api_client
    _set_agent(
        app,
        FakeAgent(
            _response(
                answer="Found parks.",
                geojson=_GEOJSON,
                feature_count=1,
            )
        ),
    )
    with caplog.at_level(logging.INFO, logger="app.api.agent"):
        await http.post("/api/v1/agent/query", json={"message": "Find parks"})
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert "FeatureCollection" not in joined
    assert "coordinates" not in joined
    assert "feature_count=1" in joined


async def test_openapi_contains_agent_endpoint(api_client):
    http, _ = api_client
    schema = (await http.get("/openapi.json")).json()
    assert "/api/v1/agent/query" in schema["paths"]
    post = schema["paths"]["/api/v1/agent/query"]["post"]
    assert "AgentQueryResponse" in str(schema.get("components", {}).get("schemas", {})) or (
        "response" in post
    )
    assert post["summary"]


async def test_request_id_header_is_echoed(api_client):
    http, app = api_client
    _set_agent(app, FakeAgent(_response(answer="ok")))
    response = await http.post(
        "/api/v1/agent/query",
        json={"message": "Find parks"},
        headers={"X-Request-ID": "test-req-1"},
    )
    assert response.headers["X-Request-ID"] == "test-req-1"
    assert response.json()["request_id"] == "test-req-1"


async def test_message_is_trimmed(api_client):
    http, app = api_client
    agent = FakeAgent(_response(answer="ok"))
    _set_agent(app, agent)
    await http.post("/api/v1/agent/query", json={"message": "  Find parks  "})
    assert agent.calls[0].message == "Find parks"


async def test_startup_uses_injected_services_once(settings: Settings):
    agent = FakeAgent(_response(answer="ok"))
    services = _fake_services(settings, agent)
    app = create_app(settings, services=services)
    async with app.router.lifespan_context(app):
        assert app.state.services is services
        assert app.state.geo_agent is agent
        assert app.state.embedding_provider.is_loaded is False


async def test_shutdown_closes_owned_services(settings: Settings, monkeypatch):
    agent = FakeAgent(_response(answer="ok"))
    services = MagicMock()
    services.settings = settings
    services.database = MagicMock()
    services.embedding_provider = MagicMock(is_loaded=False)
    services.query_osm_tool = MagicMock()
    services.llm_provider = MagicMock()
    services.tool_registry.names = ("query_osm", "search_osm_knowledge")
    services.geo_agent = agent
    services.aclose = AsyncMock()
    monkeypatch.setattr("app.main.build_application_services", lambda _settings: services)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        assert app.state.geo_agent is agent
    services.aclose.assert_awaited_once()
