"""Overpass retry, error classification, and grounded failure handling (offline)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from app.agent.accumulator import ResultAccumulator
from app.agent.contracts import GeoAgentRequest, GeoAgentResponse, TraceEvent
from app.agent.loop import LoopLimits
from app.agent.orchestrator import PlannerExecutorAgent, _grounded_live_failure_answer
from app.analytics.datasets import DatasetRegistry
from app.api.schemas import derive_result_status, to_agent_query_response
from app.core.errors import (
    OverpassBadResponseError,
    OverpassTimeoutError,
    OverpassUpstreamError,
)
from app.llm.contracts import ChatMessage, LLMResponse, ToolCall
from app.osm.client import HttpOverpassClient
from app.osm.geojson import OverpassGeoJsonEncoder
from app.osm.query_spec import OsmFeatureQuery, TagFilter
from app.places.contracts import PlaceRegistry
from app.tools.context import AnalysisRunState, GroundingState, ToolContext
from app.tools.overpass_failures import format_overpass_failure_observation
from app.tools.query_osm import QueryOsmTool, QueryOsmToolError
from app.tools.registry import ToolRegistry


def _client(
    handler: object,
    *,
    max_attempts: int = 2,
    retry_backoff_seconds: float = 0.01,
    max_response_bytes: int = 10_000,
    timeout_seconds: float = 5.0,
) -> HttpOverpassClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    http = httpx.AsyncClient(transport=transport, base_url="https://overpass.test")
    return HttpOverpassClient(
        base_url="https://overpass.test/api/interpreter",
        timeout_seconds=timeout_seconds,
        max_response_bytes=max_response_bytes,
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        client=http,
    )


async def test_http_504_maps_to_overpass_timeout():
    client = _client(lambda request: httpx.Response(504, text="<html>Gateway Timeout</html>"))
    try:
        with pytest.raises(OverpassTimeoutError) as exc_info:
            await client.run("query")
    finally:
        await client.aclose()
    assert exc_info.value.code == "overpass_timeout"
    assert exc_info.value.upstream_status == 504
    assert "<html" not in exc_info.value.message.lower()


async def test_504_is_retried_once_then_stops():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(504, text="<html>timeout</html>")

    client = _client(handler, max_attempts=2, retry_backoff_seconds=0.01)
    try:
        with pytest.raises(OverpassTimeoutError) as exc_info:
            await client.run("query")
    finally:
        await client.aclose()
    assert calls["n"] == 2
    assert exc_info.value.attempts == 2


async def test_successful_retry_after_504_returns_result():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(504, text="<html>timeout</html>")
        return httpx.Response(200, json={"elements": []})

    client = _client(handler, max_attempts=2, retry_backoff_seconds=0.01)
    try:
        response = await client.run("query")
    finally:
        await client.aclose()
    assert calls["n"] == 2
    assert response.elements == ()
    assert response.attempts == 2


async def test_400_is_not_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad query")

    client = _client(handler, max_attempts=2)
    try:
        with pytest.raises(OverpassBadResponseError):
            await client.run("query")
    finally:
        await client.aclose()
    assert calls["n"] == 1


async def test_html_body_is_not_exposed_in_error_message():
    client = _client(
        lambda request: httpx.Response(
            503,
            text="<!DOCTYPE html><html><body>Service Unavailable</body></html>",
        )
    )
    try:
        with pytest.raises(OverpassUpstreamError) as exc_info:
            await client.run("query")
    finally:
        await client.aclose()
    assert "html" not in exc_info.value.message.lower()
    assert "Service Unavailable" not in exc_info.value.message


async def test_query_osm_failure_observation_preserves_grounding():
    query = OsmFeatureQuery(
        place="Tehran, Iran",
        tags=[TagFilter(key="leisure", value="park")],
        limit=20,
    )
    error = OverpassTimeoutError(
        "The configured Overpass service returned HTTP 504",
        upstream_status=504,
        attempts=2,
    )
    observation = format_overpass_failure_observation(query, error)
    payload = json.loads(observation)
    tool_error = payload["tool_error"]
    assert tool_error["code"] == "overpass_timeout"
    assert tool_error["grounding_remains_valid"] is True
    assert tool_error["validated_tags"] == ["leisure=park"]
    assert tool_error["scope"] == "Tehran, Iran"
    assert tool_error["effective_limit"] == 20
    assert tool_error["retryable"] is True
    assert any(
        "do not invent replacement osm tags" in item.lower() for item in tool_error["instructions"]
    )


async def test_query_osm_tool_failure_does_not_fabricate_geojson():
    client = _client(lambda request: httpx.Response(504, text="<html>Gateway Timeout</html>"))
    tool = QueryOsmTool(
        client,
        OverpassGeoJsonEncoder(),
        timeout_seconds=25,
        max_results=1000,
    )
    context = ToolContext(
        datasets=DatasetRegistry(),
        analysis=AnalysisRunState(),
        user_message="Find parks in Tehran",
        places=PlaceRegistry(),
        grounding=GroundingState(),
    )
    try:
        with pytest.raises(QueryOsmToolError) as exc_info:
            await tool.execute(
                OsmFeatureQuery(
                    place="Tehran, Iran",
                    tags=[TagFilter(key="leisure", value="park")],
                    limit=20,
                ),
                context,
            )
    finally:
        await tool.aclose()
    error = exc_info.value
    assert error.code == "overpass_timeout"
    assert "leisure=park" in error.observation
    assert error.failure_meta["effective_limit"] == 20
    assert error.failure_meta["grounding_remains_valid"] is True


class _ScriptedLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.model_name = "fake"
        self.provider_name = "fake"

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[Any] | None = None,
        options: Any = None,
    ) -> LLMResponse:
        del messages, tools, options
        return self._responses.pop(0)

    async def aclose(self) -> None:
        return None


async def test_overpass_timeout_becomes_grounded_final_answer_not_llm_error():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(504, text="<html>Gateway Timeout</html>")

    client = _client(handler, max_attempts=2, retry_backoff_seconds=0.01)
    tool = QueryOsmTool(client, OverpassGeoJsonEncoder(), timeout_seconds=25, max_results=1000)
    registry = ToolRegistry()
    registry.register(tool)
    llm = _ScriptedLLM(
        [
            LLMResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="1",
                        name="query_osm",
                        arguments={
                            "place": "Tehran, Iran",
                            "tags": [{"key": "leisure", "value": "park"}],
                            "limit": 20,
                        },
                    ),
                ),
            ),
            LLMResponse(
                content=(
                    "Maybe try public_park or invent coordinates lat=35.6. "
                    "Write Overpass QL instead."
                ),
                tool_calls=(),
            ),
        ]
    )
    agent = PlannerExecutorAgent(llm, registry, LoopLimits(max_tool_rounds=4, max_tool_calls=8))
    result = await agent.run(
        GeoAgentRequest(message="Find up to 20 leisure=park features in Tehran, Iran.")
    )
    await tool.aclose()

    assert calls["n"] == 2  # client retry, not a second model tool call
    assert result.live_query_failed is True
    assert result.live_error_code == "overpass_timeout"
    assert result.geojson is None
    assert result.feature_count is None
    assert result.effective_limit == 20
    assert result.scope_summary == "Search area: Tehran, Iran"
    assert result.validated_tags == ["leisure=park"]
    assert result.stop_reason == "final_answer"
    assert "leisure=park" in result.answer
    assert "public_park" not in result.answer.lower()
    assert "no live features" in result.answer.lower()
    assert any(
        event.kind == "tool_error" and event.error_code == "overpass_timeout"
        for event in result.trace
    )
    public = to_agent_query_response(result, request_id="t1")
    assert public.status == "timed_out"
    assert public.live_data_available is False
    assert public.live_query_failed is True
    assert "<html" not in json.dumps(public.model_dump(mode="json")).lower()


async def test_empty_featurecollection_is_not_a_timeout():
    result = GeoAgentResponse(
        answer="No parks found.",
        overpass_query="out geom 20;",
        geojson={"type": "FeatureCollection", "features": []},
        feature_count=0,
        effective_limit=20,
        scope_summary="Search area: Tehran, Iran",
        validated_tags=["leisure=park"],
        live_query_failed=False,
        stop_reason="final_answer",
        model="fake",
    )
    assert derive_result_status(result) == "no_matching_features"


def test_grounded_failure_answer_keeps_leisure_park():
    state = ResultAccumulator()
    state.validated_tags = ["leisure=park"]
    state.scope_summary = "Search area: Tehran, Iran"
    state.live_error_code = "overpass_timeout"
    state.live_query_failed = True
    answer = _grounded_live_failure_answer(state)
    assert "leisure=park" in answer
    assert "timed out" in answer.lower()
    assert "public_park" not in answer


def test_protocol_repair_event_is_warning_not_tool_error():
    events = [
        TraceEvent(
            kind="protocol_repair",
            message="prompt format corrected",
            error_code="llm_protocol_error",
            details={"status": "warning"},
        )
    ]
    result = GeoAgentResponse(
        answer="ok",
        trace=events,
        stop_reason="final_answer",
        model="fake",
        warnings=["llm_protocol_error (repair 1): bad"],
    )
    public = to_agent_query_response(result, request_id=None)
    step = public.execution_trace[0]
    assert step.event == "protocol_repair"
    assert step.status == "warning"
