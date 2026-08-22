"""Correctable tool_argument_error observations and bounded recovery."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from app.agent.contracts import GeoAgentRequest
from app.agent.loop import LoopLimits
from app.agent.orchestrator import PlannerExecutorAgent
from app.llm.contracts import ToolCall
from app.osm.query_spec import OsmFeatureQuery
from app.tools.context import ToolContext
from app.tools.contracts import ToolOutcome
from app.tools.query_osm import OsmQuerySource, QueryOsmResult
from app.tools.registry import ToolRegistry
from app.tools.search_osm_knowledge import SearchOsmKnowledgeArgs, SearchOsmKnowledgeResult
from app.tools.validation import format_tool_argument_observation
from pydantic import ValidationError

from tests.test_orchestrator import ScriptedLLM, _final, _tool_response


class _KnowledgeTool:
    name = "search_osm_knowledge"
    description = "documentation"
    args_model = SearchOsmKnowledgeArgs

    async def execute(
        self, args: SearchOsmKnowledgeArgs, context: ToolContext
    ) -> ToolOutcome[SearchOsmKnowledgeResult]:
        del args, context
        return ToolOutcome(
            observation="Found 0 OSM documentation passage(s) (not live map data).",
            payload=SearchOsmKnowledgeResult(query="x", passages=[]),
        )


class _LiveTool:
    name = "query_osm"
    description = "live"
    args_model = OsmFeatureQuery

    def __init__(self) -> None:
        self.calls: list[OsmFeatureQuery] = []

    async def execute(
        self, args: OsmFeatureQuery, context: ToolContext
    ) -> ToolOutcome[QueryOsmResult]:
        del context
        self.calls.append(args)
        result = QueryOsmResult(
            feature_count=0,
            geojson={"type": "FeatureCollection", "features": []},
            overpass_query="out geom;",
            source=OsmQuerySource(
                endpoint="https://overpass.test/api/interpreter",
                retrieved_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            ),
            warnings=[],
            effective_limit=args.limit,
            scope_summary=f"Search area: {args.place}",
        )
        return ToolOutcome(
            observation="status=ok source=live_osm feature_count=0",
            payload=result,
        )


def test_format_tool_argument_observation_is_compact_json():
    with pytest.raises(ValidationError) as exc_info:
        OsmFeatureQuery.model_validate({"place": "Tehran", "max_items": 20})
    text = format_tool_argument_observation(
        "query_osm",
        exc_info.value,
        argument_keys=["place", "max_items"],
    )
    payload = json.loads(text)
    assert payload["tool_error"]["code"] == "tool_argument_error"
    assert payload["tool_error"]["tool"] == "query_osm"
    assert "tags" in payload["tool_error"]["missing_fields"]
    assert "max_items" in payload["tool_error"]["extra_fields"]
    assert "limit" in payload["tool_error"]["hint"]


async def test_invalid_query_osm_args_become_observation_and_can_be_corrected():
    live = _LiveTool()
    registry = ToolRegistry()
    registry.register(_KnowledgeTool())
    registry.register(live)
    llm = ScriptedLLM(
        [
            _tool_response(
                ToolCall(
                    id="1",
                    name="query_osm",
                    arguments={"place": "Tehran, Iran", "max_items": 20},
                )
            ),
            _tool_response(
                ToolCall(
                    id="2",
                    name="query_osm",
                    arguments={
                        "place": "Tehran, Iran",
                        "tags": [{"key": "leisure", "value": "park"}],
                        "limit": 20,
                    },
                )
            ),
            _final("Found parks in Tehran using leisure=park."),
        ]
    )
    agent = PlannerExecutorAgent(llm, registry, LoopLimits(max_tool_rounds=4, max_tool_calls=8))
    result = await agent.run(
        GeoAgentRequest(message="Find public parks in Tehran, Iran. Return at most 20 features.")
    )
    assert result.stop_reason == "final_answer"
    assert live.calls
    assert live.calls[0].place == "Tehran, Iran"
    assert live.calls[0].limit == 20
    # Intermediate argument error must not be a fatal errors[] entry.
    assert not any(item.startswith("tool_argument_error:") for item in result.errors)
    assert any("tool_argument_error" in warning for warning in result.warnings)


async def test_repeated_identical_invalid_call_stops_safely():
    registry = ToolRegistry()
    registry.register(_KnowledgeTool())
    registry.register(_LiveTool())
    bad = ToolCall(
        id="1",
        name="query_osm",
        arguments={"place": "Tehran", "max_items": 20},
    )
    llm = ScriptedLLM(
        [
            _tool_response(bad),
            _tool_response(ToolCall(id="2", name=bad.name, arguments=dict(bad.arguments))),
            _final("should not be used"),
        ]
    )
    agent = PlannerExecutorAgent(llm, registry, LoopLimits(max_tool_rounds=4, max_tool_calls=8))
    result = await agent.run(GeoAgentRequest(message="Find parks in Tehran"))
    assert result.stop_reason == "llm_error"
    assert any("repeated identical invalid" in err for err in result.errors)
