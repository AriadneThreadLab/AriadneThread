"""Regression coverage for the post-MVP corrective iteration."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
from app.agent.contracts import GeoAgentRequest, TraceEvent
from app.agent.loop import LoopLimits
from app.agent.orchestrator import PlannerExecutorAgent
from app.api.schemas import derive_result_status, sanitize_public_answer, to_agent_query_response
from app.llm.contracts import ChatMessage, LLMResponse, ToolCall
from app.llm.ollama import OllamaConfig, OllamaProvider
from app.llm.tool_protocol import parse_reply
from app.osm.contracts import GeoJsonFeatureCollection
from app.osm.query_spec import OsmFeatureQuery
from app.rag.contracts import RetrievedPassage
from app.tools.context import ToolContext
from app.tools.contracts import ToolOutcome
from app.tools.query_osm import OsmQuerySource, QueryOsmResult
from app.tools.registry import ToolRegistry
from app.tools.search_osm_knowledge import SearchOsmKnowledgeArgs, SearchOsmKnowledgeResult

from tests.test_orchestrator import ScriptedLLM, _final, _tool_response

_PASSAGE = RetrievedPassage(
    content="Public parks are tagged leisure=park. Do not use amenity=park.",
    document_title="Tag:leisure=park",
    section="Description",
    source_url="https://wiki.openstreetmap.org/wiki/Tag:leisure%3Dpark",
    score=0.91,
)

_FC: GeoJsonFeatureCollection = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [51.4, 35.7]},
            "properties": {"osm_type": "node", "osm_id": 1, "tags": {"leisure": "park"}},
        }
    ],
}


class KnowledgeTool:
    name = "search_osm_knowledge"
    description = "documentation"
    args_model = SearchOsmKnowledgeArgs

    def __init__(self) -> None:
        self.calls: list[SearchOsmKnowledgeArgs] = []

    async def execute(
        self, args: SearchOsmKnowledgeArgs, context: ToolContext
    ) -> ToolOutcome[SearchOsmKnowledgeResult]:
        del context
        self.calls.append(args)
        result = SearchOsmKnowledgeResult(query=args.query, passages=[_PASSAGE])
        return ToolOutcome(
            observation="Found 1 OSM documentation passage(s) (not live map data).",
            payload=result,
        )


class LiveTool:
    """Accepts OsmFeatureQuery-shaped arguments for trust + tag checks."""

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
            feature_count=1,
            geojson=_FC,
            overpass_query='area["name"="Tehran"]; out geom 20;',
            source=OsmQuerySource(
                endpoint="https://overpass.test/api/interpreter",
                retrieved_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            ),
            warnings=[],
            effective_limit=args.limit,
            scope_summary=f"Search area: {args.place}" if args.place else "scope",
        )
        return ToolOutcome(
            observation=(
                f"status=ok source=live_osm feature_count=1 warnings=0 "
                f"query=leisure=park in {args.place}. "
                "Do not invent additional features beyond this result."
            ),
            payload=result,
        )


def _agent(script: list[LLMResponse], *tools: object) -> PlannerExecutorAgent:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)  # type: ignore[arg-type]
    return PlannerExecutorAgent(
        ScriptedLLM(script),
        registry,
        LoopLimits(max_tool_rounds=4, max_tool_calls=8),
    )


async def test_fenced_tool_call_executes_query_osm_and_does_not_leak_protocol():
    fenced = """```json
{
  "tool_calls": [
    {
      "name": "query_osm",
      "arguments": {
        "place": "Tehran, Iran",
        "tags": [["leisure", "park"]],
        "limit": 20
      }
    }
  ]
}
```"""
    parsed = parse_reply(fenced)
    assert parsed.protocol_error is None
    assert parsed.content == ""
    assert parsed.tool_calls[0].name == "query_osm"

    knowledge = KnowledgeTool()
    live = LiveTool()
    agent = _agent(
        [
            LLMResponse(content="", tool_calls=parsed.tool_calls),
            _final("Found leisure=park features in Tehran."),
        ],
        knowledge,
        live,
    )
    result = await agent.run(
        GeoAgentRequest(message="Find up to 20 leisure=park features in Tehran, Iran.")
    )
    assert result.stop_reason == "final_answer"
    assert "tool_calls" not in result.answer
    assert "```" not in result.answer
    assert live.calls
    assert live.calls[0].place == "Tehran, Iran"
    assert live.calls[0].tags[0].key == "leisure"
    assert live.calls[0].tags[0].value == "park"
    assert result.geojson == _FC
    assert result.feature_count == 1


async def test_fenced_protocol_as_content_without_tool_calls_is_not_final_answer():
    agent = _agent(
        [
            LLMResponse(
                content="",
                tool_calls=(),
                protocol_error="model reply was not valid tool_calls/final_answer JSON",
            ),
            LLMResponse(
                content="",
                tool_calls=(),
                protocol_error="model reply was not valid tool_calls/final_answer JSON",
            ),
            LLMResponse(
                content="",
                tool_calls=(),
                protocol_error="model reply was not valid tool_calls/final_answer JSON",
            ),
        ],
        KnowledgeTool(),
        LiveTool(),
    )
    result = await agent.run(GeoAgentRequest(message="Find parks in Tehran"))
    assert result.stop_reason == "llm_error"
    assert "invalid structured response" in result.answer.lower()
    assert "tool_calls" not in result.answer
    assert any("llm_protocol_error" in err for err in result.errors)


async def test_ollama_prompted_mode_parses_fenced_tool_calls():
    from app.llm.contracts import ToolDefinition

    raw = (
        '```json\n{"tool_calls":[{"name":"query_osm","arguments":'
        '{"place":"Tehran, Iran","tags":[["leisure","park"]],"limit":20}}]}\n```'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "deepseek-r1:7b", "message": {"role": "assistant", "content": raw}},
        )

    config = OllamaConfig(
        base_url="http://127.0.0.1:11434",
        model="deepseek-r1:7b",
        num_ctx=4096,
        request_timeout_seconds=5.0,
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=config.base_url,
        timeout=httpx.Timeout(5.0),
    )
    provider = OllamaProvider(config, client=client)
    response = await provider.chat(
        [ChatMessage(role="user", content="parks?")],
        tools=[ToolDefinition(name="query_osm", description="x", parameters_schema={})],
    )
    await provider.aclose()
    assert response.has_tool_calls
    assert response.content == ""
    assert response.protocol_error is None
    assert response.tool_calls[0].arguments["place"] == "Tehran, Iran"


async def test_semantic_public_park_uses_rag_then_leisure_park_named_place():
    knowledge = KnowledgeTool()
    live = LiveTool()
    agent = _agent(
        [
            _tool_response(
                ToolCall(
                    id="c1",
                    name="search_osm_knowledge",
                    arguments={"query": "public park OSM tag", "top_k": 5},
                )
            ),
            _tool_response(
                ToolCall(
                    id="c2",
                    name="query_osm",
                    arguments={
                        "place": "Tehran, Iran",
                        "tags": [{"key": "leisure", "value": "park"}],
                        "limit": 20,
                    },
                )
            ),
            _final("Public parks in Tehran use leisure=park."),
        ],
        knowledge,
        live,
    )
    result = await agent.run(
        GeoAgentRequest(message="Find public parks in Tehran, Iran. Return at most 20 features.")
    )
    assert knowledge.calls
    assert live.calls
    assert live.calls[0].place == "Tehran, Iran"
    assert [(t.key, t.value) for t in live.calls[0].tags] == [("leisure", "park")]
    assert all(t.key != "amenity" for t in live.calls[0].tags)
    assert result.geojson == _FC


async def test_explicit_leisure_park_may_skip_rag():
    knowledge = KnowledgeTool()
    live = LiveTool()
    agent = _agent(
        [
            _tool_response(
                ToolCall(
                    id="c1",
                    name="query_osm",
                    arguments={
                        "place": "Tehran, Iran",
                        "tags": [["leisure", "park"]],
                        "limit": 20,
                    },
                )
            ),
            _final("Found leisure=park features."),
        ],
        knowledge,
        live,
    )
    result = await agent.run(
        GeoAgentRequest(message="Find up to 20 leisure=park features in Tehran, Iran.")
    )
    assert knowledge.calls == []
    assert live.calls[0].limit == 20
    assert result.effective_limit == 20


async def test_invented_bbox_is_rejected_before_tool_execution():
    knowledge = KnowledgeTool()
    live = LiveTool()
    agent = _agent(
        [
            _tool_response(
                ToolCall(
                    id="c1",
                    name="query_osm",
                    arguments={
                        "bbox": {
                            "west": 59.937208,
                            "east": 60.147208,
                            "north": 35.950469,
                            "south": 35.550469,
                        },
                        "tags": [["leisure", "park"]],
                        "limit": 20,
                    },
                )
            ),
            _final("I need a named place or user-supplied coordinates."),
        ],
        knowledge,
        live,
    )
    result = await agent.run(
        GeoAgentRequest(message="Find public parks in Tehran, Iran. Return at most 20 features.")
    )
    assert live.calls == []
    assert any("tool_argument_error" in warning for warning in result.warnings)
    assert result.geojson is None


async def test_api_sanitizer_strips_protocol_leakage():
    answer = sanitize_public_answer(
        '```json\n{"tool_calls":[{"name":"query_osm","arguments":{}}]}\n```',
        errors=["llm_protocol_error: bad"],
        feature_count=None,
    )
    assert "tool_calls" not in answer
    assert "invalid structured response" in answer.lower()


async def test_documentation_only_keeps_geojson_null_and_live_flag_false():
    from app.agent.contracts import GeoAgentResponse

    result = GeoAgentResponse(
        answer="leisure=park vs landuse=grass",
        passages=[_PASSAGE],
        trace=[TraceEvent(kind="final_answer", message="done")],
        stop_reason="final_answer",
        model="fake",
    )
    payload = to_agent_query_response(result, request_id="r1")
    assert payload.geojson is None
    assert payload.live_query_executed is False
    assert payload.status == "completed"


async def test_zero_live_results_remain_empty_feature_collection():
    from app.agent.contracts import GeoAgentResponse

    empty: GeoJsonFeatureCollection = {"type": "FeatureCollection", "features": []}
    result = GeoAgentResponse(
        answer="No matching features.",
        geojson=empty,
        feature_count=0,
        overpass_query="out geom 20;",
        effective_limit=20,
        scope_summary="Search area: Tehran, Iran",
        trace=[TraceEvent(kind="final_answer", message="done")],
        stop_reason="final_answer",
        model="fake",
    )
    payload = to_agent_query_response(result, request_id="r2")
    assert payload.geojson == empty
    assert payload.feature_count == 0
    assert payload.live_query_executed is True
    assert payload.status == "no_matching_features"


async def test_invalid_model_response_status():
    from app.agent.contracts import GeoAgentResponse

    result = GeoAgentResponse(
        answer=(
            "The language model returned an invalid structured response. "
            "No geographic query was executed."
        ),
        errors=["llm_protocol_error: bad json"],
        trace=[TraceEvent(kind="stopped", message="stopped", error_code="llm_protocol_error")],
        stop_reason="llm_error",
        model="fake",
    )
    assert derive_result_status(result) == "invalid_model_response"


async def test_think_blocks_remain_stripped_from_fenced_payload():
    raw = '<think>secret plan</think>\n```json\n{"final_answer":"Parks use leisure=park."}\n```'
    parsed = parse_reply(raw)
    assert "secret" not in parsed.content
    assert "<think>" not in parsed.content
    assert parsed.content == "Parks use leisure=park."
