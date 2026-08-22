"""Bounded planner-executor orchestration with fake LLM and fake tools."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from app.agent.contracts import GeoAgentRequest
from app.agent.loop import LoopLimits
from app.agent.orchestrator import PlannerExecutorAgent
from app.core.errors import LLMTimeoutError, ToolExecutionError, ToolTimeoutError
from app.llm.contracts import ChatMessage, LLMOptions, LLMResponse, ToolCall, ToolDefinition
from app.osm.contracts import OSM_ATTRIBUTION, GeoJsonFeatureCollection
from app.rag.contracts import OSM_KNOWLEDGE_DOMAIN, RetrievedPassage
from app.tools.context import ToolContext
from app.tools.contracts import ToolOutcome
from app.tools.query_osm import OsmQuerySource, QueryOsmResult
from app.tools.registry import ToolRegistry
from app.tools.search_osm_knowledge import SearchOsmKnowledgeArgs, SearchOsmKnowledgeResult
from pydantic import BaseModel, ConfigDict, Field

_PASSAGE = RetrievedPassage(
    content="Public parks are tagged leisure=park.",
    document_title="Tag:leisure=park",
    section="Description",
    source_url="https://wiki.openstreetmap.org/wiki/Tag:leisure%3Dpark",
    score=0.9,
)

_EMPTY_FC: GeoJsonFeatureCollection = {"type": "FeatureCollection", "features": []}
_PARK_FC: GeoJsonFeatureCollection = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [13.4, 52.5]},
            "properties": {"osm_id": 1, "tags": {"leisure": "park"}},
        }
    ],
}


class ScriptedLLM:
    """Returns scripted replies and records every conversation snapshot."""

    def __init__(self, script: Sequence[LLMResponse | BaseException]) -> None:
        self._script = list(script)
        self.seen: list[list[ChatMessage]] = []
        self.tools_seen: list[list[ToolDefinition] | None] = []

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def provider_name(self) -> str:
        return "fake"

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDefinition] | None = None,
        options: LLMOptions | None = None,
    ) -> LLMResponse:
        self.seen.append(list(messages))
        self.tools_seen.append(tools)
        if not self._script:
            raise AssertionError("ScriptedLLM has no more responses")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        return None


class FakeKnowledgeTool:
    name = "search_osm_knowledge"
    description = "documentation"
    args_model = SearchOsmKnowledgeArgs

    def __init__(self, passages: list[RetrievedPassage] | None = None) -> None:
        self.passages = passages if passages is not None else [_PASSAGE]
        self.calls: list[SearchOsmKnowledgeArgs] = []

    async def execute(
        self, args: SearchOsmKnowledgeArgs, context: ToolContext
    ) -> ToolOutcome[SearchOsmKnowledgeResult]:
        del context
        self.calls.append(args)
        result = SearchOsmKnowledgeResult(query=args.query, passages=self.passages)
        return ToolOutcome(
            observation=(
                f"Found {len(self.passages)} OSM documentation passage(s) (not live map data)."
            ),
            payload=result,
        )


class QueryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    place: str = Field(min_length=1)
    tag_key: str = Field(default="leisure")
    tag_value: str = Field(default="park")


class FakeQueryOsmTool:
    name = "query_osm"
    description = "live features"
    args_model = QueryArgs

    def __init__(
        self,
        *,
        geojson: GeoJsonFeatureCollection | None = None,
        feature_count: int | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        self.geojson = geojson if geojson is not None else _PARK_FC
        self.feature_count = (
            feature_count if feature_count is not None else len(self.geojson.get("features", []))  # type: ignore[arg-type]
        )
        self.fail_with = fail_with
        self.calls: list[QueryArgs] = []

    async def execute(self, args: QueryArgs, context: ToolContext) -> ToolOutcome[QueryOsmResult]:
        del context
        self.calls.append(args)
        if self.fail_with is not None:
            raise self.fail_with
        result = QueryOsmResult(
            feature_count=self.feature_count,
            geojson=self.geojson,
            overpass_query=f'area["name"="{args.place}"]; out geom;',
            source=OsmQuerySource(
                endpoint="https://overpass.test/api/interpreter",
                retrieved_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            ),
            warnings=[],
            effective_limit=20,
            scope_summary=f"Search area: {args.place}",
        )
        return ToolOutcome(
            observation=(
                f"status=ok source=live_osm feature_count={result.feature_count} "
                f"warnings=0 query={args.tag_key}={args.tag_value} in {args.place}. "
                "Do not invent additional features beyond this result."
            ),
            payload=result,
        )


def _registry(*tools: Any) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _agent(
    script: Sequence[LLMResponse | BaseException],
    registry: ToolRegistry,
    *,
    max_tool_rounds: int = 4,
    max_tool_calls: int = 8,
) -> tuple[PlannerExecutorAgent, ScriptedLLM]:
    llm = ScriptedLLM(script)
    agent = PlannerExecutorAgent(
        llm,
        registry,
        LoopLimits(max_tool_rounds=max_tool_rounds, max_tool_calls=max_tool_calls),
    )
    return agent, llm


def _tool_response(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(content="", tool_calls=calls, model="fake-model")


def _final(text: str) -> LLMResponse:
    return LLMResponse(content=text, model="fake-model")


def _conversation_text(llm: ScriptedLLM) -> str:
    chunks: list[str] = []
    for messages in llm.seen:
        for message in messages:
            chunks.append(message.content)
    return "\n".join(chunks)


# --- routing behaviours (scripted LLM) ---


async def test_explicit_tag_routes_directly_to_query_osm():
    knowledge = FakeKnowledgeTool()
    query = FakeQueryOsmTool()
    agent, llm = _agent(
        [
            _tool_response(
                ToolCall(
                    id="c1",
                    name="query_osm",
                    arguments={"place": "Berlin", "tag_key": "leisure", "tag_value": "park"},
                )
            ),
            _final("Found leisure=park features in Berlin from live OSM data."),
        ],
        _registry(knowledge, query),
    )
    result = await agent.run(GeoAgentRequest(message="Find leisure=park features in Berlin."))

    assert knowledge.calls == []
    assert len(query.calls) == 1
    assert result.geojson == _PARK_FC
    assert result.feature_count == 1
    assert result.overpass_query is not None
    assert result.stop_reason == "final_answer"
    assert [event.kind for event in result.trace] == [
        "request_received",
        "llm_turn",
        "tool_call",
        "tool_result",
        "llm_turn",
        "final_answer",
    ]
    second_turn = llm.seen[1]
    assistant = next(message for message in second_turn if message.role == "assistant")
    assert assistant.tool_calls[0].id == "c1"
    observation = next(message for message in second_turn if message.role == "tool")
    assert observation.tool_call_id == "c1"


async def test_ambiguous_concept_routes_rag_then_query_osm():
    knowledge = FakeKnowledgeTool()
    query = FakeQueryOsmTool()
    agent, _ = _agent(
        [
            _tool_response(
                ToolCall(
                    id="c1",
                    name="search_osm_knowledge",
                    arguments={"query": "public parks OSM tag", "top_k": 5},
                )
            ),
            _tool_response(
                ToolCall(
                    id="c2",
                    name="query_osm",
                    arguments={"place": "Berlin", "tag_key": "leisure", "tag_value": "park"},
                )
            ),
            _final("Public parks in Berlin are leisure=park (from OSM docs + live query)."),
        ],
        _registry(knowledge, query),
    )
    result = await agent.run(GeoAgentRequest(message="Find public parks in Berlin."))

    assert len(knowledge.calls) == 1
    assert len(query.calls) == 1
    assert result.passages == [_PASSAGE]
    assert result.geojson == _PARK_FC
    assert {source.kind for source in result.sources} == {
        "osm_documentation",
        "osm_features",
    }


async def test_documentation_only_question_uses_only_rag():
    knowledge = FakeKnowledgeTool()
    query = FakeQueryOsmTool()
    agent, _ = _agent(
        [
            _tool_response(
                ToolCall(
                    id="c1",
                    name="search_osm_knowledge",
                    arguments={"query": "leisure=park vs landuse=grass", "top_k": 5},
                )
            ),
            _final("leisure=park is for parks; landuse=grass is managed grass."),
        ],
        _registry(knowledge, query),
    )
    result = await agent.run(
        GeoAgentRequest(message="What is the difference between leisure=park and landuse=grass?")
    )

    assert len(knowledge.calls) == 1
    assert query.calls == []
    assert result.geojson is None
    assert result.feature_count is None
    assert result.overpass_query is None
    assert all(source.kind == "osm_documentation" for source in result.sources)


async def test_final_answer_without_tool_call():
    agent, _ = _agent(
        [_final("Hello — ask me about OpenStreetMap tags or features.")],
        _registry(FakeKnowledgeTool(), FakeQueryOsmTool()),
    )
    result = await agent.run(GeoAgentRequest(message="Hi there"))
    assert result.answer.startswith("Hello")
    assert result.stop_reason == "final_answer"
    assert result.geojson is None


async def test_multiple_valid_sequential_tool_calls():
    knowledge = FakeKnowledgeTool()
    query = FakeQueryOsmTool()
    agent, _ = _agent(
        [
            _tool_response(
                ToolCall(id="c1", name="search_osm_knowledge", arguments={"query": "park"}),
                ToolCall(
                    id="c2",
                    name="query_osm",
                    arguments={"place": "Berlin", "tag_key": "leisure", "tag_value": "park"},
                ),
            ),
            _final("Done."),
        ],
        _registry(knowledge, query),
    )
    result = await agent.run(GeoAgentRequest(message="Find parks in Berlin"))
    assert len(knowledge.calls) == 1
    assert len(query.calls) == 1
    assert result.answer == "Done."


# --- failures and bounds ---


async def test_unknown_tool_request_becomes_structured_error():
    agent, _ = _agent(
        [
            _tool_response(ToolCall(id="c1", name="rm_rf", arguments={})),
            _final("I cannot run unknown tools."),
        ],
        _registry(FakeKnowledgeTool()),
    )
    result = await agent.run(GeoAgentRequest(message="delete everything"))
    assert any(event.kind == "tool_error" for event in result.trace)
    assert any("tool_not_registered" in error for error in result.errors)
    assert result.stop_reason == "final_answer"


async def test_malformed_tool_arguments_are_reported():
    query = FakeQueryOsmTool()
    agent, _ = _agent(
        [
            _tool_response(ToolCall(id="c1", name="query_osm", arguments={"place": ""})),
            _final("I need a valid place."),
        ],
        _registry(query),
    )
    result = await agent.run(GeoAgentRequest(message="Find parks"))
    # Correctable argument errors stay in warnings when the loop recovers.
    assert any("tool_argument_error" in warning for warning in result.warnings)
    assert query.calls == []
    assert any(event.kind == "tool_error" for event in result.trace)


async def test_malformed_prompted_json_triggers_protocol_repair_then_answer():
    agent, llm = _agent(
        [
            LLMResponse(content='{"tool_calls": [{"name": ', model="fake-model"),
            _final("Recovered after protocol error."),
        ],
        _registry(FakeKnowledgeTool()),
    )
    result = await agent.run(GeoAgentRequest(message="Find parks"))
    assert any("llm_protocol_error" in warning for warning in result.warnings)
    assert result.answer == "Recovered after protocol error."
    assert any("protocol_error" in message.content for message in llm.seen[1])


async def test_empty_native_style_reply_is_protocol_error():
    agent, _ = _agent(
        [
            LLMResponse(content="", model="fake-model"),
            _final("Ok after empty reply."),
        ],
        _registry(FakeKnowledgeTool()),
    )
    result = await agent.run(GeoAgentRequest(message="Find parks"))
    assert any("llm_protocol_error" in warning for warning in result.warnings)


async def test_tool_timeout_is_observed_not_success():
    query = FakeQueryOsmTool(fail_with=ToolTimeoutError("Overpass timed out"))
    agent, _ = _agent(
        [
            _tool_response(ToolCall(id="c1", name="query_osm", arguments={"place": "Berlin"})),
            _final("Overpass timed out; no features."),
        ],
        _registry(query),
    )
    result = await agent.run(GeoAgentRequest(message="Find parks in Berlin"))
    assert result.geojson is None
    assert any(event.kind == "tool_error" for event in result.trace)
    assert any("tool_timeout" in error for error in result.errors)
    assert "timed out" in result.answer.lower() or "no features" in result.answer.lower()


async def test_tool_execution_failure_is_not_fabricated_success():
    query = FakeQueryOsmTool(fail_with=ToolExecutionError("upstream refused"))
    agent, _ = _agent(
        [
            _tool_response(ToolCall(id="c1", name="query_osm", arguments={"place": "Berlin"})),
            _final("The live query failed."),
        ],
        _registry(query),
    )
    result = await agent.run(GeoAgentRequest(message="Find parks"))
    assert result.geojson is None
    assert any("tool_execution_error" in error for error in result.errors)
    assert "failed" in result.answer.lower()


async def test_llm_timeout_stops_with_llm_error():
    agent, _ = _agent(
        [LLMTimeoutError("Ollama request timed out after 1s")],
        _registry(FakeKnowledgeTool()),
    )
    result = await agent.run(GeoAgentRequest(message="Find parks"))
    assert result.stop_reason == "llm_error"
    assert any("llm_timeout" in error for error in result.errors)
    assert "think" not in result.answer.lower() or "timed out" in result.answer.lower()


async def test_maximum_tool_calls_stops_safely():
    query = FakeQueryOsmTool()
    agent, _ = _agent(
        [
            _tool_response(
                ToolCall(id="c1", name="query_osm", arguments={"place": "Berlin"}),
                ToolCall(id="c2", name="query_osm", arguments={"place": "Munich"}),
            ),
            _tool_response(ToolCall(id="c3", name="query_osm", arguments={"place": "Hamburg"})),
        ],
        _registry(query),
        max_tool_rounds=3,
        max_tool_calls=1,
    )
    result = await agent.run(GeoAgentRequest(message="Find parks"))
    assert len(query.calls) == 1
    assert result.stop_reason == "max_tool_calls"
    assert "budget" in result.answer.lower() or "limit" in result.answer.lower()


async def test_maximum_rounds_stops_safely():
    query = FakeQueryOsmTool()
    agent, _ = _agent(
        [
            _tool_response(ToolCall(id="c1", name="query_osm", arguments={"place": "Berlin"})),
            _tool_response(ToolCall(id="c2", name="query_osm", arguments={"place": "Munich"})),
        ],
        _registry(query),
        max_tool_rounds=1,
        max_tool_calls=8,
    )
    result = await agent.run(GeoAgentRequest(message="Find parks"))
    assert len(query.calls) == 1
    assert result.stop_reason == "max_tool_rounds"


async def test_zero_feature_overpass_result_is_preserved():
    query = FakeQueryOsmTool(geojson=_EMPTY_FC, feature_count=0)
    agent, _ = _agent(
        [
            _tool_response(ToolCall(id="c1", name="query_osm", arguments={"place": "Berlin"})),
            _final("Zero features matched; I will not invent parks."),
        ],
        _registry(query),
    )
    result = await agent.run(GeoAgentRequest(message="Find leisure=park in Berlin"))
    assert result.feature_count == 0
    assert result.geojson == _EMPTY_FC
    assert result.overpass_query is not None
    assert "invent" in result.answer.lower()


# --- observation / source / reasoning safety ---


async def test_full_geojson_is_not_appended_to_llm_conversation():
    query = FakeQueryOsmTool()
    agent, llm = _agent(
        [
            _tool_response(ToolCall(id="c1", name="query_osm", arguments={"place": "Berlin"})),
            _final("Done."),
        ],
        _registry(query),
    )
    result = await agent.run(GeoAgentRequest(message="Find parks"))
    joined = _conversation_text(llm)
    assert "FeatureCollection" not in joined
    assert '"coordinates"' not in joined
    assert result.geojson == _PARK_FC
    assert "source=live_osm" in joined
    assert "feature_count=1" in joined


async def test_only_query_osm_populates_geojson():
    knowledge = FakeKnowledgeTool()
    agent, _ = _agent(
        [
            _tool_response(
                ToolCall(id="c1", name="search_osm_knowledge", arguments={"query": "park"})
            ),
            _final("Docs only."),
        ],
        _registry(knowledge, FakeQueryOsmTool()),
    )
    result = await agent.run(GeoAgentRequest(message="What tag is a park?"))
    assert result.geojson is None
    assert result.passages


async def test_knowledge_and_live_sources_remain_distinct():
    knowledge = FakeKnowledgeTool()
    query = FakeQueryOsmTool()
    agent, _ = _agent(
        [
            _tool_response(
                ToolCall(id="c1", name="search_osm_knowledge", arguments={"query": "park"})
            ),
            _tool_response(ToolCall(id="c2", name="query_osm", arguments={"place": "Berlin"})),
            _final("Both."),
        ],
        _registry(knowledge, query),
    )
    result = await agent.run(GeoAgentRequest(message="parks"))
    kinds = [source.kind for source in result.sources]
    assert "osm_documentation" in kinds
    assert "osm_features" in kinds
    assert all(passage.source_url.startswith("https://wiki") for passage in result.passages)
    assert result.geojson is not None


async def test_hidden_reasoning_absent_from_result_and_trace():
    # Provider boundary strips <think>; orchestrator must not reintroduce it.
    agent, _ = _agent(
        [_final("Visible answer only.")],
        _registry(FakeKnowledgeTool()),
    )
    result = await agent.run(GeoAgentRequest(message="Hello"))
    blob = result.answer + json_trace(result)
    assert "<think>" not in blob
    assert "Visible answer only." in result.answer


def json_trace(result: Any) -> str:
    return " ".join(event.message for event in result.trace)


async def test_operational_trace_order_is_deterministic():
    knowledge = FakeKnowledgeTool()
    query = FakeQueryOsmTool()
    agent, _ = _agent(
        [
            _tool_response(
                ToolCall(id="c1", name="search_osm_knowledge", arguments={"query": "park"})
            ),
            _tool_response(ToolCall(id="c2", name="query_osm", arguments={"place": "Berlin"})),
            _final("Done."),
        ],
        _registry(knowledge, query),
    )
    result = await agent.run(GeoAgentRequest(message="parks"))
    assert [event.kind for event in result.trace] == [
        "request_received",
        "llm_turn",
        "tool_call",
        "tool_result",
        "llm_turn",
        "tool_call",
        "tool_result",
        "llm_turn",
        "final_answer",
    ]


async def test_prompted_and_native_modes_share_registry_tool_contracts():
    """Both provider modes consume the same ToolDefinition list from the registry."""
    registry = _registry(FakeKnowledgeTool(), FakeQueryOsmTool())
    definitions = registry.definitions()
    names = {item.name for item in definitions}
    assert names == {"search_osm_knowledge", "query_osm"}
    for definition in definitions:
        assert "properties" in definition.parameters_schema
        assert definition.description

    agent, llm = _agent(
        [_final("ok")],
        registry,
    )
    await agent.run(GeoAgentRequest(message="Hi"))
    assert llm.tools_seen[0] is not None
    assert {tool.name for tool in llm.tools_seen[0]} == names


async def test_attribution_constant_available_for_live_results():
    assert "OpenStreetMap" in OSM_ATTRIBUTION
    assert OSM_KNOWLEDGE_DOMAIN == "osm_knowledge"
