"""Post-RAG prompted protocol: search_osm_knowledge → query_osm (offline)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone

from app.agent.accumulator import ResultAccumulator
from app.agent.contracts import GeoAgentRequest
from app.agent.loop import LoopLimits
from app.agent.orchestrator import PlannerExecutorAgent, _model_facing_tools
from app.llm.contracts import ChatMessage, LLMOptions, LLMResponse, ToolCall, ToolDefinition
from app.osm.query_spec import OsmFeatureQuery
from app.rag.contracts import RetrievedPassage
from app.tools.context import ToolContext
from app.tools.contracts import ToolOutcome
from app.tools.query_osm import OsmQuerySource, QueryOsmResult
from app.tools.registry import ToolRegistry
from app.tools.search_osm_knowledge import (
    SearchOsmKnowledgeArgs,
    SearchOsmKnowledgeResult,
    _summarise_for_llm,
)

_PASSAGES = [
    RetrievedPassage(
        content="Use leisure=park for a public park or recreation ground.",
        document_title="Tag:leisure=park",
        section="Description",
        source_url="https://wiki.openstreetmap.org/wiki/Tag:leisure%3Dpark",
        score=0.92,
    ),
    RetrievedPassage(
        content="Unrelated Overpass example about amenities.",
        document_title="Overpass API",
        section="Examples",
        source_url="https://wiki.openstreetmap.org/wiki/Overpass_API",
        score=0.41,
    ),
    RetrievedPassage(
        content="landuse=grass is for managed grass, not a public park.",
        document_title="Tag:landuse=grass",
        section="Description",
        source_url="https://wiki.openstreetmap.org/wiki/Tag:landuse%3Dgrass",
        score=0.55,
    ),
]


class ScriptedLLM:
    def __init__(self, script: Sequence[LLMResponse]) -> None:
        self._script = list(script)
        self.seen_tools: list[list[str]] = []
        self.seen_messages: list[list[ChatMessage]] = []

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
        del options
        self.seen_messages.append(list(messages))
        self.seen_tools.append([tool.name for tool in tools or []])
        return self._script.pop(0)

    async def aclose(self) -> None:
        return None


class FakeKnowledgeTool:
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
        return ToolOutcome(
            observation=_summarise_for_llm(_PASSAGES),
            payload=SearchOsmKnowledgeResult(query=args.query, passages=list(_PASSAGES)),
        )


class FakeLiveTool:
    name = "query_osm"
    description = "live"
    args_model = OsmFeatureQuery

    def __init__(self) -> None:
        self.calls: list[OsmFeatureQuery] = []

    async def execute(
        self, args: OsmFeatureQuery, context: ToolContext
    ) -> ToolOutcome[QueryOsmResult]:
        self.calls.append(args)
        result = QueryOsmResult(
            feature_count=0,
            geojson={"type": "FeatureCollection", "features": []},
            overpass_query="out geom 20;",
            source=OsmQuerySource(
                endpoint="https://overpass.test/api/interpreter",
                retrieved_at=datetime.now(tz=timezone.utc),
            ),
            effective_limit=args.limit,
            scope_summary=f"Search area: {args.place}",
        )
        context.datasets.register(result, args)
        return ToolOutcome(observation="status=empty feature_count=0", payload=result)


def test_llm_observation_is_compact_and_marks_tool_data():
    observation = _summarise_for_llm(_PASSAGES)
    assert "TOOL DATA" in observation
    assert "leisure=park" in observation
    assert "query_osm" in observation
    assert observation.count("\n") < 20
    # Full retrieval list is not dumped; observation is capped.
    assert "top 3 of 3" in observation


def test_analyze_features_hidden_for_simple_retrieval():
    defs = [
        ToolDefinition(name="search_osm_knowledge", description="d", parameters_schema={}),
        ToolDefinition(name="query_osm", description="d", parameters_schema={}),
        ToolDefinition(name="analyze_features", description="d", parameters_schema={}),
    ]
    state = ResultAccumulator()
    names = [
        item.name for item in _model_facing_tools(defs, state=state, user_message="Find parks")
    ]
    assert names == ["search_osm_knowledge", "query_osm"]


def test_analyze_features_requires_datasets_and_analytical_intent():
    defs = [
        ToolDefinition(name="search_osm_knowledge", description="d", parameters_schema={}),
        ToolDefinition(name="query_osm", description="d", parameters_schema={}),
        ToolDefinition(name="analyze_features", description="d", parameters_schema={}),
    ]
    state = ResultAccumulator()
    # Analytical but no datasets yet → still hidden.
    names = [
        item.name
        for item in _model_facing_tools(
            defs,
            state=state,
            user_message="Which area has more parks and denser coverage?",
        )
    ]
    assert "analyze_features" not in names

    class _Refs:
        def __init__(self, refs: tuple[str, ...]) -> None:
            self._refs = refs

        def refs(self) -> tuple[str, ...]:
            return self._refs

    state.datasets = _Refs(("osm_result_1",))  # type: ignore[assignment]
    names = [
        item.name
        for item in _model_facing_tools(defs, state=state, user_message="Find parks in Tehran")
    ]
    assert "analyze_features" not in names
    names = [
        item.name
        for item in _model_facing_tools(
            defs,
            state=state,
            user_message="Which area has more parks?",
        )
    ]
    assert "analyze_features" in names


async def test_post_rag_turn_produces_valid_query_osm_without_protocol_error():
    knowledge = FakeKnowledgeTool()
    live = FakeLiveTool()
    registry = ToolRegistry()
    registry.register(knowledge)
    registry.register(live)

    llm = ScriptedLLM(
        [
            LLMResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="1",
                        name="search_osm_knowledge",
                        arguments={"query": "public park OpenStreetMap tag", "top_k": 5},
                    ),
                ),
                model="fake",
            ),
            LLMResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="2",
                        name="query_osm",
                        arguments={
                            "place": "Tehran, Iran",
                            "tags": [{"key": "leisure", "value": "park"}],
                            "limit": 20,
                        },
                    ),
                ),
                model="fake",
            ),
            LLMResponse(
                content="Found parks using leisure=park. No features matched in this test.",
                model="fake",
            ),
        ]
    )
    agent = PlannerExecutorAgent(llm, registry, LoopLimits(max_tool_rounds=4, max_tool_calls=8))
    result = await agent.run(
        GeoAgentRequest(message="Find public parks in Tehran, Iran. Return at most 20 features.")
    )

    assert result.stop_reason == "final_answer"
    assert not any(err.startswith("llm_protocol_error") for err in result.errors)
    assert [call.query for call in knowledge.calls]
    assert live.calls[0].place == "Tehran, Iran"
    assert live.calls[0].tags[0].key == "leisure"
    assert live.calls[0].tags[0].value == "park"
    assert live.calls[0].limit == 20
    # Second planning turn saw the compact observation as a tool message.
    second_turn = llm.seen_messages[1]
    tool_msgs = [message for message in second_turn if message.role == "tool"]
    assert tool_msgs
    assert "TOOL DATA" in tool_msgs[0].content or "osm_documentation" in tool_msgs[0].content
    # Simple retrieval never advertised analyze_features.
    assert all("analyze_features" not in names for names in llm.seen_tools)


async def test_protocol_repair_hint_is_compact_json():
    from app.agent.orchestrator import _PROTOCOL_REPAIR_HINT

    payload = json.loads(_PROTOCOL_REPAIR_HINT)
    assert payload["protocol_error"]["allowed_top_level_forms"] == [
        "tool_calls",
        "final_answer",
    ]
    assert "Return exactly one JSON object" in payload["protocol_error"]["instruction"]
