"""Offline tests for compact multi-target comparison orchestration."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from app.agent.accumulator import ResultAccumulator
from app.agent.comparison_executor import (
    should_use_multi_target_runner,
)
from app.agent.comparison_workflow import (
    COMPARISON_PLANNER_SYSTEM,
    GREEN_SPACE_TAGS,
    ComparisonTargetDraft,
    MultiTargetComparisonPlan,
    canonicalize_place_query,
    extract_landmark_labels,
    is_multi_target_landmark_comparison,
    normalize_plan_with_user_context,
    seed_plan_from_user_message,
    tags_for_feature_concept,
)
from app.agent.contracts import GeoAgentRequest
from app.agent.loop import LoopLimits
from app.agent.orchestrator import PlannerExecutorAgent, _model_facing_tools
from app.agent.planning_diagnostics import measure_planning_messages
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.spatial_trust import untrusted_spatial_scope_error
from app.analytics.factory import build_analyze_features_tool
from app.api.schemas import derive_result_status
from app.core.errors import LLMTimeoutError, PlaceResolutionError
from app.llm.contracts import ChatMessage, LLMOptions, LLMResponse, ToolDefinition
from app.osm.contracts import GeoJsonConversionResult, OverpassElement, OverpassResponse
from app.places.contracts import GeocoderHit, PlaceRegistry
from app.places.nominatim import NominatimPlaceResolver
from app.places.selection import evaluate_hit, select_trusted_hit
from app.rag.contracts import RetrievedPassage
from app.tools.analyze_features import TOOL_NAME as ANALYZE_FEATURES
from app.tools.context import ToolContext
from app.tools.contracts import ToolOutcome
from app.tools.query_osm import TOOL_NAME as QUERY_OSM
from app.tools.query_osm import QueryOsmTool
from app.tools.registry import ToolRegistry
from app.tools.resolve_place import ResolvePlaceTool
from app.tools.search_osm_knowledge import SearchOsmKnowledgeArgs, SearchOsmKnowledgeResult

_COMPARE = (
    "Compare public parks within 2 km of the University of Tehran and Sharif "
    "University of Technology. Choose the best metric, compare them, and return "
    "the park features for both areas."
)
_COMPARE_GREEN = (
    "Compare green spaces within 2 km of the University of Tehran and Sharif "
    "University of Technology. Return the green-space features for both areas."
)


def test_multi_target_detection_and_seed_plan():
    assert is_multi_target_landmark_comparison(_COMPARE)
    assert should_use_multi_target_runner(_COMPARE)
    labels = extract_landmark_labels(_COMPARE)
    assert labels[0] == "University of Tehran"
    assert labels[1] == "Sharif University of Technology"
    seed = seed_plan_from_user_message(_COMPARE)
    assert seed is not None
    assert seed.radius_m == 2000
    assert seed.targets[0].place_query == "University of Tehran, Iran"
    assert "City Hall" not in seed.targets[0].place_query
    assert seed.feature_concept == "public parks"
    assert tags_for_feature_concept(seed.feature_concept) == ["leisure=park"]


def test_green_space_seed_uses_union_tags():
    assert is_multi_target_landmark_comparison(_COMPARE_GREEN)
    seed = seed_plan_from_user_message(_COMPARE_GREEN)
    assert seed is not None
    assert seed.feature_concept == "green spaces"
    assert seed.radius_m == 2000
    assert tags_for_feature_concept(seed.feature_concept) == list(GREEN_SPACE_TAGS)
    assert tags_for_feature_concept("public parks") == ["leisure=park"]


def test_normalize_keeps_green_spaces_when_planner_says_parks():
    plan = MultiTargetComparisonPlan(
        feature_concept="public parks",
        targets=[
            ComparisonTargetDraft(
                label="University of Tehran",
                place_query="University of Tehran, Iran",
            ),
            ComparisonTargetDraft(
                label="Sharif University of Technology",
                place_query="Sharif University of Technology, Iran",
            ),
        ],
        radius_m=500,
        comparison_goal="park availability",
    )
    normalized = normalize_plan_with_user_context(plan, _COMPARE_GREEN)
    assert normalized.feature_concept == "green spaces"
    assert normalized.radius_m == 2000
    assert tags_for_feature_concept(normalized.feature_concept) == list(GREEN_SPACE_TAGS)


def test_simple_tehran_query_not_multi_target():
    msg = "Find public parks in Tehran, Iran. Return at most 20 features."
    assert not is_multi_target_landmark_comparison(msg)


def test_canonicalize_rejects_unrelated_proposed_query():
    query = canonicalize_place_query(
        "University of Tehran",
        user_message=_COMPARE,
        proposed_query="Tehran City Hall, Tehran, Iran",
    )
    assert query == "University of Tehran, Iran"


def test_city_hall_candidate_rejected_for_university_query():
    hit = GeocoderHit(
        display_name="Tehran City Hall, Tehran, Iran",
        latitude=35.69,
        longitude=51.39,
        source_id="9",
        raw_class="amenity",
        raw_type="townhall",
    )
    decision = evaluate_hit("University of Tehran, Tehran, Iran", hit, wants_tehran=True)
    assert not decision.accepted
    assert decision.reason_code in {"name_mismatch", "unrelated_landmark_type"}


def test_unrelated_nominatim_candidates_raise():
    with pytest.raises(PlaceResolutionError):
        select_trusted_hit(
            "University of Tehran, Tehran, Iran",
            (
                GeocoderHit(
                    display_name="Tehran City Hall, Tehran, Iran",
                    latitude=35.69,
                    longitude=51.39,
                    source_id="9",
                    raw_class="amenity",
                    raw_type="townhall",
                ),
            ),
        )


def test_university_type_accepted_when_display_name_non_latin():
    hit = GeocoderHit(
        display_name="پردیس مرکزی دانشگاه تهران, تهران, ایران",
        latitude=35.7022,
        longitude=51.3950,
        source_id="111",
        raw_class="amenity",
        raw_type="university",
    )
    decision = evaluate_hit("University of Tehran, Iran", hit, wants_tehran=True)
    assert decision.accepted
    chosen = select_trusted_hit("University of Tehran, Iran", (hit,))
    assert chosen.source_id == "111"


def test_compact_planner_prompt_much_smaller_than_full_system():
    assert len(COMPARISON_PLANNER_SYSTEM) < 1200
    assert len(COMPARISON_PLANNER_SYSTEM) * 4 < len(SYSTEM_PROMPT)


def test_planning_diagnostics_do_not_include_prompt_body():
    messages = [
        ChatMessage(role="system", content=COMPARISON_PLANNER_SYSTEM),
        ChatMessage(role="user", content=_COMPARE),
    ]
    diag = measure_planning_messages(
        messages,
        tools=None,
        tool_catalog_text="",
        model="fake",
    )
    from dataclasses import asdict

    dumped = json.dumps(asdict(diag))
    assert COMPARISON_PLANNER_SYSTEM not in dumped
    assert _COMPARE not in dumped
    assert diag.eligible_tool_count == 0
    assert diag.total_message_chars < 2000


def test_initial_multi_target_eligibility_hides_query_and_analyze():
    defs = [
        ToolDefinition(name="search_osm_knowledge", description="d", parameters_schema={}),
        ToolDefinition(name="resolve_place", description="d", parameters_schema={}),
        ToolDefinition(name="query_osm", description="d", parameters_schema={}),
        ToolDefinition(name="analyze_features", description="d", parameters_schema={}),
    ]
    state = ResultAccumulator()
    names = [
        item.name
        for item in _model_facing_tools(
            defs,
            state=state,
            user_message=_COMPARE,
            places=PlaceRegistry(),
        )
    ]
    assert QUERY_OSM not in names
    assert ANALYZE_FEATURES not in names
    assert "resolve_place" in names


def test_missing_place_ref_blocks_query_osm_trust():
    err = untrusted_spatial_scope_error(
        {
            "place_ref_scope": {"place_ref": "place_1", "radius_m": 2000},
            "tags": [{"key": "leisure", "value": "park"}],
        },
        _COMPARE,
        places=PlaceRegistry(),
    )
    assert err is not None
    assert "place_ref" in err


def test_invented_coords_still_rejected():
    err = untrusted_spatial_scope_error(
        {
            "point": {"lat": 35.7, "lon": 51.4, "radius_m": 2000},
            "tags": [{"key": "leisure", "value": "park"}],
        },
        _COMPARE,
    )
    assert err is not None


def test_llm_timeout_status_not_overpass():
    from app.agent.contracts import GeoAgentResponse

    result = GeoAgentResponse(
        answer="The local model timed out during request planning. No live OSM query was executed.",
        errors=["llm_timeout: Ollama request timed out after 180.0s"],
        stop_reason="llm_error",
        live_query_failed=False,
    )
    assert derive_result_status(result) == "timed_out"
    assert result.live_query_failed is False


class _PlanLLM:
    model_name = "fake-model"
    provider_name = "fake"

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.seen_options: list[LLMOptions | None] = []
        self.seen_tools: list[list[ToolDefinition] | None] = []

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDefinition] | None = None,
        options: LLMOptions | None = None,
    ) -> LLMResponse:
        self.seen_options.append(options)
        self.seen_tools.append(tools)
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, LLMResponse)
        return item

    async def aclose(self) -> None:
        return None


def _park_element(osm_id: int, lon: float, lat: float) -> OverpassElement:
    return {
        "type": "node",
        "id": osm_id,
        "lat": lat,
        "lon": lon,
        "tags": {"leisure": "park", "name": f"Park {osm_id}"},
    }


class ScriptedOverpass:
    def __init__(self, batches: list[tuple[OverpassElement, ...]]) -> None:
        self._batches = list(batches)
        self.queries: list[str] = []

    @property
    def endpoint(self) -> str:
        return "https://overpass.test/api/interpreter"

    async def run(self, query: str) -> OverpassResponse:
        self.queries.append(query)
        elements = self._batches.pop(0) if self._batches else ()
        return OverpassResponse(
            elements=elements,
            query=query,
            endpoint=self.endpoint,
            retrieved_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            response_bytes=len(query),
            truncated=False,
        )

    async def aclose(self) -> None:
        return None


class PassthroughEncoder:
    def encode(self, elements: Sequence[OverpassElement]) -> GeoJsonConversionResult:
        features: list[dict[str, Any]] = []
        for element in elements:
            lon = float(element["lon"])
            lat = float(element["lat"])
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [lon, lat],
                                [lon + 0.0003, lat],
                                [lon + 0.0003, lat + 0.0003],
                                [lon, lat + 0.0003],
                                [lon, lat],
                            ]
                        ],
                    },
                    "properties": {
                        "osm_id": element["id"],
                        "osm_type": "way",
                        "tags": dict(element.get("tags") or {}),
                    },
                }
            )
        return GeoJsonConversionResult(
            feature_collection={"type": "FeatureCollection", "features": features},
            warnings=(),
        )


def _nominatim_handler(request: httpx.Request) -> httpx.Response:
    q = request.url.params.get("q", "").lower()
    if "sharif" in q:
        body = [
            {
                "lat": "35.7036",
                "lon": "51.3515",
                "display_name": "Sharif University of Technology, Tehran, Iran",
                "osm_id": 222,
                "class": "amenity",
                "type": "university",
                "importance": 0.8,
            },
            {
                "lat": "35.6892",
                "lon": "51.3890",
                "display_name": "Tehran City Hall, Tehran, Iran",
                "osm_id": 999,
                "class": "amenity",
                "type": "townhall",
                "importance": 0.7,
            },
        ]
    else:
        body = [
            {
                "lat": "35.7022",
                "lon": "51.3950",
                "display_name": "University of Tehran, Tehran, Iran",
                "osm_id": 111,
                "class": "amenity",
                "type": "university",
                "importance": 0.9,
            },
            {
                "lat": "35.6892",
                "lon": "51.3890",
                "display_name": "Tehran City Hall, Tehran, Iran",
                "osm_id": 999,
                "class": "amenity",
                "type": "townhall",
                "importance": 0.85,
            },
        ]
    return httpx.Response(200, json=body)


class FakeKnowledgeTool:
    name = "search_osm_knowledge"
    description = "docs"
    args_model = SearchOsmKnowledgeArgs

    async def execute(self, args: SearchOsmKnowledgeArgs, context: ToolContext):
        del context
        result = SearchOsmKnowledgeResult(
            query=args.query,
            passages=[
                RetrievedPassage(
                    content="Public parks are tagged leisure=park.",
                    document_title="Tag:leisure=park",
                    section="Description",
                    source_url="https://wiki.openstreetmap.org/wiki/Tag:leisure%3Dpark",
                    score=0.95,
                )
            ],
        )
        return ToolOutcome(
            observation="status=ok Documented OSM tags: leisure=park.",
            payload=result,
        )


@pytest.mark.asyncio
async def test_backend_controls_dependency_order_offline():
    transport = httpx.MockTransport(_nominatim_handler)
    client = httpx.AsyncClient(transport=transport)
    resolve = ResolvePlaceTool(
        NominatimPlaceResolver(
            base_url="https://nominatim.test/search",
            user_agent="test",
            timeout_seconds=5.0,
            client=client,
        )
    )
    overpass = ScriptedOverpass(
        [
            (_park_element(1, 51.396, 35.703), _park_element(2, 51.397, 35.704)),
            (_park_element(3, 51.352, 35.704),),
        ]
    )
    query = QueryOsmTool(overpass, PassthroughEncoder(), timeout_seconds=25, max_results=1000)
    analyze = build_analyze_features_tool()
    registry = ToolRegistry()
    registry.register(FakeKnowledgeTool())
    registry.register(resolve)
    registry.register(query)
    registry.register(analyze)

    plan_json = json.dumps(
        {
            "analysis_type": "comparison",
            "feature_concept": "public parks",
            "targets": [
                {
                    "label": "University of Tehran",
                    "place_query": "University of Tehran, Tehran, Iran",
                },
                {
                    "label": "Sharif University of Technology",
                    "place_query": "Sharif University of Technology, Tehran, Iran",
                },
            ],
            "radius_m": 2000,
            "comparison_goal": "park availability",
        }
    )
    llm = _PlanLLM(
        [
            LLMResponse(content=plan_json, model="fake"),
            LLMResponse(
                content=json.dumps(
                    {
                        "final_answer": (
                            "University of Tehran has 2 parks; Sharif has 1 (computed counts)."
                        )
                    }
                ),
                model="fake",
            ),
        ]
    )
    agent = PlannerExecutorAgent(llm, registry, LoopLimits(max_tool_rounds=6, max_tool_calls=12))
    try:
        result = await agent.run(GeoAgentRequest(message=_COMPARE))
    finally:
        await resolve.aclose()

    assert result.stop_reason == "final_answer"
    assert result.analysis is not None
    assert result.analysis.status == "completed"
    assert result.validated_tags == ["leisure=park"]
    results_only = [e.tool_name for e in result.trace if e.kind == "tool_result"]
    assert results_only[0] == "search_osm_knowledge"
    assert results_only[1:3] == ["resolve_place", "resolve_place"]
    assert results_only[3:5] == ["query_osm", "query_osm"]
    assert "analyze_features" not in results_only
    assert result.analysis.decision_trace.selected_indicator_id == "feature_count"
    assert llm.seen_tools[0] is None  # compact planner: no tools
    assert any(opt is not None and opt.json_mode for opt in llm.seen_options)
    assert "City Hall" not in result.answer
    assert result.geojson is not None
    assert result.geojson["features"]
    labels = {feature["properties"]["analysis_target"] for feature in result.geojson["features"]}
    assert labels == {"University of Tehran", "Sharif University of Technology"}
    indexes = {feature["properties"]["target_index"] for feature in result.geojson["features"]}
    assert indexes == {0, 1}
    assert all("out geom" in query for query in overpass.queries)
    assert '["leisure"="park"]["landuse"' not in overpass.queries[0]


@pytest.mark.asyncio
async def test_green_space_comparison_queries_union_tags_for_both_targets():
    transport = httpx.MockTransport(_nominatim_handler)
    client = httpx.AsyncClient(transport=transport)
    resolve = ResolvePlaceTool(
        NominatimPlaceResolver(
            base_url="https://nominatim.test/search",
            user_agent="test",
            timeout_seconds=5.0,
            client=client,
        )
    )
    overpass = ScriptedOverpass(
        [
            (_park_element(1, 51.396, 35.703), _park_element(2, 51.397, 35.704)),
            (_park_element(3, 51.352, 35.704),),
        ]
    )
    query = QueryOsmTool(overpass, PassthroughEncoder(), timeout_seconds=25, max_results=1000)
    analyze = build_analyze_features_tool()
    registry = ToolRegistry()
    registry.register(FakeKnowledgeTool())
    registry.register(resolve)
    registry.register(query)
    registry.register(analyze)

    plan_json = json.dumps(
        {
            "analysis_type": "comparison",
            "feature_concept": "green spaces",
            "targets": [
                {
                    "label": "University of Tehran",
                    "place_query": "University of Tehran, Tehran, Iran",
                },
                {
                    "label": "Sharif University of Technology",
                    "place_query": "Sharif University of Technology, Tehran, Iran",
                },
            ],
            "radius_m": 2000,
            "comparison_goal": "green-space provision",
        }
    )
    llm = _PlanLLM(
        [
            LLMResponse(content=plan_json, model="fake"),
            LLMResponse(
                content=json.dumps(
                    {
                        "final_answer": (
                            "University of Tehran has a higher mapped green-space ratio "
                            "than Sharif (computed coverage)."
                        )
                    }
                ),
                model="fake",
            ),
        ]
    )
    agent = PlannerExecutorAgent(llm, registry, LoopLimits(max_tool_rounds=6, max_tool_calls=12))
    try:
        result = await agent.run(GeoAgentRequest(message=_COMPARE_GREEN))
    finally:
        await resolve.aclose()

    assert result.stop_reason == "final_answer"
    assert result.analysis is not None
    catalog_green = ("leisure=park", "leisure=garden", "landuse=grass", "natural=wood")
    assert result.validated_tags == list(catalog_green)
    assert result.analysis.decision_trace.selected_indicator_id == "green_space_ratio"
    assert result.geojson is not None
    labels = {feature["properties"]["analysis_target"] for feature in result.geojson["features"]}
    assert labels == {"University of Tehran", "Sharif University of Technology"}
    assert {feature["properties"]["target_index"] for feature in result.geojson["features"]} == {
        0,
        1,
    }
    assert len(overpass.queries) == 2
    for built in overpass.queries:
        assert "out geom" in built
        for tag in catalog_green:
            key, value = tag.split("=", 1)
            assert f'["{key}"="{value}"]' in built
        assert '["leisure"="park"]["leisure"="garden"]' not in built


@pytest.mark.asyncio
async def test_llm_timeout_on_comparison_planner():
    registry = ToolRegistry()
    registry.register(FakeKnowledgeTool())
    llm = _PlanLLM([LLMTimeoutError("Ollama request timed out after 180.0s")])
    agent = PlannerExecutorAgent(llm, registry, LoopLimits(max_tool_rounds=4, max_tool_calls=8))
    result = await agent.run(GeoAgentRequest(message=_COMPARE))
    assert result.stop_reason == "llm_error"
    assert any("llm_timeout" in err for err in result.errors)
    assert result.live_query_failed is False
    assert result.geojson is None
    assert "No live OSM query was executed" in result.answer


@pytest.mark.asyncio
async def test_unresolved_place_blocks_comparison_queries():
    async def empty_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(empty_handler)
    client = httpx.AsyncClient(transport=transport)
    resolve = ResolvePlaceTool(
        NominatimPlaceResolver(
            base_url="https://nominatim.test/search",
            user_agent="test",
            timeout_seconds=5.0,
            client=client,
        )
    )
    registry = ToolRegistry()
    registry.register(FakeKnowledgeTool())
    registry.register(resolve)
    registry.register(
        QueryOsmTool(
            ScriptedOverpass([]),
            PassthroughEncoder(),
            timeout_seconds=25,
            max_results=100,
        )
    )
    plan_json = json.dumps(
        {
            "analysis_type": "comparison",
            "feature_concept": "public parks",
            "targets": [
                {
                    "label": "University of Tehran",
                    "place_query": "University of Tehran, Tehran, Iran",
                },
                {
                    "label": "Sharif University of Technology",
                    "place_query": "Sharif University of Technology, Tehran, Iran",
                },
            ],
            "radius_m": 2000,
            "comparison_goal": "park availability",
        }
    )
    llm = _PlanLLM([LLMResponse(content=plan_json, model="fake")])
    agent = PlannerExecutorAgent(llm, registry, LoopLimits(max_tool_rounds=4, max_tool_calls=8))
    try:
        result = await agent.run(GeoAgentRequest(message=_COMPARE))
    finally:
        await resolve.aclose()
    assert result.geojson is None
    assert any("place_resolution" in err for err in result.errors)
    assert "could not be resolved" in result.answer.lower()
    assert not any(e.tool_name == QUERY_OSM for e in result.trace if e.kind == "tool_result")
