"""Offline multi-target landmark comparison workflow."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from app.agent.accumulator import ResultAccumulator
from app.agent.contracts import GeoAgentRequest
from app.agent.geojson_combine import combine_target_feature_collections
from app.agent.loop import LoopLimits
from app.agent.orchestrator import PlannerExecutorAgent, _model_facing_tools
from app.agent.spatial_trust import untrusted_spatial_scope_error
from app.analytics.datasets import DatasetRegistry
from app.analytics.factory import build_analyze_features_tool
from app.core.errors import PlaceResolutionError, ToolArgumentError
from app.llm.contracts import ToolCall, ToolDefinition
from app.osm.contracts import (
    GeoJsonConversionResult,
    OverpassElement,
    OverpassResponse,
)
from app.osm.query_spec import OsmFeatureQuery, PlaceRefScope, TagFilter
from app.places.contracts import GeocoderHit, PlaceRegistry
from app.places.nominatim import NominatimPlaceResolver
from app.places.selection import select_trusted_hit
from app.rag.contracts import RetrievedPassage
from app.tools.context import AnalysisRunState, GroundingState, ToolContext
from app.tools.contracts import ToolOutcome
from app.tools.query_osm import QueryOsmTool
from app.tools.registry import ToolRegistry
from app.tools.resolve_place import PlaceResolutionRequest, ResolvePlaceTool
from app.tools.search_osm_knowledge import SearchOsmKnowledgeArgs, SearchOsmKnowledgeResult
from pydantic import ValidationError

from tests.test_orchestrator import ScriptedLLM, _final, _tool_response

_PASSAGE = RetrievedPassage(
    content="Public parks are tagged leisure=park.",
    document_title="Tag:leisure=park",
    section="Description",
    source_url="https://wiki.openstreetmap.org/wiki/Tag:leisure%3Dpark",
    score=0.95,
)

_COMPARE_MSG = (
    "Compare public parks within 2 km of the University of Tehran and Sharif "
    "University of Technology. Choose the best comparison metric, generate a "
    "comparison report, and return the park features for both areas on the map "
    "and as GeoJSON."
)


def _ctx(
    message: str = _COMPARE_MSG,
    *,
    places: PlaceRegistry | None = None,
    grounding: GroundingState | None = None,
) -> ToolContext:
    return ToolContext(
        datasets=DatasetRegistry(),
        analysis=AnalysisRunState(),
        user_message=message,
        places=places or PlaceRegistry(),
        grounding=grounding or GroundingState(),
    )


def _park_element(osm_id: int, lon: float, lat: float) -> OverpassElement:
    return {
        "type": "node",
        "id": osm_id,
        "lat": lat,
        "lon": lon,
        "tags": {"leisure": "park", "name": f"Park {osm_id}"},
    }


class FakeKnowledgeTool:
    name = "search_osm_knowledge"
    description = "docs"
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
            observation=(
                "status=ok source=osm_documentation Documented OSM tags mentioned "
                "in titles: leisure=park. For public parks, prefer leisure=park."
            ),
            payload=result,
        )


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
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(element["lon"]), float(element["lat"])],
                    },
                    "properties": {
                        "osm_id": element["id"],
                        "osm_type": element["type"],
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
            }
        ]
    elif "amirkabir" in q:
        body = [
            {
                "lat": "35.7013",
                "lon": "51.3906",
                "display_name": "Amirkabir University of Technology, Tehran, Iran",
                "osm_id": 444,
                "class": "amenity",
                "type": "university",
                "importance": 0.8,
            }
        ]
    elif "azadi" in q:
        body = [
            {
                "lat": "35.6997",
                "lon": "51.3381",
                "display_name": "Azadi Square, Tehran, Iran",
                "osm_id": 333,
                "class": "place",
                "type": "square",
                "importance": 0.7,
            }
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
            }
        ]
    return httpx.Response(200, json=body)


def _resolve_tool() -> ResolvePlaceTool:
    transport = httpx.MockTransport(_nominatim_handler)
    client = httpx.AsyncClient(transport=transport)
    resolver = NominatimPlaceResolver(
        base_url="https://nominatim.test/search",
        user_agent="OSM-GeoAgent-test",
        timeout_seconds=5.0,
        client=client,
    )
    return ResolvePlaceTool(resolver)


def _query_tool(batches: list[tuple[OverpassElement, ...]]) -> QueryOsmTool:
    return QueryOsmTool(
        ScriptedOverpass(batches),
        PassthroughEncoder(),
        timeout_seconds=25,
        max_results=1000,
    )


def test_place_list_rejected_by_schema():
    with pytest.raises(ValidationError, match="single named area"):
        OsmFeatureQuery.model_validate(
            {
                "place": ["University of Tehran", "Sharif University of Technology"],
                "tags": [{"key": "leisure", "value": "park"}],
                "limit": 20,
            }
        )


def test_buffer_field_forbidden_on_query_osm():
    with pytest.raises(ValidationError):
        OsmFeatureQuery.model_validate(
            {
                "place": "Tehran, Iran",
                "buffer": 2000,
                "tags": [{"key": "leisure", "value": "park"}],
            }
        )


def test_invented_university_coordinates_rejected():
    err = untrusted_spatial_scope_error(
        {
            "point": {"lat": 35.7022, "lon": 51.3950, "radius_m": 2000},
            "tags": [{"key": "leisure", "value": "park"}],
        },
        _COMPARE_MSG,
    )
    assert err is not None


def test_user_provided_coordinates_still_accepted():
    message = "Find parks within 2000 m of 35.7022, 51.3950"
    assert (
        untrusted_spatial_scope_error(
            {
                "point": {"lat": 35.7022, "lon": 51.3950, "radius_m": 2000},
                "tags": [{"key": "leisure", "value": "park"}],
            },
            message,
        )
        is None
    )


def test_place_ref_scope_trusted_after_resolve():
    places = PlaceRegistry()
    places.register(
        query="University of Tehran, Tehran, Iran",
        label="University of Tehran",
        display_name="University of Tehran, Tehran, Iran",
        latitude=35.7022,
        longitude=51.3950,
        source="nominatim",
        source_id="111",
    )
    assert (
        untrusted_spatial_scope_error(
            {
                "place_ref_scope": {"place_ref": "place_1", "radius_m": 2000},
                "tags": [{"key": "leisure", "value": "park"}],
            },
            _COMPARE_MSG,
            places=places,
        )
        is None
    )


def test_unknown_place_ref_rejected():
    err = untrusted_spatial_scope_error(
        {
            "place_ref_scope": {"place_ref": "place_9", "radius_m": 2000},
            "tags": [{"key": "leisure", "value": "park"}],
        },
        _COMPARE_MSG,
        places=PlaceRegistry(),
    )
    assert err is not None
    assert "place_ref" in err


def test_radius_must_match_user_request():
    places = PlaceRegistry()
    places.register(
        query="x",
        label="x",
        display_name="x",
        latitude=35.7,
        longitude=51.4,
        source="nominatim",
        source_id="1",
    )
    err = untrusted_spatial_scope_error(
        {
            "place_ref_scope": {"place_ref": "place_1", "radius_m": 5000},
            "tags": [{"key": "leisure", "value": "park"}],
        },
        _COMPARE_MSG,
        places=places,
    )
    assert err is not None
    assert "radius_m" in err


@pytest.mark.asyncio
async def test_resolve_place_returns_trusted_coords_offline():
    tool = _resolve_tool()
    ctx = _ctx()
    try:
        outcome = await tool.execute(
            PlaceResolutionRequest(query="University of Tehran, Tehran, Iran"),
            ctx,
        )
    finally:
        await tool.aclose()
    assert outcome.payload.place_ref == "place_1"
    assert outcome.payload.latitude == pytest.approx(35.7022)
    assert ctx.places.has("place_1")
    assert "place_ref=place_1" in outcome.observation


@pytest.mark.asyncio
async def test_query_osm_place_ref_equal_radius_and_grounding():
    places = PlaceRegistry()
    places.register(
        query="University of Tehran, Tehran, Iran",
        label="University of Tehran",
        display_name="University of Tehran, Tehran, Iran",
        latitude=35.7022,
        longitude=51.3950,
        source="nominatim",
        source_id="111",
    )
    places.register(
        query="Sharif University of Technology, Tehran, Iran",
        label="Sharif University of Technology",
        display_name="Sharif University of Technology, Tehran, Iran",
        latitude=35.7036,
        longitude=51.3515,
        source="nominatim",
        source_id="222",
    )
    grounding = GroundingState(tags=["leisure=park"])
    ctx = _ctx(places=places, grounding=grounding)
    client = ScriptedOverpass(
        [
            (_park_element(1, 51.396, 35.703), _park_element(2, 51.397, 35.704)),
            (_park_element(3, 51.352, 35.704),),
        ]
    )
    tool = QueryOsmTool(client, PassthroughEncoder(), timeout_seconds=25, max_results=1000)
    first = await tool.execute(
        OsmFeatureQuery(
            place_ref_scope=PlaceRefScope(place_ref="place_1", radius_m=2000),
            tags=[TagFilter(key="leisure", value="park")],
            limit=50,
        ),
        ctx,
    )
    second = await tool.execute(
        OsmFeatureQuery(
            place_ref_scope=PlaceRefScope(place_ref="place_2", radius_m=2000),
            tags=[TagFilter(key="leisure", value="park")],
            limit=50,
        ),
        ctx,
    )
    assert first.payload.analysis_target == "University of Tehran"
    assert second.payload.analysis_target == "Sharif University of Technology"
    assert ctx.datasets.refs() == ("osm_result_1", "osm_result_2")
    assert "around:2000,35.7022,51.395" in client.queries[0]
    assert "around:2000,35.7036,51.3515" in client.queries[1]
    with pytest.raises(ToolArgumentError, match="grounded"):
        await tool.execute(
            OsmFeatureQuery(
                place_ref_scope=PlaceRefScope(place_ref="place_1", radius_m=2000),
                tags=[TagFilter(key="amenity", value="public_park")],
                limit=50,
            ),
            ctx,
        )


def test_full_comparison_uses_backend_runner_path():
    """Landmark comparisons use MultiTargetComparisonRunner.

    End-to-end offline coverage is in tests/test_comparison_orchestration.py.
    """
    from app.agent.comparison_executor import should_use_multi_target_runner

    assert should_use_multi_target_runner(_COMPARE_MSG)


@pytest.mark.asyncio
async def test_ineligible_analyze_features_rejected():
    analyze = build_analyze_features_tool()
    registry = ToolRegistry()
    registry.register(FakeKnowledgeTool())
    registry.register(analyze)
    llm = ScriptedLLM(
        [
            _tool_response(
                ToolCall(
                    id="1",
                    name="analyze_features",
                    arguments={
                        "analysis_type": "comparison",
                        "feature_concept": "parks",
                        "comparison_goal": "more parks",
                        "targets": [
                            {
                                "target_id": "a",
                                "label": "A",
                                "dataset_ref": "osm_result_1",
                            }
                        ],
                        "metrics": [
                            {
                                "metric": "count",
                                "role": "primary",
                                "inferred_goal": "abundance",
                            }
                        ],
                    },
                )
            ),
            _final("stopped"),
        ]
    )
    agent = PlannerExecutorAgent(llm, registry, LoopLimits(max_tool_rounds=4, max_tool_calls=8))
    # Stay on the general planner path (not multi-target landmark runner).
    result = await agent.run(GeoAgentRequest(message="Which area has more parks?"))
    assert any("tool_not_eligible" in warning for warning in result.warnings)
    assert result.analysis is None


@pytest.mark.asyncio
async def test_future_dataset_ref_rejected():
    resolve = _resolve_tool()
    client = ScriptedOverpass([(_park_element(1, 51.396, 35.703),)])
    query = QueryOsmTool(client, PassthroughEncoder(), timeout_seconds=25, max_results=1000)
    analyze = build_analyze_features_tool()
    registry = ToolRegistry()
    registry.register(resolve)
    registry.register(query)
    registry.register(analyze)
    llm = ScriptedLLM(
        [
            _tool_response(
                ToolCall(
                    id="0",
                    name="resolve_place",
                    arguments={"query": "Azadi Square, Tehran, Iran"},
                )
            ),
            _tool_response(
                ToolCall(
                    id="1",
                    name="query_osm",
                    arguments={
                        "place_ref_scope": {"place_ref": "place_1", "radius_m": 2000},
                        "tags": [{"key": "leisure", "value": "park"}],
                        "limit": 20,
                    },
                )
            ),
            _tool_response(
                ToolCall(
                    id="2",
                    name="analyze_features",
                    arguments={
                        "analysis_type": "comparison",
                        "feature_concept": "parks",
                        "comparison_goal": "more parks",
                        "targets": [
                            {
                                "target_id": "a",
                                "label": "A",
                                "dataset_ref": "osm_result_1",
                            },
                            {
                                "target_id": "b",
                                "label": "B",
                                "dataset_ref": "osm_result_2",
                            },
                        ],
                        "metrics": [
                            {
                                "metric": "count",
                                "role": "primary",
                                "inferred_goal": "abundance",
                            }
                        ],
                    },
                )
            ),
            _final("done"),
        ]
    )
    agent = PlannerExecutorAgent(llm, registry, LoopLimits(max_tool_rounds=6, max_tool_calls=12))
    try:
        result = await agent.run(
            GeoAgentRequest(message="Which area has more parks within 2000 m of Azadi Square?")
        )
    finally:
        await resolve.aclose()
    assert any("osm_result_2" in warning for warning in result.warnings)
    assert result.analysis is None


def test_single_collection_omits_target_index():
    collection: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [51.4, 35.7]},
                "properties": {"osm_id": 1, "osm_type": "node"},
            }
        ],
    }
    combined = combine_target_feature_collections([("Tehran", collection)])
    props = combined["features"][0]["properties"]
    assert "target_index" not in props
    assert props["analysis_target"] == "Tehran"
    assert props["analysis_target_label"] == "Tehran"


def test_overlapping_features_are_kept_per_target():
    ring = [
        [51.37, 35.70],
        [51.38, 35.70],
        [51.38, 35.71],
        [51.37, 35.71],
        [51.37, 35.70],
    ]
    shared = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {"osm_id": 99, "osm_type": "way", "tags": {"leisure": "park"}},
    }
    a: dict[str, Any] = {"type": "FeatureCollection", "features": [shared]}
    b: dict[str, Any] = {"type": "FeatureCollection", "features": [shared]}
    combined = combine_target_feature_collections(
        [("University of Tehran", a), ("Sharif University of Technology", b)]
    )
    assert len(combined["features"]) == 2
    labels = {feature["properties"]["analysis_target"] for feature in combined["features"]}
    assert labels == {"University of Tehran", "Sharif University of Technology"}
    indexes = {feature["properties"]["target_index"] for feature in combined["features"]}
    assert indexes == {0, 1}
    assert combined["features"][0]["geometry"]["type"] == "Polygon"


def test_analyze_hidden_until_datasets():
    defs = [
        ToolDefinition(name="resolve_place", description="d", parameters_schema={}),
        ToolDefinition(name="query_osm", description="d", parameters_schema={}),
        ToolDefinition(name="analyze_features", description="d", parameters_schema={}),
    ]
    state = ResultAccumulator()
    names = [
        item.name for item in _model_facing_tools(defs, state=state, user_message=_COMPARE_MSG)
    ]
    assert "analyze_features" not in names


def test_tehran_window_selection_rejects_far_hits():
    with pytest.raises(PlaceResolutionError):
        select_trusted_hit(
            "University of Tehran, Tehran, Iran",
            (
                GeocoderHit(
                    display_name="Somewhere else",
                    latitude=40.0,
                    longitude=40.0,
                    source_id="1",
                ),
            ),
        )


@pytest.mark.asyncio
async def test_simple_tehran_query_still_works_without_resolve():
    client = ScriptedOverpass([(_park_element(1, 51.4, 35.7),)])
    query = QueryOsmTool(client, PassthroughEncoder(), timeout_seconds=25, max_results=1000)
    registry = ToolRegistry()
    registry.register(query)
    llm = ScriptedLLM(
        [
            _tool_response(
                ToolCall(
                    id="1",
                    name="query_osm",
                    arguments={
                        "place": "Tehran, Iran",
                        "tags": [{"key": "leisure", "value": "park"}],
                        "limit": 20,
                    },
                )
            ),
            _final("Found parks in Tehran."),
        ]
    )
    agent = PlannerExecutorAgent(llm, registry, LoopLimits(max_tool_rounds=4, max_tool_calls=8))
    result = await agent.run(
        GeoAgentRequest(message="Find public parks in Tehran, Iran. Return at most 20 features.")
    )
    assert result.feature_count == 1
    assert result.geojson is not None
    assert result.analysis is None
