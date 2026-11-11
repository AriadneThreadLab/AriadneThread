"""Solar decision routing: slot filling, adaptive questions, persistence."""

from __future__ import annotations

import json
from typing import Any

import pytest
from app.agent.accumulator import ResultAccumulator
from app.agent.comparison_workflow import is_indicator_analysis_request
from app.agent.contracts import GeoAgentRequest
from app.agent.loop import LoopLimits
from app.agent.orchestrator import PlannerExecutorAgent, _model_facing_tools
from app.agent.solar_intent import (
    detect_decision_mode,
    detect_site_selection_scope,
    is_solar_request,
)
from app.analytics.datasets import DatasetRegistry
from app.llm.contracts import ToolCall, ToolDefinition
from app.places.contracts import PlaceRegistry
from app.solar.contracts import (
    PROVIDER_FILLED_FIELDS,
    SolarMissingRequirement,
)
from app.solar.questions import GOAL_PROMPT, questions_for_missing
from app.solar.slots import extract_solar_slots
from app.solar.store import InMemorySolarDecisionStore
from app.tools.context import AnalysisRunState, GroundingState, ToolContext
from app.tools.factory import build_tool_registry
from app.tools.registry import ToolRegistry
from app.tools.solar_tools import (
    TOOL_NAME as ANALYZE_SOLAR_SITE,
)
from app.tools.solar_tools import (
    AnalyzeSolarSiteArgs,
    AnalyzeSolarSiteTool,
)
from pydantic import ValidationError

from tests.test_orchestrator import ScriptedLLM, _final, _tool_response

VIENNA_FARM = "Find good solar farm locations in Vienna"
AOI_SEARCH = "Find the best location inside this polygon"
RANK_ROOFS = "Rank these buildings for rooftop PV"
OWN_HOUSE = "I own this house and want solar panels"

_PROVIDER_WORDS = (
    "ghi",
    "slope",
    "elevation",
    "irradiation",
    "aspect",
    "distance to roads",
    "distance to grid",
    "land cover",
    "temperature",
)


def _ctx(
    message: str,
    *,
    conversation_id: str | None = "conv-solar",
    store: InMemorySolarDecisionStore | None = None,
) -> ToolContext:
    return ToolContext(
        datasets=DatasetRegistry(),
        analysis=AnalysisRunState(),
        user_message=message,
        places=PlaceRegistry(),
        grounding=GroundingState(),
        conversation_id=conversation_id,
        solar_store=store if store is not None else InMemorySolarDecisionStore(),
    )


def _registry(store: InMemorySolarDecisionStore | None = None) -> ToolRegistry:
    return build_tool_registry(analyze_solar_site_tool=AnalyzeSolarSiteTool(store=store))


async def _route(
    message: str,
    *,
    args: dict[str, Any] | None = None,
    store: InMemorySolarDecisionStore | None = None,
    conversation_id: str = "conv-solar",
    backend: Any | None = None,
):
    resolved = store if store is not None else InMemorySolarDecisionStore()
    tool = AnalyzeSolarSiteTool(store=resolved, backend=backend)
    context = _ctx(message, conversation_id=conversation_id, store=resolved)
    validated = AnalyzeSolarSiteArgs.model_validate(args or {})
    return await tool.execute(validated, context)


def test_four_documented_routes() -> None:
    assert detect_decision_mode(VIENNA_FARM) == "site_selection"
    assert detect_site_selection_scope(VIENNA_FARM) == "region_search"
    assert detect_decision_mode(AOI_SEARCH) == "site_selection"
    assert detect_site_selection_scope(AOI_SEARCH) == "aoi_search"
    assert detect_decision_mode(RANK_ROOFS) == "site_selection"
    assert detect_site_selection_scope(RANK_ROOFS) == "candidate_ranking"
    assert detect_decision_mode(OWN_HOUSE) == "fixed_site"
    assert detect_site_selection_scope(OWN_HOUSE) is None


def test_slot_filling_from_natural_language() -> None:
    vienna = extract_solar_slots(VIENNA_FARM)
    assert vienna.mode == "site_selection"
    assert vienna.site_selection_scope == "region_search"
    assert vienna.project_type == "utility_scale"
    assert vienna.region is not None
    assert vienna.region.name == "Vienna"

    house = extract_solar_slots(OWN_HOUSE)
    assert house.mode == "fixed_site"
    assert house.project_type == "residential_rooftop"
    assert house.location is not None
    assert house.location.name is not None
    assert "house" in house.location.name.lower()

    ranked = extract_solar_slots(RANK_ROOFS)
    assert ranked.mode == "site_selection"
    assert ranked.site_selection_scope == "candidate_ranking"
    assert ranked.project_type == "rooftop"
    assert ranked.candidate_ids == ()

    aoi = extract_solar_slots(AOI_SEARCH)
    assert aoi.site_selection_scope == "aoi_search"
    assert aoi.aoi is not None


@pytest.mark.asyncio
async def test_vienna_farm_does_not_ask_redundant_or_provider_questions() -> None:
    outcome = await _route(VIENNA_FARM)
    payload = outcome.payload
    assert payload.status == "needs_input"
    assert payload.mode == "site_selection"
    assert payload.site_selection_scope == "region_search"
    assert payload.project_type == "utility_scale"
    assert payload.capability == "solar_site_selection"
    assert payload.problem["region"]["name"] == "Vienna"
    fields = [item.field for item in payload.questions]
    assert "bbox" not in fields
    assert "generation_strategy" not in fields
    assert "cell_size" not in fields
    assert "constraint_profile" not in fields
    assert "ranking_profile" in fields
    assert payload.problem.get("location") is None
    blob = json.dumps(payload.model_dump()).lower() + outcome.observation.lower()
    for word in _PROVIDER_WORDS:
        assert word not in " ".join(item.prompt.lower() for item in payload.questions)
        assert f"ask only: {word}" not in blob
    assert "science" in outcome.observation
    assert "not_computed" in outcome.observation


@pytest.mark.asyncio
async def test_house_request_is_fixed_site_without_asking_region() -> None:
    outcome = await _route(OWN_HOUSE)
    payload = outcome.payload
    assert payload.mode == "fixed_site"
    assert payload.capability == "solar_fixed_site_analysis"
    assert payload.status == "needs_input"
    fields = [item.field for item in payload.questions]
    assert "site_point" not in fields
    assert "constraint_profile" not in fields
    assert "target_capacity_kw" in fields
    assert payload.problem.get("region") is None
    assert payload.problem.get("aoi") is None
    prompts = [item.prompt.lower() for item in payload.questions]
    assert not any("region" in prompt or "aoi" in prompt for prompt in prompts)


@pytest.mark.asyncio
async def test_ranking_without_ids_asks_only_for_candidates() -> None:
    outcome = await _route(RANK_ROOFS)
    payload = outcome.payload
    assert payload.status == "needs_input"
    assert payload.site_selection_scope == "candidate_ranking"
    assert payload.capability == "solar_candidate_ranking"
    fields = [item.field for item in payload.questions]
    assert "candidates" in fields
    assert "ranking_profile" in fields
    assert "bbox" not in fields
    assert "Rooftop or ground-mounted?" not in [item.prompt for item in payload.questions]
    assert GOAL_PROMPT not in [item.prompt for item in payload.questions]


@pytest.mark.asyncio
async def test_incomplete_request_asks_only_required_slots() -> None:
    outcome = await _route("I want solar PV")
    payload = outcome.payload
    assert payload.status == "needs_input"
    prompts = [item.prompt for item in payload.questions]
    assert prompts == [GOAL_PROMPT]
    assert "What is your target capacity?" not in prompts
    assert "Do you have a budget constraint?" not in prompts
    assert "Is battery storage required?" not in prompts
    assert all("tilt" not in prompt.lower() for prompt in prompts)
    assert all("azimuth" not in prompt.lower() for prompt in prompts)
    for word in _PROVIDER_WORDS:
        assert all(word not in prompt.lower() for prompt in prompts)


@pytest.mark.asyncio
async def test_planner_cannot_skip_goal_with_invented_fixed_site() -> None:
    outcome = await _route(
        "I want solar panels.",
        args={"mode": "fixed_site", "capability_id": "solar_fixed_site_analysis"},
    )
    payload = outcome.payload
    assert payload.mode is None
    assert [item.field for item in payload.questions] == ["mode"]
    assert payload.questions[0].prompt == GOAL_PROMPT
    assert "constraint_profile" not in [item.field for item in payload.questions]
    assert "site_point" not in [item.field for item in payload.questions]


@pytest.mark.asyncio
async def test_goal_follow_up_collects_existing_building_parameters() -> None:
    store = InMemorySolarDecisionStore()
    first = await _route("I want solar panels.", store=store, conversation_id="goal-1")
    assert [item.field for item in first.payload.questions] == ["mode"]
    follow = await _route(
        "existing building evaluation",
        store=store,
        conversation_id="goal-1",
    )
    payload = follow.payload
    assert payload.mode == "fixed_site"
    fields = [item.field for item in payload.questions]
    assert "location" in fields
    assert "project_type" in fields
    assert "target_capacity_kw" in fields
    assert "constraint_profile" not in fields
    assert "tilt_deg" not in fields
    prompts = [item.prompt.lower() for item in payload.questions]
    assert all("tilt" not in prompt for prompt in prompts)
    assert all("azimuth" not in prompt for prompt in prompts)
    assert all("loss" not in prompt for prompt in prompts)


@pytest.mark.asyncio
async def test_goal_follow_up_collects_site_selection_parameters() -> None:
    store = InMemorySolarDecisionStore()
    await _route("I want solar panels.", store=store, conversation_id="goal-2")
    follow = await _route("2. Site selection", store=store, conversation_id="goal-2")
    payload = follow.payload
    assert payload.mode == "site_selection"
    assert payload.site_selection_scope == "region_search"
    fields = [item.field for item in payload.questions]
    assert "region" in fields
    assert "ranking_profile" in fields
    assert "bbox" not in fields
    assert "generation_strategy" not in fields
    assert "cell_size" not in fields


@pytest.mark.asyncio
async def test_goal_follow_up_collects_candidate_ranking_parameters() -> None:
    store = InMemorySolarDecisionStore()
    await _route("I want solar panels.", store=store, conversation_id="goal-3")
    follow = await _route("candidate ranking", store=store, conversation_id="goal-3")
    payload = follow.payload
    assert payload.site_selection_scope == "candidate_ranking"
    fields = [item.field for item in payload.questions]
    assert "candidates" in fields
    assert "ranking_profile" in fields
    assert "location" not in fields
    assert "bbox" not in fields


@pytest.mark.asyncio
async def test_never_asks_provider_filled_layers_even_if_reported_missing() -> None:
    draft = extract_solar_slots(VIENNA_FARM)
    fake_missing = (
        SolarMissingRequirement(field="ghi", required=True, reason="should never be asked"),
        SolarMissingRequirement(field="slope", required=True, reason="should never be asked"),
        SolarMissingRequirement(field="elevation", required=True, reason="should never be asked"),
        SolarMissingRequirement(
            field="distance_to_roads", required=True, reason="should never be asked"
        ),
        SolarMissingRequirement(
            field="distance_to_grid", required=True, reason="should never be asked"
        ),
        SolarMissingRequirement(field="land_cover", required=True, reason="should never be asked"),
        SolarMissingRequirement(field="temperature", required=True, reason="should never be asked"),
        SolarMissingRequirement(field="region", required=True, reason="already filled"),
    )
    questions = questions_for_missing(fake_missing, draft=draft)
    fields = {item.field for item in questions}
    assert not fields.intersection(PROVIDER_FILLED_FIELDS)
    assert "ghi" not in fields
    assert "slope" not in fields
    assert "elevation" not in fields


@pytest.mark.asyncio
async def test_yield_language_does_not_ask_tilt_azimuth_or_loss() -> None:
    outcome = await _route("I want solar panels. Estimate annual generation for a rooftop site.")
    fields = [item.field for item in outcome.payload.questions]
    prompts = [item.prompt.lower() for item in outcome.payload.questions]
    assert "tilt_deg" not in fields
    assert "azimuth_deg" not in fields
    assert "system_loss_percent" not in fields
    assert "mounting" not in fields
    assert fields == ["mode"]
    assert all("ghi" not in prompt for prompt in prompts)
    assert all("slope" not in prompt for prompt in prompts)
    assert all("irradiation" not in prompt for prompt in prompts)


@pytest.mark.asyncio
async def test_conversation_persists_slots_and_skips_repeat_questions() -> None:
    store = InMemorySolarDecisionStore()
    first = await _route(VIENNA_FARM, store=store, conversation_id="c1")
    assert first.payload.status == "needs_input"
    persisted = store.load("c1")
    assert persisted is not None
    assert persisted.region is not None
    assert persisted.region.name == "Vienna"

    follow = await _route(
        "The budget is 250000 and battery storage is required",
        store=store,
        conversation_id="c1",
        args={"budget_amount": 250000, "battery_required": True},
    )
    problem = follow.payload.problem
    assert problem["region"]["name"] == "Vienna"
    assert problem["budget"]["value"] == 250000
    assert problem["battery_required"] is True
    fields = [item.field for item in follow.payload.questions]
    assert "bbox" not in fields
    assert "generation_strategy" not in fields
    assert "cell_size" not in fields
    assert "What region/AOI should be searched?" not in [
        item.prompt for item in follow.payload.questions
    ]
    assert "What area should be searched?" not in [item.prompt for item in follow.payload.questions]


def test_tool_forbids_indicator_fields() -> None:
    with pytest.raises(ValidationError):
        AnalyzeSolarSiteArgs.model_validate({"ghi": 1800, "mode": "fixed_site"})
    with pytest.raises(ValidationError):
        AnalyzeSolarSiteArgs.model_validate({"slope": 5})


def test_solar_intent_is_not_an_osm_comparison() -> None:
    assert is_solar_request(VIENNA_FARM) is True
    assert is_indicator_analysis_request(VIENNA_FARM) is False
    assert is_solar_request("Compare parks within 2 km of University of Tehran") is False


def test_model_facing_tools_for_solar_hide_osm() -> None:
    registry = _registry()
    defs = [
        ToolDefinition(name="search_osm_knowledge", description="d", parameters_schema={}),
        ToolDefinition(name="search_solar_knowledge", description="d", parameters_schema={}),
        ToolDefinition(name="query_osm", description="d", parameters_schema={}),
        ToolDefinition(name="analyze_features", description="d", parameters_schema={}),
        ToolDefinition(name="simbench_query", description="d", parameters_schema={}),
        *registry.definitions(),
    ]
    names = [
        item.name
        for item in _model_facing_tools(
            defs,
            state=ResultAccumulator(),
            user_message=VIENNA_FARM,
            places=PlaceRegistry(),
        )
    ]
    assert names == ["search_solar_knowledge", ANALYZE_SOLAR_SITE]
    assert "query_osm" not in names


@pytest.mark.asyncio
async def test_orchestrator_routes_slots_without_computing_indicators() -> None:
    store = InMemorySolarDecisionStore()
    tool = AnalyzeSolarSiteTool(store=store)
    registry = ToolRegistry()
    registry.register(tool)
    llm = ScriptedLLM(
        [
            _tool_response(
                ToolCall(
                    id="1",
                    name=ANALYZE_SOLAR_SITE,
                    arguments={
                        "mode": "site_selection",
                        "site_selection_scope": "region_search",
                        "project_type": "utility_scale",
                        "region_name": "Vienna",
                        "capability_id": "solar_site_selection",
                    },
                )
            ),
            _final("Vienna is routed for solar farm site selection. No GHI was calculated."),
        ]
    )
    agent = PlannerExecutorAgent(
        llm,  # type: ignore[arg-type]
        registry,
        LoopLimits(max_tool_rounds=4, max_tool_calls=8),
        solar_store=store,
    )
    result = await agent.run(GeoAgentRequest(message=VIENNA_FARM, conversation_id="orch-1"))
    assert result.solar_decision is not None
    assert result.solar_decision.status == "needs_input"
    assert result.solar_decision.capability_id == "solar_site_selection"
    assert result.geojson is None
    observation = "\n".join(
        message.content for turn in llm.seen for message in turn if message.role == "tool"
    )
    assert "not_computed" in observation
    for word in ("ghi", "slope", "elevation"):
        assert word not in " ".join(q.prompt.lower() for q in result.solar_decision.questions)


def test_llm_cannot_force_unregistered_capability() -> None:
    with pytest.raises(ValidationError):
        AnalyzeSolarSiteArgs.model_validate({"capability_id": "invented_ghi_calculator"})


SITE_SELECTION_COMPLETE = (
    "Find and rank suitable locations for a 5 MW grid-connected utility-scale solar PV "
    "project inside this AOI, prioritize annual energy production, and estimate annual "
    "generation for the top three sites. bbox: 16.30,48.15,16.32,48.17 "
    "regular grid cell_size: 0.01 deg. tilt 30 deg azimuth 180 system loss 14%."
)


class _FakeSolarBackend:
    def analyze(self, *, capability_id: str, request: dict[str, Any]) -> dict[str, Any]:
        assert capability_id == "solar_site_selection"
        assert request.get("bbox") == [16.30, 48.15, 16.32, 48.17]
        parameters = request.get("parameters") or {}
        assert parameters.get("ranking_profile") == "energy_focused"
        assert parameters.get("generation_strategy") == "regular_grid"
        assert parameters.get("cell_size") == 0.01
        assert "location" not in request or request["location"].get("longitude") is None
        return {
            "status": "success",
            "capability": capability_id,
            "statistics": {
                "candidate_count": 2,
                "n_ranked": 2,
                "annual_energy": 12000.0,
                "ghi_annual": 1450.0,
            },
            "ranked": [
                {
                    "candidate_id": "r0c0",
                    "rank": 1,
                    "score": 0.9,
                    "ghi": 1450.0,
                    "annual_energy": 12000.0,
                    "selected": True,
                }
            ],
            "features": [
                {
                    "type": "Feature",
                    "id": "r0c0",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [16.30, 48.15],
                                [16.31, 48.15],
                                [16.31, 48.16],
                                [16.30, 48.16],
                                [16.30, 48.15],
                            ]
                        ],
                    },
                    "properties": {
                        "candidate_id": "r0c0",
                        "rank": 1,
                        "ghi_annual": 1450.0,
                        "annual_energy": 12000.0,
                        "yield_origin": "pvgis_provider",
                    },
                }
            ],
            "charts": [
                {
                    "chart_id": "candidate_scores",
                    "title": "Candidate suitability scores",
                    "chart_type": "bar",
                    "x_label": "Candidate",
                    "y_label": "Score",
                    "series": [{"name": "WSM score", "data": [{"x": "r0c0", "y": 0.9}]}],
                    "metadata": {"engine": "SolarDSS"},
                }
            ],
            "detail": "SolarDSS computed solar_site_selection",
            "warnings": ["example DecisionProfile is configuration, not universal truth"],
            "provenance": {
                "plugin_id": "ariadne_solar",
                "capability_id": capability_id,
            },
        }


@pytest.mark.asyncio
async def test_site_selection_complete_request_copies_solardss_outputs() -> None:
    outcome = await _route(SITE_SELECTION_COMPLETE, backend=_FakeSolarBackend())
    payload = outcome.payload
    assert payload.status == "routed"
    assert payload.questions == []
    assert payload.geojson is not None
    assert payload.geojson["features"]
    assert payload.support is not None
    assert payload.support.expected_annual_energy == 12000.0
    assert payload.support.ghi == 1450.0
    assert payload.charts
    assert payload.charts[0].series[0].data[0].y == 0.9
    assert "copied_from_solardss" in outcome.observation
    assert payload.problem.get("location") is None


@pytest.mark.asyncio
async def test_fixed_site_does_not_search_a_region() -> None:
    house = extract_solar_slots(OWN_HOUSE)
    assert house.mode == "fixed_site"
    assert house.location is not None
    assert house.region is None
    assert house.aoi is None
    vienna = extract_solar_slots(VIENNA_FARM)
    assert vienna.mode == "site_selection"
    assert vienna.location is None


def test_target_use_case_extracts_ranking_yield_and_capacity() -> None:
    draft = extract_solar_slots(SITE_SELECTION_COMPLETE)
    assert draft.mode == "site_selection"
    assert draft.site_selection_scope == "aoi_search"
    assert draft.target_capacity_kw == 5000.0
    assert draft.ranking_profile == "energy_focused"
    assert draft.generation_strategy == "regular_grid"
    assert draft.cell_size == 0.01
    assert draft.cell_size_unit == "deg"
    assert draft.bbox == (16.30, 48.15, 16.32, 48.17)
    assert draft.top_n == 3
    assert draft.tilt_deg == 30.0
    assert draft.azimuth_deg == 180.0
    assert draft.system_loss_percent == 14.0
    assert draft.mounting == "ground"
    assert draft.location is None


@pytest.mark.asyncio
async def test_ranking_does_not_generate_new_locations() -> None:
    outcome = await _route(
        "Rank these buildings for rooftop PV: A, B, C",
        args={"candidate_ids": ["A", "B", "C"]},
    )
    fields = [item.field for item in outcome.payload.questions]
    assert "candidate_geometries" in fields
    assert outcome.payload.geojson is None
    assert "not_computed" in outcome.observation


class _FakeRankingBackend:
    def analyze(self, *, capability_id: str, request: dict[str, Any]) -> dict[str, Any]:
        assert capability_id == "solar_candidate_ranking"
        candidates = request.get("candidates") or []
        assert {item["candidate_id"] for item in candidates} == {"A", "B"}
        assert all("longitude" in item or "polygon" in item for item in candidates)
        return {
            "status": "success",
            "capability": capability_id,
            "statistics": {"n_ranked": 2, "annual_energy": 8000.0},
            "ranked": [
                {
                    "candidate_id": "A",
                    "rank": 1,
                    "score": 0.8,
                    "ghi": 1450.0,
                    "annual_energy": 8000.0,
                    "selected": True,
                }
            ],
            "features": [
                {
                    "type": "Feature",
                    "id": "A",
                    "geometry": {"type": "Point", "coordinates": [16.37, 48.20]},
                    "properties": {
                        "candidate_id": "A",
                        "rank": 1,
                        "annual_energy": 8000.0,
                        "yield_origin": "pvgis_provider",
                    },
                }
            ],
            "charts": [
                {
                    "chart_id": "candidate_scores",
                    "title": "Candidate suitability scores",
                    "chart_type": "bar",
                    "x_label": "Candidate",
                    "y_label": "Score",
                    "series": [{"name": "WSM score", "data": [{"x": "A", "y": 0.8}]}],
                    "metadata": {"engine": "SolarDSS"},
                }
            ],
            "detail": "SolarDSS computed solar_candidate_ranking",
            "warnings": [],
            "provenance": {"plugin_id": "ariadne_solar", "capability_id": capability_id},
        }


@pytest.mark.asyncio
async def test_ranking_with_caller_geometries_copies_solardss_outputs() -> None:
    outcome = await _route(
        "Rank these buildings for rooftop PV: A, B and estimate annual generation "
        "for the top five buildings. Prioritize annual energy production. "
        "tilt 30 deg azimuth 180 system loss 14%. 10 kW.",
        args={
            "candidate_ids": ["A", "B"],
            "candidate_geometries": [
                {"candidate_id": "A", "longitude": 16.37, "latitude": 48.20},
                {
                    "candidate_id": "B",
                    "polygon": [
                        [
                            [16.38, 48.21],
                            [16.39, 48.21],
                            [16.39, 48.22],
                            [16.38, 48.22],
                            [16.38, 48.21],
                        ]
                    ],
                },
            ],
        },
        backend=_FakeRankingBackend(),
    )
    payload = outcome.payload
    assert payload.status == "routed"
    assert payload.questions == []
    assert payload.geojson is not None
    assert payload.support is not None
    assert payload.support.expected_annual_energy == 8000.0
    assert "copied_from_solardss" in outcome.observation
    assert payload.problem.get("location") is None


class _FakeFixedSiteBackend:
    def analyze(self, *, capability_id: str, request: dict[str, Any]) -> dict[str, Any]:
        assert capability_id == "solar_fixed_site_analysis"
        location = request.get("location") or {}
        assert location.get("longitude") == 16.37
        assert location.get("latitude") == 48.20
        assert "bbox" not in request
        parameters = request.get("parameters") or {}
        assert parameters.get("constraint_profile") == "rooftop_default_example"
        return {
            "status": "success",
            "capability": capability_id,
            "statistics": {"ghi_annual": 1450.0, "annual_energy": 4200.0},
            "features": [
                {
                    "type": "Feature",
                    "id": "owned-site",
                    "geometry": {"type": "Point", "coordinates": [16.37, 48.20]},
                    "properties": {
                        "ghi_annual": 1450.0,
                        "annual_energy": 4200.0,
                        "yield_origin": "pvgis_provider",
                    },
                }
            ],
            "charts": [
                {
                    "chart_id": "monthly_energy",
                    "title": "Monthly PV energy",
                    "chart_type": "bar",
                    "x_label": "Month",
                    "y_label": "kWh",
                    "series": [{"name": "Energy", "data": [{"x": 1, "y": 350.0}]}],
                    "metadata": {"engine": "SolarDSS"},
                }
            ],
            "detail": "SolarDSS computed solar_fixed_site_analysis",
            "warnings": ["example ConstraintProfile is configuration, not a universal standard"],
            "provenance": {"plugin_id": "ariadne_solar", "capability_id": capability_id},
        }


@pytest.mark.asyncio
async def test_fixed_site_complete_request_copies_solardss_outputs() -> None:
    outcome = await _route(
        "I own this house and want solar panels. lat 48.20 lon 16.37. "
        "rooftop_default_example. estimate monthly and annual energy generation. "
        "tilt 30 deg azimuth 180 system loss 14%. 5 kW.",
        backend=_FakeFixedSiteBackend(),
    )
    payload = outcome.payload
    assert payload.status == "routed"
    assert payload.mode == "fixed_site"
    assert payload.questions == []
    assert payload.geojson is not None
    assert payload.support is not None
    assert payload.support.expected_annual_energy == 4200.0
    assert payload.support.ghi == 1450.0
    assert payload.charts
    assert payload.problem.get("region") is None
    assert payload.problem.get("aoi") is None
    assert "copied_from_solardss" in outcome.observation


def test_grid_connected_does_not_require_maximum_grid_distance() -> None:
    draft = extract_solar_slots(SITE_SELECTION_COMPLETE)
    assert draft.grid_connection_required is None
    assert draft.maximum_grid_distance_m is None
