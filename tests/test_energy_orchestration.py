"""Energy-grid tool registration and routing. No GeoLoadST algorithms."""

from __future__ import annotations

import json
import logging

import pytest
from app.agent.accumulator import ResultAccumulator
from app.agent.comparison_workflow import is_indicator_analysis_request
from app.agent.contracts import GeoAgentRequest
from app.agent.energy_intent import is_energy_grid_request, select_energy_capability
from app.agent.loop import LoopLimits
from app.agent.orchestrator import PlannerExecutorAgent, _model_facing_tools
from app.analytics.charts import AnalysisChart
from app.analytics.datasets import DatasetRegistry
from app.core.errors import (
    EnergyAnalysisInfeasibleError,
    EnergyNetworkLoadError,
    EnergyPluginInternalError,
    EnergyPluginUnavailableError,
)
from app.indicators.selection import DomainResolver
from app.llm.contracts import LLMResponse, ToolCall, ToolDefinition
from app.llm.tool_protocol import recover_prompted_envelope
from app.places.contracts import PlaceRegistry
from app.tools.context import AnalysisRunState, GroundingState, ToolContext
from app.tools.energy_capabilities import resolve_capability_id
from app.tools.energy_tools import (
    TOOL_NAME as ANALYZE_ENERGY_GRID,
)
from app.tools.energy_tools import (
    AnalyzeEnergyGridArgs,
    AnalyzeEnergyGridTool,
    EnergyAnalysisSnapshot,
    PluginEnergyAnalysisBackend,
    classify_energy_status,
    extract_statistics,
)
from app.tools.factory import build_tool_registry
from app.tools.registry import ToolRegistry
from app.tools.simbench_tools import (
    TOOL_NAME as SIMBENCH_QUERY,
)
from app.tools.simbench_tools import (
    SimBenchBus,
    SimBenchConnector,
    SimBenchLine,
    SimBenchLoadedNetwork,
    SimBenchQueryArgs,
    SimBenchQueryTool,
)
from pydantic import ValidationError

from tests.test_orchestrator import ScriptedLLM, _final, _tool_response

ENERGY_QUESTION = "Load SimBench network and analyze spatial load patterns"


def _pca_fixture_charts(network_id: str) -> tuple[AnalysisChart, AnalysisChart]:
    return (
        AnalysisChart.model_validate(
            {
                "chart_id": "pca_explained_variance",
                "title": "PCA Explained Variance",
                "chart_type": "bar",
                "x_label": "Principal component",
                "y_label": "Explained variance",
                "series": [
                    {
                        "name": "Explained variance",
                        "data": [{"x": "PC1", "y": 0.7}, {"x": "PC2", "y": 0.2}],
                    }
                ],
                "metadata": {
                    "engine": "GeoLoadST",
                    "capability_id": "multidim_pca_clustering",
                    "network_id": network_id,
                },
            }
        ),
        AnalysisChart.model_validate(
            {
                "chart_id": "instability_cluster_distribution",
                "title": "Instability Cluster Distribution",
                "chart_type": "bar",
                "x_label": "Cluster id",
                "y_label": "Number of buses",
                "series": [
                    {
                        "name": "Buses",
                        "data": [{"x": "0", "y": 1.0}, {"x": "1", "y": 1.0}],
                    }
                ],
                "metadata": {
                    "engine": "GeoLoadST",
                    "capability_id": "multidim_pca_clustering",
                    "network_id": network_id,
                },
            }
        ),
    )


def _stv_fixture_charts(network_id: str) -> tuple[AnalysisChart, AnalysisChart]:
    return (
        AnalysisChart.model_validate(
            {
                "chart_id": "spatial_semivariogram",
                "title": "Spatial Semivariogram",
                "chart_type": "line",
                "x_label": "Spatial lag",
                "y_label": "Semivariance",
                "series": [
                    {
                        "name": "Semivariance",
                        "data": [
                            {"x": 10.0, "y": 0.2},
                            {"x": 20.0, "y": 0.5},
                            {"x": 40.0, "y": 0.8},
                        ],
                    }
                ],
                "metadata": {
                    "engine": "GeoLoadST",
                    "capability_id": "space_time_variogram",
                    "network_id": network_id,
                },
            }
        ),
        AnalysisChart.model_validate(
            {
                "chart_id": "temporal_semivariogram",
                "title": "Temporal Semivariogram",
                "chart_type": "line",
                "x_label": "Temporal lag",
                "y_label": "Semivariance",
                "series": [
                    {
                        "name": "Semivariance",
                        "data": [
                            {"x": 1.0, "y": 0.1},
                            {"x": 2.0, "y": 0.3},
                            {"x": 3.0, "y": 0.4},
                        ],
                    }
                ],
                "metadata": {
                    "engine": "GeoLoadST",
                    "capability_id": "space_time_variogram",
                    "network_id": network_id,
                },
            }
        ),
    )


class FakeEnergyBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def analyze(
        self,
        *,
        capability_id: str,
        network_id: str,
        loaded: SimBenchLoadedNetwork,
    ) -> EnergyAnalysisSnapshot:
        self.calls.append((capability_id, network_id, loaded.network_id))
        if capability_id == "multidim_pca_clustering":
            return EnergyAnalysisSnapshot(
                capability_id=capability_id,
                network_id=network_id,
                status="completed",
                detail="fixture pca analysis",
                statistics={"cumulative_variance": 0.9},
                spatial_layer={
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [10.1, 53.2]},
                            "properties": {
                                "bus_id": 1,
                                "cluster_id": 0,
                                "analysis": "multidim_pca_clustering",
                                "layer_name": "GeoLoadST PCA Clusters",
                            },
                        },
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [10.2, 53.3]},
                            "properties": {
                                "bus_id": 2,
                                "cluster_id": 1,
                                "analysis": "multidim_pca_clustering",
                                "layer_name": "GeoLoadST PCA Clusters",
                            },
                        },
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[10.1, 53.2], [10.2, 53.3]],
                            },
                            "properties": {"name": "duplicate network edge"},
                        },
                    ],
                },
                warnings=(),
                charts=_pca_fixture_charts(network_id),
            )
        if capability_id == "topology_centrality":
            return EnergyAnalysisSnapshot(
                capability_id=capability_id,
                network_id=network_id,
                status="completed",
                detail="fixture topology analysis",
                statistics={},
                spatial_layer={
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [10.1, 53.2]},
                            "properties": {
                                "bus_id": 1,
                                "degree_centrality": 0.4,
                                "betweenness_centrality": 0.8,
                                "closeness_centrality": 0.3,
                                "analysis": "topology_centrality",
                                "layer_name": "GeoLoadST Topology Centrality",
                            },
                        },
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [10.2, 53.3]},
                            "properties": {
                                "bus_id": 2,
                                "degree_centrality": 0.2,
                                "betweenness_centrality": 0.1,
                                "closeness_centrality": 0.5,
                                "analysis": "topology_centrality",
                                "layer_name": "GeoLoadST Topology Centrality",
                            },
                        },
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[10.1, 53.2], [10.2, 53.3]],
                            },
                            "properties": {"name": "duplicate network edge"},
                        },
                    ],
                },
                warnings=(),
            )
        if capability_id == "space_time_variogram":
            return EnergyAnalysisSnapshot(
                capability_id=capability_id,
                network_id=network_id,
                status="completed",
                detail="fixture stv analysis",
                statistics={
                    "space_range": 120.5,
                    "time_range_steps": 6.0,
                    "time_range_hours": 1.5,
                    "analyzed_bus_count": 3.0,
                    "max_valid_space_lag": 300.0,
                    "max_valid_time_lag": 12.0,
                },
                spatial_layer={
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [10.1, 53.2]},
                            "properties": {"bus_id": 1, "name": "Bus 1"},
                        },
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[10.1, 53.2], [10.2, 53.3]],
                            },
                            "properties": {"name": "line"},
                        },
                    ],
                },
                warnings=(),
                charts=_stv_fixture_charts(network_id),
            )
        return EnergyAnalysisSnapshot(
            capability_id=capability_id,
            network_id=network_id,
            status="completed",
            detail="fixture analysis",
            statistics={"moran_i": 0.34, "p_value": 0.02},
            spatial_layer={
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [10.1, 53.2]},
                        "properties": {
                            "cluster_type": "HIGH_HIGH",
                            "indicator": "LISA",
                            "value": 0.82,
                            "p_value": 0.01,
                            "layer_name": "GeoLoadST LISA Clusters",
                            "source": "geoloadst",
                            "capability_id": capability_id,
                        },
                    }
                ],
            },
            warnings=(),
        )


class FakeSimBenchBackend:
    def list_network_ids(self) -> tuple[str, ...]:
        return ("fixture-alpha-1",)

    def load_network(self, network_id: str) -> SimBenchLoadedNetwork:
        return SimBenchLoadedNetwork(
            network_id=network_id,
            name="fixture grid",
            buses=(SimBenchBus(bus_id=1, vn_kv=20.0, longitude=10.1, latitude=53.2),),
            lines=(SimBenchLine(line_id=10, from_bus=1, to_bus=1),),
            load_count=1,
            has_load_profiles=True,
        )


def _ctx() -> ToolContext:
    return ToolContext(
        datasets=DatasetRegistry(),
        analysis=AnalysisRunState(),
        user_message=ENERGY_QUESTION,
        places=PlaceRegistry(),
        grounding=GroundingState(),
    )


def _energy_registry() -> tuple[ToolRegistry, SimBenchQueryTool, AnalyzeEnergyGridTool]:
    simbench_backend = FakeSimBenchBackend()
    connector = SimBenchConnector(simbench_backend)
    simbench = SimBenchQueryTool(connector)
    energy = AnalyzeEnergyGridTool(FakeEnergyBackend(), connector)
    registry = build_tool_registry(
        simbench_query_tool=simbench,
        analyze_energy_grid_tool=energy,
    )
    return registry, simbench, energy


def test_energy_intent_is_detected_and_not_an_osm_comparison() -> None:
    assert is_energy_grid_request(ENERGY_QUESTION) is True
    assert is_indicator_analysis_request(ENERGY_QUESTION) is False
    assert is_energy_grid_request("Compare parks within 2 km of University of Tehran") is False


def test_planner_detects_energy_grid_domain() -> None:
    selection = DomainResolver().resolve(ENERGY_QUESTION)
    assert "energy_grid" in selection.domains


def test_energy_tools_are_registered() -> None:
    registry, _, _ = _energy_registry()
    assert registry.has(SIMBENCH_QUERY)
    assert registry.has(ANALYZE_ENERGY_GRID)
    names = {item.name for item in registry.definitions()}
    assert names == {SIMBENCH_QUERY, ANALYZE_ENERGY_GRID}


def test_energy_tools_are_in_native_schema() -> None:
    from app.llm.avalai import AvalAIProvider

    registry, _, _ = _energy_registry()
    native = [AvalAIProvider._encode_tool(item) for item in registry.definitions()]
    names = {item["function"]["name"] for item in native}
    assert names == {SIMBENCH_QUERY, ANALYZE_ENERGY_GRID}
    assert all(item["type"] == "function" for item in native)


def test_model_facing_tools_for_energy_hide_osm_and_keep_energy() -> None:
    registry, _, _ = _energy_registry()
    defs = [
        ToolDefinition(name="search_osm_knowledge", description="d", parameters_schema={}),
        ToolDefinition(name="query_osm", description="d", parameters_schema={}),
        ToolDefinition(name="analyze_features", description="d", parameters_schema={}),
        *registry.definitions(),
    ]
    names = [
        item.name
        for item in _model_facing_tools(
            defs,
            state=ResultAccumulator(),
            user_message=ENERGY_QUESTION,
            places=PlaceRegistry(),
        )
    ]
    assert set(names) == {SIMBENCH_QUERY, ANALYZE_ENERGY_GRID}
    assert "query_osm" not in names


@pytest.mark.asyncio
async def test_simbench_then_geoloadst_fills_existing_geojson_pipeline() -> None:
    _, simbench, energy = _energy_registry()
    ctx = _ctx()
    loaded = await simbench.execute(
        SimBenchQueryArgs(network_id="1-complete_data-mixed-all-1-sw"),
        ctx,
    )
    assert ctx.analysis.energy_network_id == "1-complete_data-mixed-all-1-sw"
    analyzed = await energy.execute(
        AnalyzeEnergyGridArgs(
            network_id="1-complete_data-mixed-all-1-sw",
            capability_id="moran_lisa",
        ),
        ctx,
    )
    assert analyzed.payload.status == "success"
    assert analyzed.payload.network_id == "1-complete_data-mixed-all-1-sw"
    assert analyzed.payload.capability == "lisa_instability"
    assert analyzed.payload.requested_capability == "moran_lisa"
    assert analyzed.payload.statistics == {"moran_i": 0.34, "p_value": 0.02}
    assert analyzed.payload.spatial_layer["type"] == "FeatureCollection"
    assert analyzed.payload.geojson is not None
    lisa_feature = analyzed.payload.geojson["features"][0]
    assert lisa_feature["properties"]["cluster_type"] == "HIGH_HIGH"
    assert lisa_feature["properties"]["indicator"] == "LISA"
    assert lisa_feature["properties"]["layer_name"] == "GeoLoadST LISA Clusters"
    assert all(item["geometry"]["type"] == "Point" for item in analyzed.payload.geojson["features"])
    assert energy._backend.calls == [  # type: ignore[attr-defined]
        ("lisa_instability", "1-complete_data-mixed-all-1-sw", "1-complete_data-mixed-all-1-sw")
    ]
    state = ResultAccumulator()
    state.absorb(SIMBENCH_QUERY, loaded.payload)
    state.absorb(ANALYZE_ENERGY_GRID, analyzed.payload)
    assert state.geojson is not None
    assert any(source.kind == "energy_analysis" for source in state.sources)
    assert all(source.kind != "osm_features" for source in state.sources)
    assert "GeoLoadST LISA Clusters" in state.target_feature_counts
    lisa_points = [
        item
        for item in (state.geojson or {}).get("features", [])
        if isinstance(item, dict) and (item.get("properties") or {}).get("indicator") == "LISA"
    ]
    topology_lines = [
        item
        for item in (state.geojson or {}).get("features", [])
        if isinstance(item, dict) and (item.get("geometry") or {}).get("type") == "LineString"
    ]
    assert lisa_points
    assert topology_lines, "SimBench topology must remain as a separate layer"
    assert "SimBench Network" in state.target_feature_counts
    assert state.energy_analysis is not None
    assert state.energy_analysis.kind == "moran_lisa"


def test_recover_prompted_simbench_call_from_native_content() -> None:
    recovered = recover_prompted_envelope(
        '{"tool_calls":[{"name":"simbench_query",'
        '"arguments":{"network_id":"1-complete_data-mixed-all-1-sw"}}]}'
    )
    assert recovered is not None
    assert recovered.tool_calls[0].name == SIMBENCH_QUERY
    assert recovered.tool_calls[0].arguments["network_id"] == "1-complete_data-mixed-all-1-sw"


@pytest.mark.asyncio
async def test_orchestrator_logs_available_tools(caplog: pytest.LogCaptureFixture) -> None:
    registry, _, _ = _energy_registry()

    class _LLM:
        model_name = "fake"

        async def chat(self, messages, *, tools=None, options=None) -> LLMResponse:
            return LLMResponse(content="Need SimBench data first.")

    agent = PlannerExecutorAgent(
        _LLM(),  # type: ignore[arg-type]
        registry,
        limits=LoopLimits(max_tool_rounds=1, max_tool_calls=2),
    )
    caplog.set_level(logging.INFO, logger="app.agent.orchestrator")
    await agent.run(GeoAgentRequest(message=ENERGY_QUESTION))
    assert "available_tools" in caplog.text
    assert SIMBENCH_QUERY in caplog.text
    assert ANALYZE_ENERGY_GRID in caplog.text


def test_example_contract_validates() -> None:
    args = AnalyzeEnergyGridArgs(
        network_id="1-complete_data-mixed-all-1-sw",
        capability_id="moran_lisa",
    )
    assert args.network_id == "1-complete_data-mixed-all-1-sw"
    assert args.capability_id == "moran_lisa"
    assert resolve_capability_id(args.capability_id) == "lisa_instability"


def test_topology_centrality_is_a_registered_capability() -> None:
    args = AnalyzeEnergyGridArgs(
        network_id="1-MV-urban--0-sw",
        capability_id="topology_centrality",
    )
    assert args.capability_id == "topology_centrality"
    assert resolve_capability_id(args.capability_id) == "topology_centrality"


def test_topology_aliases_normalize_to_topology_centrality() -> None:
    for alias in ("topology_analysis", "network_centrality"):
        args = AnalyzeEnergyGridArgs(
            network_id="1-MV-urban--0-sw",
            capability_id=alias,
        )
        assert args.capability_id == alias
        assert resolve_capability_id(alias) == "topology_centrality"


def test_capability_schema_enum_comes_from_registry() -> None:
    schema = AnalyzeEnergyGridArgs.model_json_schema()
    enum = schema["properties"]["capability_id"]["enum"]
    assert "topology_centrality" in enum
    assert "lisa_instability" in enum
    assert "spatial_clustering_of_instability" in enum
    assert "invented_method" not in enum
    assert "topology_analysis" not in enum


def test_natural_language_maps_to_registered_capabilities() -> None:
    assert (
        select_energy_capability("Compute degree, betweenness and closeness centrality.")
        == "topology_centrality"
    )
    assert (
        select_energy_capability("Which buses are structurally most important?")
        == "topology_centrality"
    )
    assert (
        select_energy_capability("Find spatial clusters of unstable loads.")
        == "spatial_clustering_of_instability"
    )


def test_both_string_fields_are_required() -> None:
    with pytest.raises(ValidationError):
        AnalyzeEnergyGridArgs(capability_id="moran_lisa")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        AnalyzeEnergyGridArgs(network_id="1-complete_data-mixed-all-1-sw")  # type: ignore[call-arg]


def test_unknown_capability_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unsupported capability_id"):
        AnalyzeEnergyGridArgs(
            network_id="1-complete_data-mixed-all-1-sw",
            capability_id="invented_method",
        )


def test_unknown_capability_observation_is_structured() -> None:
    from app.core.errors import EnergyUnknownCapabilityError
    from app.tools.energy_capabilities import registered_capability_ids
    from app.tools.validation import format_tool_argument_observation

    with pytest.raises(ValidationError) as exc_info:
        AnalyzeEnergyGridArgs(
            network_id="1-MV-urban--0-sw",
            capability_id="invented_method",
        )
    observation = json.loads(
        format_tool_argument_observation(
            ANALYZE_ENERGY_GRID,
            exc_info.value,
            argument_keys=["capability_id", "network_id"],
            arguments={
                "network_id": "1-MV-urban--0-sw",
                "capability_id": "invented_method",
            },
        )
    )
    assert observation["error_code"] == "unknown_energy_capability"
    assert observation["received"] == "invented_method"
    assert "topology_centrality" in observation["allowed_capabilities"]
    assert observation["allowed_capabilities"] == list(registered_capability_ids())

    with pytest.raises(EnergyUnknownCapabilityError) as energy_exc:
        resolve_capability_id("invented_method")
    error = energy_exc.value
    assert error.code == "energy_unknown_capability"
    payload = json.loads(error.observation)
    assert payload["error_code"] == "unknown_energy_capability"
    assert payload["received"] == "invented_method"
    assert "topology_centrality" in payload["allowed_capabilities"]


@pytest.mark.asyncio
async def test_unknown_capability_is_rejected_before_execution() -> None:
    registry, _, energy = _energy_registry()
    invocation = await registry.invoke(
        ToolCall(
            id="1",
            name=ANALYZE_ENERGY_GRID,
            arguments={
                "network_id": "1-MV-urban--0-sw",
                "capability_id": "invented_method",
            },
        ),
        _ctx(),
    )
    assert invocation.ok is False
    assert invocation.error_code == "tool_argument_error"
    payload = json.loads(invocation.observation)
    assert payload["error_code"] == "unknown_energy_capability"
    assert payload["received"] == "invented_method"
    assert "topology_centrality" in payload["allowed_capabilities"]
    assert energy._backend.calls == []  # type: ignore[attr-defined]


def test_extract_statistics_copies_known_scalars_only() -> None:
    assert extract_statistics({"moran_i": 0.34, "p_value": 0.02, "label": "HH"}) == {
        "moran_i": 0.34,
        "p_value": 0.02,
    }
    assert extract_statistics({"moran_instability": {"I": 0.2, "p_sim": 0.01}}) == {
        "moran_i": 0.2,
        "p_value": 0.01,
    }
    assert extract_statistics({"clusters": [1, 0, 4]}) == {}
    assert extract_statistics(
        {
            "stv_space_range": 120.5,
            "stv_time_range_hours": 1.5,
            "bus_ids_used": {"kind": "array", "values": [1, 2, 3]},
        }
    ) == {
        "space_range": 120.5,
        "time_range_hours": 1.5,
        "analyzed_bus_count": 3.0,
    }
    assert extract_statistics(
        {
            "space_range": 120.5,
            "time_range_steps": 6.0,
            "time_range_hours": 1.5,
            "max_valid_space_lag": 300.0,
            "max_valid_time_lag": 12.0,
            "analyzed_bus_count": 134.0,
        }
    ) == {
        "space_range": 120.5,
        "time_range_steps": 6.0,
        "time_range_hours": 1.5,
        "max_valid_space_lag": 300.0,
        "max_valid_time_lag": 12.0,
        "analyzed_bus_count": 134.0,
    }


@pytest.mark.asyncio
async def test_missing_plugin_is_typed_error_and_logs_traceback(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.tools.energy_tools.find_spec", lambda _name: None)
    tool = AnalyzeEnergyGridTool(
        PluginEnergyAnalysisBackend(),
        SimBenchConnector(FakeSimBenchBackend()),
    )
    caplog.set_level(logging.ERROR)
    with pytest.raises(EnergyPluginUnavailableError):
        await tool.execute(
            AnalyzeEnergyGridArgs(
                network_id="1-complete_data-mixed-all-1-sw",
                capability_id="moran_lisa",
            ),
            _ctx(),
        )
    assert "analyze_energy_grid adapter failed" in caplog.text


@pytest.mark.asyncio
async def test_plugin_backend_uses_public_analyze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Plugin:
        @staticmethod
        def analyze(network_id: str, capability_id: str, network_data: object) -> dict[str, object]:
            assert network_id == "1-complete_data-mixed-all-1-sw"
            assert capability_id == "lisa_instability"
            assert isinstance(network_data, dict)
            return {
                "status": "success",
                "capability": "moran_lisa",
                "statistics": {"moran_i": 0.2, "p_value": 0.01},
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [10.1, 53.2]},
                        "properties": {
                            "cluster_type": "HIGH_HIGH",
                            "indicator": "LISA",
                            "value": 0.2,
                            "p_value": 0.01,
                            "layer_name": "GeoLoadST LISA Clusters",
                            "source": "geoloadst",
                        },
                    }
                ],
                "detail": "fixture plugin",
                "warnings": [],
            }

    monkeypatch.setattr("app.tools.energy_tools.find_spec", lambda _name: object())
    monkeypatch.setattr("app.tools.energy_tools.import_module", lambda _name: _Plugin)
    tool = AnalyzeEnergyGridTool(
        PluginEnergyAnalysisBackend(),
        SimBenchConnector(FakeSimBenchBackend()),
    )
    outcome = await tool.execute(
        AnalyzeEnergyGridArgs(
            network_id="1-complete_data-mixed-all-1-sw",
            capability_id="moran_lisa",
        ),
        _ctx(),
    )
    assert outcome.payload.status == "success"
    assert outcome.payload.capability == "lisa_instability"
    assert outcome.payload.requested_capability == "moran_lisa"
    assert outcome.payload.statistics == {"moran_i": 0.2, "p_value": 0.01}
    assert outcome.payload.feature_count == 1
    assert outcome.payload.geojson is not None
    assert outcome.payload.geojson["features"][0]["properties"]["cluster_type"] == "HIGH_HIGH"


@pytest.mark.asyncio
async def test_topology_centrality_reaches_geoloadst_backend() -> None:
    _, _, energy = _energy_registry()
    outcome = await energy.execute(
        AnalyzeEnergyGridArgs(
            network_id="1-MV-urban--0-sw",
            capability_id="topology_centrality",
        ),
        _ctx(),
    )
    assert outcome.payload.status == "success"
    assert outcome.payload.capability == "topology_centrality"
    assert outcome.payload.requested_capability == "topology_centrality"
    assert energy._backend.calls == [  # type: ignore[attr-defined]
        ("topology_centrality", "1-MV-urban--0-sw", "1-MV-urban--0-sw")
    ]
    report = outcome.payload.analysis
    assert report is not None
    assert report.kind == "topology_centrality"
    assert report.network_id == "1-MV-urban--0-sw"
    assert report.entity_count == 2
    assert report.layer_name == "GeoLoadST Topology Centrality"
    assert report.ranked_by == "betweenness_centrality"
    assert [item.entity_id for item in report.ranked_entities] == ["1", "2"]
    assert report.ranked_entities[0].scores["betweenness_centrality"] == 0.8
    summaries = {item.metric_id: item for item in report.metric_summaries}
    assert summaries["degree_centrality"].maximum == 0.4
    assert summaries["betweenness_centrality"].maximum == 0.8
    assert summaries["closeness_centrality"].maximum == 0.5
    overlay = (outcome.payload.geojson or {})["features"]
    assert all(item["geometry"]["type"] == "Point" for item in overlay)
    assert overlay[0]["properties"]["in_top_n"] is True


@pytest.mark.asyncio
async def test_topology_overlay_keeps_simbench_and_uses_analysis_layer_name() -> None:
    _, simbench, energy = _energy_registry()
    ctx = _ctx()
    loaded = await simbench.execute(
        SimBenchQueryArgs(network_id="1-MV-urban--0-sw"),
        ctx,
    )
    analyzed = await energy.execute(
        AnalyzeEnergyGridArgs(
            network_id="1-MV-urban--0-sw",
            capability_id="topology_centrality",
        ),
        ctx,
    )
    state = ResultAccumulator()
    state.absorb(SIMBENCH_QUERY, loaded.payload)
    state.absorb(ANALYZE_ENERGY_GRID, analyzed.payload)
    assert "SimBench Network" in state.target_feature_counts
    assert "GeoLoadST Topology Centrality" in state.target_feature_counts
    assert "1-MV-urban--0-sw" not in state.target_feature_counts
    assert state.energy_analysis is not None
    assert state.energy_analysis.layer_name == "GeoLoadST Topology Centrality"
    overlay = [
        item
        for item in (state.geojson or {}).get("features", [])
        if isinstance(item, dict)
        and (item.get("properties") or {}).get("analysis") == "topology_centrality"
    ]
    lines = [
        item
        for item in (state.geojson or {}).get("features", [])
        if isinstance(item, dict) and (item.get("geometry") or {}).get("type") == "LineString"
    ]
    assert overlay
    assert all(item["geometry"]["type"] == "Point" for item in overlay)
    assert lines, "SimBench topology must remain underneath the overlay"


@pytest.mark.asyncio
async def test_empty_topology_does_not_reuse_simbench_graph() -> None:
    class _EmptyTopology:
        def analyze(
            self,
            *,
            capability_id: str,
            network_id: str,
            loaded: SimBenchLoadedNetwork,
        ) -> EnergyAnalysisSnapshot:
            del capability_id, network_id, loaded
            return EnergyAnalysisSnapshot(
                capability_id="topology_centrality",
                network_id="1-MV-urban--0-sw",
                status="completed",
                detail="no centrality geometries",
                statistics={},
                spatial_layer={"type": "FeatureCollection", "features": []},
                warnings=(),
            )

    tool = AnalyzeEnergyGridTool(_EmptyTopology(), SimBenchConnector(FakeSimBenchBackend()))
    outcome = await tool.execute(
        AnalyzeEnergyGridArgs(
            network_id="1-MV-urban--0-sw",
            capability_id="topology_centrality",
        ),
        _ctx(),
    )
    assert outcome.payload.geojson is None
    assert outcome.payload.feature_count == 0
    assert any("not reused" in item for item in outcome.payload.warnings)


@pytest.mark.asyncio
async def test_lisa_does_not_reuse_simbench_topology() -> None:
    class _EmptyLisa:
        def analyze(
            self,
            *,
            capability_id: str,
            network_id: str,
            loaded: SimBenchLoadedNetwork,
        ) -> EnergyAnalysisSnapshot:
            del capability_id, network_id, loaded
            return EnergyAnalysisSnapshot(
                capability_id="moran_lisa",
                network_id="1-complete_data-mixed-all-1-sw",
                status="completed",
                detail="no cluster geometries",
                statistics={"moran_i": 0.1, "p_value": 0.04},
                spatial_layer={"type": "FeatureCollection", "features": []},
                warnings=(),
            )

    tool = AnalyzeEnergyGridTool(_EmptyLisa(), SimBenchConnector(FakeSimBenchBackend()))
    outcome = await tool.execute(
        AnalyzeEnergyGridArgs(
            network_id="1-complete_data-mixed-all-1-sw",
            capability_id="moran_lisa",
        ),
        _ctx(),
    )
    assert outcome.payload.geojson is None
    assert outcome.payload.spatial_layer["features"] == []
    assert any("topology" in item.lower() for item in outcome.payload.warnings)


@pytest.mark.asyncio
async def test_pca_clustering_returns_charts_and_cluster_overlay() -> None:
    _, simbench, energy = _energy_registry()
    ctx = _ctx()
    loaded = await simbench.execute(
        SimBenchQueryArgs(network_id="1-MV-urban--0-sw"),
        ctx,
    )
    analyzed = await energy.execute(
        AnalyzeEnergyGridArgs(
            network_id="1-MV-urban--0-sw",
            capability_id="multidim_pca_clustering",
        ),
        ctx,
    )
    assert analyzed.payload.status == "success"
    assert analyzed.payload.capability == "multidim_pca_clustering"
    assert [item.chart_id for item in analyzed.payload.charts] == [
        "pca_explained_variance",
        "instability_cluster_distribution",
    ]
    assert [point.y for point in analyzed.payload.charts[0].series[0].data] == [0.7, 0.2]
    assert analyzed.payload.analysis is not None
    assert analyzed.payload.analysis.cluster_counts == {"0": 1, "1": 1}
    assert analyzed.payload.analysis.layer_name == "GeoLoadST PCA Clusters"
    overlay = analyzed.payload.geojson["features"] if analyzed.payload.geojson else []
    assert all(item["geometry"]["type"] == "Point" for item in overlay)
    assert {item["properties"]["cluster_id"] for item in overlay} == {0, 1}
    state = ResultAccumulator()
    state.absorb(SIMBENCH_QUERY, loaded.payload)
    state.absorb(ANALYZE_ENERGY_GRID, analyzed.payload)
    assert "SimBench Network" in state.target_feature_counts
    assert "GeoLoadST PCA Clusters" in state.target_feature_counts
    assert [item.chart_id for item in state.charts] == [
        "pca_explained_variance",
        "instability_cluster_distribution",
    ]
    lines = [
        item
        for item in (state.geojson or {}).get("features", [])
        if isinstance(item, dict) and (item.get("geometry") or {}).get("type") == "LineString"
    ]
    assert lines, "SimBench topology must remain underneath the PCA overlay"


@pytest.mark.asyncio
async def test_space_time_variogram_returns_charts_and_range_statistics() -> None:
    _, simbench, energy = _energy_registry()
    ctx = _ctx()
    loaded = await simbench.execute(
        SimBenchQueryArgs(network_id="1-MV-urban--0-sw"),
        ctx,
    )
    analyzed = await energy.execute(
        AnalyzeEnergyGridArgs(
            network_id="1-MV-urban--0-sw",
            capability_id="space_time_variogram",
        ),
        ctx,
    )
    assert analyzed.payload.status == "success"
    assert analyzed.payload.capability == "space_time_variogram"
    assert [item.chart_id for item in analyzed.payload.charts] == [
        "spatial_semivariogram",
        "temporal_semivariogram",
    ]
    assert analyzed.payload.charts[0].chart_type == "line"
    assert [point.y for point in analyzed.payload.charts[0].series[0].data] == [0.2, 0.5, 0.8]
    report = analyzed.payload.analysis
    assert report is not None
    assert report.kind == "space_time_variogram"
    assert report.statistics["space_range"] == 120.5
    assert report.statistics["time_range_hours"] == 1.5
    assert report.statistics["analyzed_bus_count"] == 3.0
    labels = {item.key: item.label for item in report.statistic_items}
    assert labels["space_range"] == "Spatial range"
    assert labels["time_range_steps"] == "Temporal range"
    state = ResultAccumulator()
    state.absorb(SIMBENCH_QUERY, loaded.payload)
    state.absorb(ANALYZE_ENERGY_GRID, analyzed.payload)
    assert "SimBench Network" in state.target_feature_counts
    assert [item.chart_id for item in state.charts] == [
        "spatial_semivariogram",
        "temporal_semivariogram",
    ]
    lines = [
        item
        for item in (state.geojson or {}).get("features", [])
        if isinstance(item, dict) and (item.get("geometry") or {}).get("type") == "LineString"
    ]
    assert lines, "SimBench topology must remain on the map"


@pytest.mark.asyncio
async def test_network_load_failure_is_typed_error() -> None:
    class _Missing:
        def load_network(self, network_id: str) -> SimBenchLoadedNetwork:
            raise RuntimeError(f"no such network {network_id}")

        def list_network_ids(self) -> tuple[str, ...]:
            return ()

    tool = AnalyzeEnergyGridTool(FakeEnergyBackend(), SimBenchConnector(_Missing()))
    with pytest.raises(EnergyNetworkLoadError, match="no such network"):
        await tool.execute(
            AnalyzeEnergyGridArgs(
                network_id="1-complete_data-mixed-all-1-sw",
                capability_id="moran_lisa",
            ),
            _ctx(),
        )


def test_classify_energy_status_keeps_internal_and_infeasible_apart() -> None:
    internal = classify_energy_status(
        "engine_error",
        "GeoLoadST failed during 'compute_multidim_instability'",
        capability_id="multidim_pca_clustering",
    )
    assert isinstance(internal, EnergyPluginInternalError)
    assert internal.code == "energy_plugin_internal_error"
    payload = json.loads(internal.observation)
    assert payload["do_not_replan"] is True
    assert payload["capability_id"] == "multidim_pca_clustering"

    infeasible = classify_energy_status(
        "infeasible",
        "prepared load table is missing",
        capability_id="multidim_pca_clustering",
    )
    assert isinstance(infeasible, EnergyAnalysisInfeasibleError)
    assert json.loads(infeasible.observation)["do_not_replan"] is False


@pytest.mark.asyncio
async def test_plugin_nameerror_is_internal_error(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Broken:
        @staticmethod
        def analyze(network_id: str, capability_id: str, network_data: object) -> dict[str, object]:
            del network_id, capability_id, network_data
            raise NameError("name 'charts_from_result' is not defined")

    monkeypatch.setattr("app.tools.energy_tools.find_spec", lambda _name: object())
    monkeypatch.setattr("app.tools.energy_tools.import_module", lambda _name: _Broken)
    tool = AnalyzeEnergyGridTool(
        PluginEnergyAnalysisBackend(),
        SimBenchConnector(FakeSimBenchBackend()),
    )
    caplog.set_level(logging.ERROR)
    with pytest.raises(EnergyPluginInternalError) as exc_info:
        await tool.execute(
            AnalyzeEnergyGridArgs(
                network_id="1-MV-urban--0-sw",
                capability_id="multidim_pca_clustering",
            ),
            _ctx(),
        )
    error = exc_info.value
    assert error.code == "energy_plugin_internal_error"
    payload = json.loads(error.observation)
    assert payload["error_code"] == "energy_plugin_internal_error"
    assert payload["do_not_replan"] is True
    assert payload["capability_id"] == "multidim_pca_clustering"
    assert "charts_from_result" in caplog.text
    assert "NameError" in caplog.text


@pytest.mark.asyncio
async def test_plugin_engine_error_status_is_internal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EngineFail:
        @staticmethod
        def analyze(network_id: str, capability_id: str, network_data: object) -> dict[str, object]:
            del network_id, capability_id, network_data
            return {
                "status": "engine_error",
                "detail": "GeoLoadST failed during 'compute_multidim_instability'",
                "statistics": {},
                "features": [],
            }

    monkeypatch.setattr("app.tools.energy_tools.find_spec", lambda _name: object())
    monkeypatch.setattr("app.tools.energy_tools.import_module", lambda _name: _EngineFail)
    tool = AnalyzeEnergyGridTool(
        PluginEnergyAnalysisBackend(),
        SimBenchConnector(FakeSimBenchBackend()),
    )
    with pytest.raises(EnergyPluginInternalError) as exc_info:
        await tool.execute(
            AnalyzeEnergyGridArgs(
                network_id="1-MV-urban--0-sw",
                capability_id="multidim_pca_clustering",
            ),
            _ctx(),
        )
    assert json.loads(exc_info.value.observation)["do_not_replan"] is True


@pytest.mark.asyncio
async def test_infeasible_capability_does_not_stop_energy_workflow() -> None:
    class _InfeasibleThenOk:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def analyze(
            self,
            *,
            capability_id: str,
            network_id: str,
            loaded: SimBenchLoadedNetwork,
        ) -> EnergyAnalysisSnapshot:
            del network_id, loaded
            self.calls.append(capability_id)
            if capability_id == "multidim_pca_clustering":
                return EnergyAnalysisSnapshot(
                    capability_id=capability_id,
                    network_id="1-MV-urban--0-sw",
                    status="rejected",
                    detail="required load profiles are missing",
                    statistics={},
                    spatial_layer={"type": "FeatureCollection", "features": []},
                    warnings=(),
                )
            return EnergyAnalysisSnapshot(
                capability_id=capability_id,
                network_id="1-MV-urban--0-sw",
                status="completed",
                detail="fixture",
                statistics={},
                spatial_layer={"type": "FeatureCollection", "features": []},
                warnings=(),
            )

    backend = _InfeasibleThenOk()
    connector = SimBenchConnector(FakeSimBenchBackend())
    energy = AnalyzeEnergyGridTool(backend, connector)
    registry = build_tool_registry(
        simbench_query_tool=SimBenchQueryTool(connector),
        analyze_energy_grid_tool=energy,
    )
    with pytest.raises(EnergyAnalysisInfeasibleError):
        await energy.execute(
            AnalyzeEnergyGridArgs(
                network_id="1-MV-urban--0-sw",
                capability_id="multidim_pca_clustering",
            ),
            _ctx(),
        )
    invocation = await registry.invoke(
        ToolCall(
            id="2",
            name=ANALYZE_ENERGY_GRID,
            arguments={
                "network_id": "1-MV-urban--0-sw",
                "capability_id": "topology_centrality",
            },
        ),
        _ctx(),
    )
    assert invocation.ok is True
    assert backend.calls == ["multidim_pca_clustering", "topology_centrality"]


def test_model_facing_tools_hide_energy_after_internal_stop() -> None:
    registry, _, _ = _energy_registry()
    state = ResultAccumulator()
    state.energy_workflow_stopped = True
    state.energy_terminal_error_code = "energy_plugin_internal_error"
    names = [
        item.name
        for item in _model_facing_tools(
            registry.definitions(),
            state=state,
            user_message=ENERGY_QUESTION,
            places=PlaceRegistry(),
        )
    ]
    assert ANALYZE_ENERGY_GRID not in names
    assert SIMBENCH_QUERY in names


@pytest.mark.asyncio
async def test_internal_adapter_error_does_not_replan_other_capability() -> None:
    class _Boom:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def analyze(
            self,
            *,
            capability_id: str,
            network_id: str,
            loaded: SimBenchLoadedNetwork,
        ) -> EnergyAnalysisSnapshot:
            del network_id, loaded
            self.calls.append(capability_id)
            raise NameError("name 'charts_from_result' is not defined")

    backend = _Boom()
    connector = SimBenchConnector(FakeSimBenchBackend())
    energy = AnalyzeEnergyGridTool(backend, connector)
    registry = build_tool_registry(
        simbench_query_tool=SimBenchQueryTool(connector),
        analyze_energy_grid_tool=energy,
    )
    llm = ScriptedLLM(
        [
            _tool_response(
                ToolCall(
                    id="1",
                    name=SIMBENCH_QUERY,
                    arguments={"network_id": "1-MV-urban--0-sw"},
                )
            ),
            _tool_response(
                ToolCall(
                    id="2",
                    name=ANALYZE_ENERGY_GRID,
                    arguments={
                        "network_id": "1-MV-urban--0-sw",
                        "capability_id": "multidim_pca_clustering",
                    },
                )
            ),
            _tool_response(
                ToolCall(
                    id="3",
                    name=ANALYZE_ENERGY_GRID,
                    arguments={
                        "network_id": "1-MV-urban--0-sw",
                        "capability_id": "spatial_clustering_of_instability",
                    },
                )
            ),
            _final("Trying a different scientific method instead."),
        ]
    )
    agent = PlannerExecutorAgent(
        llm,  # type: ignore[arg-type]
        registry,
        limits=LoopLimits(max_tool_rounds=6, max_tool_calls=8),
    )
    result = await agent.run(
        GeoAgentRequest(
            message=(
                "Analyze load instability in SimBench network 1-MV-urban--0-sw "
                "using GeoLoadST PCA and clustering."
            )
        )
    )
    assert backend.calls == ["multidim_pca_clustering"]
    assert any("energy_plugin_internal_error" in item for item in result.errors)
    assert "do not" in result.answer.lower() or "not run" in result.answer.lower()
    assert result.charts == []
    assert llm.seen  # first planning turns happened
    assert len(llm.seen) == 2  # simbench + pca, then stop before the replan turn
