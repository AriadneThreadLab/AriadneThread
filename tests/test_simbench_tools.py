"""Offline SimBench data-connector tests. No SimBench grids are committed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.agent.accumulator import ResultAccumulator
from app.analytics.datasets import DatasetRegistry
from app.core.errors import ToolExecutionError
from app.llm.contracts import ToolCall
from app.places.contracts import PlaceRegistry
from app.tools.context import AnalysisRunState, GroundingState, ToolContext
from app.tools.factory import build_tool_registry
from app.tools.simbench_tools import (
    TOOL_NAME,
    SimBenchBus,
    SimBenchConnector,
    SimBenchLine,
    SimBenchLoadedNetwork,
    SimBenchPackageBackend,
    SimBenchQueryArgs,
    SimBenchQueryTool,
    extract_loaded_network,
)
from pydantic import ValidationError


def _ctx() -> ToolContext:
    return ToolContext(
        datasets=DatasetRegistry(),
        analysis=AnalysisRunState(),
        user_message="Load a SimBench network",
        places=PlaceRegistry(),
        grounding=GroundingState(),
    )


class _Row(dict[str, object]):
    pass


class _Frame:
    def __init__(self, rows: dict[int, dict[str, object]]) -> None:
        self.index = list(rows)
        self.loc = {key: _Row(value) for key, value in rows.items()}


class FakeSimBenchBackend:
    def __init__(self) -> None:
        self.listed = 0
        self.loaded: list[str] = []

    def list_network_ids(self) -> tuple[str, ...]:
        self.listed += 1
        return ("fixture-alpha-1", "fixture-beta-2")

    def load_network(self, network_id: str) -> SimBenchLoadedNetwork:
        self.loaded.append(network_id)
        if network_id == "missing-net":
            raise ToolExecutionError(f"SimBench network '{network_id}' could not be loaded")
        return SimBenchLoadedNetwork(
            network_id=network_id,
            name="fixture grid",
            buses=(
                SimBenchBus(bus_id=1, vn_kv=20.0, longitude=10.1, latitude=53.2),
                SimBenchBus(bus_id=2, vn_kv=20.0, longitude=10.2, latitude=53.3),
                SimBenchBus(bus_id=3, vn_kv=110.0, longitude=None, latitude=None),
            ),
            lines=(
                SimBenchLine(line_id=10, from_bus=1, to_bus=2),
                SimBenchLine(line_id=11, from_bus=2, to_bus=3),
            ),
            load_count=4,
            has_load_profiles=True,
        )


def test_simbench_tool_registers_in_registry() -> None:
    tool = SimBenchQueryTool(SimBenchConnector(FakeSimBenchBackend()))
    registry = build_tool_registry(simbench_query_tool=tool)
    assert registry.has(TOOL_NAME)
    names = {item.name for item in registry.definitions()}
    assert TOOL_NAME in names


@pytest.mark.asyncio
async def test_registry_invokes_simbench_query_by_network_id() -> None:
    tool = SimBenchQueryTool(SimBenchConnector(FakeSimBenchBackend()))
    registry = build_tool_registry(simbench_query_tool=tool)
    result = await registry.invoke(
        ToolCall(
            id="c1",
            name=TOOL_NAME,
            arguments={"network_id": "1-complete_data-mixed-all-1-sw"},
        ),
        _ctx(),
    )
    assert result.ok is True
    assert result.payload is not None
    assert result.payload.metadata is not None
    assert result.payload.metadata.network_id == "1-complete_data-mixed-all-1-sw"


def test_example_network_id_validates() -> None:
    args = SimBenchQueryArgs(network_id="1-complete_data-mixed-all-1-sw")
    assert args.network_id == "1-complete_data-mixed-all-1-sw"


@pytest.mark.asyncio
async def test_lists_networks_from_backend_not_committed_data() -> None:
    backend = FakeSimBenchBackend()
    tool = SimBenchQueryTool(SimBenchConnector(backend))
    outcome = await tool.execute(SimBenchQueryArgs(), _ctx())
    assert backend.listed == 1
    assert outcome.payload.action == "list"
    assert outcome.payload.network_ids == ["fixture-alpha-1", "fixture-beta-2"]
    assert outcome.payload.geojson is None
    assert "simbench_dataset" in outcome.observation
    assert "GeoLoadST" in outcome.observation


@pytest.mark.asyncio
async def test_loads_network_and_extracts_metadata() -> None:
    backend = FakeSimBenchBackend()
    tool = SimBenchQueryTool(SimBenchConnector(backend))
    outcome = await tool.execute(SimBenchQueryArgs(network_id="fixture-alpha-1"), _ctx())
    assert backend.loaded == ["fixture-alpha-1"]
    metadata = outcome.payload.metadata
    assert metadata is not None
    assert metadata.network_id == "fixture-alpha-1"
    assert metadata.name == "fixture grid"
    assert metadata.bus_count == 3
    assert metadata.line_count == 2
    assert metadata.load_count == 4
    assert metadata.has_load_profiles is True
    assert metadata.buses_with_wgs84_coordinates == 2
    assert metadata.mapped_feature_count == 3  # two bus points + one drawable line
    assert outcome.payload.geojson is not None
    kinds = {feature["geometry"]["type"] for feature in outcome.payload.geojson["features"]}
    assert kinds == {"Point", "LineString"}
    assert all(
        feature["properties"]["source"] == "simbench"
        for feature in outcome.payload.geojson["features"]
    )


@pytest.mark.asyncio
async def test_accumulator_exposes_simbench_geojson_not_osm_source() -> None:
    tool = SimBenchQueryTool(SimBenchConnector(FakeSimBenchBackend()))
    outcome = await tool.execute(SimBenchQueryArgs(network_id="fixture-alpha-1"), _ctx())
    state = ResultAccumulator()
    state.absorb(TOOL_NAME, outcome.payload)
    assert state.geojson is not None
    assert state.feature_count == 3
    assert state.sources[0].kind == "simbench_network"
    assert "SimBench" in state.sources[0].title
    assert all(source.kind != "osm_features" for source in state.sources)


def test_extract_metadata_from_pandapower_like_object() -> None:
    net = SimpleNamespace(
        name="duck-net",
        bus=_Frame(
            {
                7: {
                    "vn_kv": 10.0,
                    "geo": '{"type":"Point","coordinates":[8.5,49.1]}',
                }
            }
        ),
        bus_geodata=None,
        line=_Frame({3: {"from_bus": 7, "to_bus": 7}}),
        load=_Frame({1: {"bus": 7}, 2: {"bus": 7}}),
        profiles={"load": object()},
    )
    loaded = extract_loaded_network("duck-code-1", net)
    assert loaded.network_id == "duck-code-1"
    assert loaded.name == "duck-net"
    assert loaded.load_count == 2
    assert loaded.has_load_profiles is True
    assert loaded.buses[0].longitude == 8.5
    assert loaded.buses[0].latitude == 49.1
    connector = SimBenchConnector(FakeSimBenchBackend())
    metadata = connector.metadata_for(loaded, mapped_feature_count=1, truncated=False)
    assert metadata.bus_count == 1
    assert metadata.buses_with_wgs84_coordinates == 1


def test_projected_coordinates_are_not_invented_as_wgs84() -> None:
    loaded = extract_loaded_network(
        "projected-1",
        SimpleNamespace(
            name=None,
            bus=_Frame({1: {"vn_kv": 20.0, "geo": None}}),
            bus_geodata=_Frame({1: {"x": 500000.0, "y": 5400000.0}}),
            line=_Frame({}),
            load=_Frame({}),
            profiles=None,
        ),
    )
    assert loaded.buses[0].longitude is None
    assert loaded.buses[0].latitude is None


def test_invalid_network_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SimBenchQueryArgs(network_id="not a code")


def test_package_backend_reports_missing_simbench(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.simbench_tools.find_spec", lambda name: None)
    backend = SimBenchPackageBackend()
    with pytest.raises(ToolExecutionError, match="simbench is not installed"):
        backend.list_network_ids()
