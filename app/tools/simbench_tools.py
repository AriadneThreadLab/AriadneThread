"""``simbench_query`` — SimBench energy-network data acquisition.

This tool loads SimBench datasets and returns metadata plus optional GeoJSON
for the existing map viewer. It does not run GeoLoadST analysis and it is not
an OpenStreetMap / Overpass query.

The live ``simbench`` package is imported only when a network is listed or
loaded. Tests inject a connector backend so no SimBench grids are committed.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.errors import ToolArgumentError, ToolExecutionError
from app.osm.contracts import GeoJsonFeatureCollection
from app.tools.context import ToolContext
from app.tools.contracts import ToolOutcome

TOOL_NAME = "simbench_query"
SIMBENCH_ATTRIBUTION = "SimBench dataset"
TOOL_DESCRIPTION = (
    "Acquire a SimBench power-network dataset (not OSM). "
    "Omit network_id to list available codes. Pass network_id to load one "
    "network's metadata and map geometries. Example: "
    '{"network_id":"1-complete_data-mixed-all-1-sw"}. '
    "Does not compute instability or other GeoLoadST statistics."
)

NETWORK_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{1,79}$"
MAX_MAPPED_FEATURES = 1000


class SimBenchQueryArgs(BaseModel):
    """Arguments for listing or loading a SimBench network."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    network_id: str | None = Field(
        default=None,
        description="SimBench network code. Omit to list available codes.",
    )

    @field_validator("network_id")
    @classmethod
    def _normalize_network_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            return None
        if re.fullmatch(NETWORK_ID_PATTERN, text) is None:
            raise ValueError("network_id must be a SimBench-style code")
        return text


class SimBenchNetworkMetadata(BaseModel):
    """Compact facts about one loaded SimBench network."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    network_id: str
    name: str | None = None
    bus_count: int = Field(ge=0)
    line_count: int = Field(ge=0)
    load_count: int = Field(ge=0)
    buses_with_wgs84_coordinates: int = Field(ge=0)
    has_load_profiles: bool
    mapped_feature_count: int = Field(ge=0)
    truncated: bool = False


class SimBenchQueryResult(BaseModel):
    """Structured payload for the API/UI. The model sees only the observation."""

    model_config = ConfigDict(frozen=True)

    action: str
    network_ids: list[str] = Field(default_factory=list)
    metadata: SimBenchNetworkMetadata | None = None
    geojson: GeoJsonFeatureCollection | None = None
    feature_count: int = Field(default=0, ge=0)
    scope_summary: str
    warnings: list[str] = Field(default_factory=list)
    source_title: str
    retrieved_at: datetime


@dataclass(frozen=True, slots=True)
class SimBenchBus:
    """One bus after coordinate extraction. Analysis-free."""

    bus_id: int
    vn_kv: float | None
    longitude: float | None
    latitude: float | None


@dataclass(frozen=True, slots=True)
class SimBenchLine:
    """One line edge. Geometry is derived later from bus coordinates."""

    line_id: int
    from_bus: int
    to_bus: int


@dataclass(frozen=True, slots=True)
class SimBenchLoadedNetwork:
    """Normalized SimBench snapshot used by the connector and tests."""

    network_id: str
    name: str | None
    buses: tuple[SimBenchBus, ...]
    lines: tuple[SimBenchLine, ...]
    load_count: int
    has_load_profiles: bool


class SimBenchBackend(Protocol):
    """Data-access port. Implementations must not run scientific analysis."""

    def list_network_ids(self) -> tuple[str, ...]: ...

    def load_network(self, network_id: str) -> SimBenchLoadedNetwork: ...


class SimBenchConnector:
    """List and load SimBench networks through an injectable backend."""

    def __init__(self, backend: SimBenchBackend | None = None) -> None:
        self._backend = backend if backend is not None else SimBenchPackageBackend()

    def list_available_networks(self) -> tuple[str, ...]:
        return tuple(self._backend.list_network_ids())

    def load_network(self, network_id: str) -> SimBenchLoadedNetwork:
        return self._backend.load_network(network_id)

    def metadata_for(
        self,
        loaded: SimBenchLoadedNetwork,
        *,
        mapped_feature_count: int,
        truncated: bool,
    ) -> SimBenchNetworkMetadata:
        wgs84 = sum(
            1 for bus in loaded.buses if bus.longitude is not None and bus.latitude is not None
        )
        return SimBenchNetworkMetadata(
            network_id=loaded.network_id,
            name=loaded.name,
            bus_count=len(loaded.buses),
            line_count=len(loaded.lines),
            load_count=loaded.load_count,
            buses_with_wgs84_coordinates=wgs84,
            has_load_profiles=loaded.has_load_profiles,
            mapped_feature_count=mapped_feature_count,
            truncated=truncated,
        )


class SimBenchPackageBackend:
    """Lazy wrapper around the optional ``simbench`` package."""

    def list_network_ids(self) -> tuple[str, ...]:
        module = _require_simbench()
        collector = getattr(module, "collect_all_simbench_codes", None)
        if collector is None:
            codes_mod = getattr(module, "simbench_code", None)
            collector = getattr(codes_mod, "collect_all_simbench_codes", None)
        if collector is None:
            raise ToolExecutionError(
                "installed simbench package does not expose collect_all_simbench_codes"
            )
        try:
            codes = collector()
        except Exception as exc:
            raise ToolExecutionError(f"failed to list SimBench networks: {exc}") from exc
        return tuple(str(code) for code in codes)

    def load_network(self, network_id: str) -> SimBenchLoadedNetwork:
        module = _require_simbench()
        loader = getattr(module, "get_simbench_net", None)
        if loader is None:
            raise ToolExecutionError("installed simbench package does not expose get_simbench_net")
        try:
            net = loader(network_id)
        except Exception as exc:
            raise ToolExecutionError(
                f"SimBench network '{network_id}' could not be loaded"
            ) from exc
        return extract_loaded_network(network_id, net)


class SimBenchQueryTool:
    """Registered Tool Registry adapter for SimBench acquisition."""

    def __init__(
        self,
        connector: SimBenchConnector | None = None,
        *,
        max_mapped_features: int = MAX_MAPPED_FEATURES,
    ) -> None:
        self._connector = connector if connector is not None else SimBenchConnector()
        self._max_mapped_features = max_mapped_features

    @property
    def name(self) -> str:
        return TOOL_NAME

    @property
    def description(self) -> str:
        return TOOL_DESCRIPTION

    @property
    def args_model(self) -> type[SimBenchQueryArgs]:
        return SimBenchQueryArgs

    async def execute(
        self, args: SimBenchQueryArgs, context: ToolContext
    ) -> ToolOutcome[SimBenchQueryResult]:
        retrieved_at = datetime.now(tz=timezone.utc)
        if args.network_id is None:
            return self._list(retrieved_at)
        return self._load(args.network_id, retrieved_at, context)

    def _list(self, retrieved_at: datetime) -> ToolOutcome[SimBenchQueryResult]:
        try:
            network_ids = list(self._connector.list_available_networks())
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"failed to list SimBench networks: {exc}") from exc
        observation = (
            f"status=ok source=simbench_dataset action=list "
            f"network_count={len(network_ids)}. "
            "These are SimBench dataset codes, not OSM features. "
            "No GeoLoadST analysis was performed."
        )
        payload = SimBenchQueryResult(
            action="list",
            network_ids=network_ids,
            metadata=None,
            geojson=None,
            feature_count=0,
            scope_summary=f"SimBench catalog: {len(network_ids)} network code(s)",
            warnings=[],
            source_title="SimBench network catalog",
            retrieved_at=retrieved_at,
        )
        return ToolOutcome(observation=observation, payload=payload)

    def _load(
        self,
        network_id: str,
        retrieved_at: datetime,
        context: ToolContext,
    ) -> ToolOutcome[SimBenchQueryResult]:
        try:
            loaded = self._connector.load_network(network_id)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(
                f"SimBench network '{network_id}' could not be loaded"
            ) from exc
        if loaded.network_id != network_id:
            raise ToolArgumentError("loaded network_id does not match the requested code")
        geojson, warnings, truncated = network_to_geojson(loaded, limit=self._max_mapped_features)
        feature_count = len(geojson.get("features") or [])
        metadata = self._connector.metadata_for(
            loaded, mapped_feature_count=feature_count, truncated=truncated
        )
        context.analysis.energy_network_id = metadata.network_id
        if feature_count == 0:
            warnings.append(
                "No WGS84 bus/line geometries were available to draw on the map. "
                "Metadata is still valid. Coordinates were not invented."
            )
        observation = (
            f"status=ok source=simbench_dataset action=load "
            f"network_id={metadata.network_id} buses={metadata.bus_count} "
            f"lines={metadata.line_count} loads={metadata.load_count} "
            f"has_load_profiles={str(metadata.has_load_profiles).lower()} "
            f"mapped_features={feature_count}. "
            "Dataset only — no GeoLoadST scientific analysis was performed."
        )
        payload = SimBenchQueryResult(
            action="load",
            network_ids=[metadata.network_id],
            metadata=metadata,
            geojson=geojson if feature_count else None,
            feature_count=feature_count,
            scope_summary=(
                f"SimBench network {metadata.network_id}: "
                f"{metadata.bus_count} buses, {metadata.line_count} lines"
            ),
            warnings=warnings,
            source_title=f"SimBench network {metadata.network_id}",
            retrieved_at=retrieved_at,
        )
        return ToolOutcome(observation=observation, payload=payload)


def extract_loaded_network(network_id: str, net: Any) -> SimBenchLoadedNetwork:
    """Pull bus/line/load facts from a pandapower-like object. No analysis."""
    bus_ids = _index_ids(getattr(net, "bus", None))
    buses = tuple(_bus_from_net(net, bus_id) for bus_id in bus_ids)
    lines = tuple(_iter_lines(net))
    load_frame = getattr(net, "load", None)
    load_count = len(_index_ids(load_frame))
    profiles = getattr(net, "profiles", None)
    has_profiles = False
    if isinstance(profiles, dict):
        has_profiles = "load" in profiles and profiles["load"] is not None
    name = getattr(net, "name", None)
    if name is not None:
        name = str(name)
    return SimBenchLoadedNetwork(
        network_id=network_id,
        name=name,
        buses=buses,
        lines=lines,
        load_count=load_count,
        has_load_profiles=has_profiles,
    )


def network_to_geojson(
    loaded: SimBenchLoadedNetwork,
    *,
    limit: int,
) -> tuple[GeoJsonFeatureCollection, list[str], bool]:
    """Encode buses and lines that already have WGS84 coordinates."""
    by_id = {bus.bus_id: bus for bus in loaded.buses}
    features: list[dict[str, Any]] = []
    warnings: list[str] = []
    truncated = False

    for bus in loaded.buses:
        if len(features) >= limit:
            truncated = True
            break
        if bus.longitude is None or bus.latitude is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [bus.longitude, bus.latitude],
                },
                "properties": {
                    "name": f"bus {bus.bus_id}",
                    "simbench_element": "bus",
                    "bus_id": bus.bus_id,
                    "vn_kv": bus.vn_kv,
                    "analysis_target": loaded.network_id,
                    "source": "simbench",
                },
            }
        )

    for line in loaded.lines:
        if len(features) >= limit:
            truncated = True
            break
        start = by_id.get(line.from_bus)
        end = by_id.get(line.to_bus)
        if (
            start is None
            or end is None
            or start.longitude is None
            or start.latitude is None
            or end.longitude is None
            or end.latitude is None
        ):
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [start.longitude, start.latitude],
                        [end.longitude, end.latitude],
                    ],
                },
                "properties": {
                    "name": f"line {line.line_id}",
                    "simbench_element": "line",
                    "line_id": line.line_id,
                    "from_bus": line.from_bus,
                    "to_bus": line.to_bus,
                    "analysis_target": loaded.network_id,
                    "source": "simbench",
                },
            }
        )

    if truncated:
        warnings.append(f"Map geometries truncated to {limit} features")
    collection: GeoJsonFeatureCollection = {
        "type": "FeatureCollection",
        "features": features,
    }
    return collection, warnings, truncated


def _require_simbench() -> Any:
    if find_spec("simbench") is None:
        raise ToolExecutionError(
            "simbench is not installed. OSM tools remain available. "
            "Install the optional energy extra to enable simbench_query."
        )
    return import_module("simbench")


def _bus_from_net(net: Any, bus_id: int) -> SimBenchBus:
    longitude, latitude = _coords_for_bus(net, bus_id)
    return SimBenchBus(
        bus_id=bus_id,
        vn_kv=_as_optional_float(_cell(getattr(net, "bus", None), bus_id, "vn_kv")),
        longitude=longitude,
        latitude=latitude,
    )


def _index_ids(frame: Any) -> list[int]:
    if frame is None:
        return []
    index = getattr(frame, "index", None)
    if index is None:
        try:
            return [int(item) for item in frame]
        except Exception:
            return []
    try:
        return [int(item) for item in index]
    except Exception:
        return []


def _cell(frame: Any, row_id: int, column: str) -> Any:
    if frame is None:
        return None
    loc = getattr(frame, "loc", None)
    if loc is None:
        return None
    try:
        row = loc[row_id]
    except Exception:
        return None
    if isinstance(row, dict):
        return row.get(column)
    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(column)
    try:
        return row[column]
    except Exception:
        return None


def _coords_for_bus(net: Any, bus_id: int) -> tuple[float | None, float | None]:
    geodata = getattr(net, "bus_geodata", None)
    if geodata is not None:
        x = _as_optional_float(_cell(geodata, bus_id, "x"))
        y = _as_optional_float(_cell(geodata, bus_id, "y"))
        wgs = _as_wgs84(x, y)
        if wgs is not None:
            return wgs
    geo_raw = _cell(getattr(net, "bus", None), bus_id, "geo")
    parsed = _coords_from_geo(geo_raw)
    if parsed is not None:
        return parsed
    return (None, None)


def _coords_from_geo(raw: Any) -> tuple[float, float] | None:
    if raw is None:
        return None
    if isinstance(raw, list | tuple) and len(raw) >= 2:
        return _as_wgs84(_as_optional_float(raw[0]), _as_optional_float(raw[1]))
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError, ValueError):
        return None
    if isinstance(payload, dict):
        coords = payload.get("coordinates")
        if isinstance(coords, list | tuple) and len(coords) >= 2:
            return _as_wgs84(_as_optional_float(coords[0]), _as_optional_float(coords[1]))
    if isinstance(payload, list | tuple) and len(payload) >= 2:
        return _as_wgs84(_as_optional_float(payload[0]), _as_optional_float(payload[1]))
    return None


def _as_wgs84(x: float | None, y: float | None) -> tuple[float, float] | None:
    if x is None or y is None:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if abs(x) <= 180.0 and abs(y) <= 90.0:
        return (x, y)
    return None


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _iter_lines(net: Any) -> list[SimBenchLine]:
    frame = getattr(net, "line", None)
    rows: list[SimBenchLine] = []
    for line_id in _index_ids(frame):
        from_bus = _as_optional_float(_cell(frame, line_id, "from_bus"))
        to_bus = _as_optional_float(_cell(frame, line_id, "to_bus"))
        if from_bus is None or to_bus is None:
            continue
        rows.append(SimBenchLine(line_id=line_id, from_bus=int(from_bus), to_bus=int(to_bus)))
    return rows


def build_simbench_query_tool() -> SimBenchQueryTool:
    """Composition helper. Does not import or download SimBench at call time."""
    return SimBenchQueryTool()
