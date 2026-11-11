"""Copied SolarDSS values for the Ariadne map and result panel.

Ariadne does not compute GHI, slope, energy, or ranking scores. Every number
on :class:`SolarDecisionSupportReport` is copied from the integration layer.
Missing fields stay empty so the UI can render an em dash.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.analytics.charts import AnalysisChart, parse_charts
from app.osm.contracts import GeoJsonFeatureCollection

SOLAR_ENGINE_NAME = "SolarDSS"
SOLAR_MAP_LAYER_NAME = "SolarDSS"

SOLAR_MAP_LAYER_IDS: tuple[str, ...] = (
    "search_area",
    "excluded_areas",
    "candidate_sites",
    "top_ranked_sites",
    "solar_resource",
    "land_cover",
    "protected_areas",
    "roads",
    "power_lines",
    "substations",
    "buildings",
    "selected_site",
)

SOLAR_CHART_IDS: frozenset[str] = frozenset(
    {
        "monthly_pv_generation",
        "monthly_energy",
        "monthly_irradiation",
        "candidate_ranking",
        "candidate_scores",
        "criteria_comparison",
        "pv_loss_breakdown",
        "solar_resource_seasonality",
        "alternative_comparison",
        "site_indicators",
    }
)

_GEOMETRY_TYPES = frozenset(
    {
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
    }
)

_STRING_FIELDS: dict[str, tuple[str, ...]] = {
    "decision_mode": ("decision_mode", "mode"),
    "project_type": ("project_type",),
    "selected_site": (
        "selected_site",
        "selected_site_id",
        "selected_candidate_id",
        "selected_id",
    ),
    "protected_area_status": (
        "protected_area_status",
        "protected_status",
        "in_protected_area",
    ),
    "module_technology": ("module_technology", "module_type", "technology"),
}

_FLOAT_FIELDS: dict[str, tuple[str, ...]] = {
    "suitability_score": ("suitability_score", "score", "total_score", "top_score"),
    "ghi": ("ghi", "ghi_annual", "ghi_kwh_m2"),
    "dni": ("dni", "dni_annual", "dni_kwh_m2"),
    "slope": ("slope", "slope_deg", "slope_degrees"),
    "aspect": ("aspect", "aspect_deg", "TERRAIN_ASPECT"),
    "shading_impact": (
        "shading_impact",
        "shading",
        "TERRAIN_SHADING_POTENTIAL",
    ),
    "roof_suitability": ("roof_suitability", "rooftop_suitability"),
    "available_area": ("available_area", "available_area_m2", "area_m2"),
    "distance_to_grid": (
        "distance_to_grid",
        "distance_to_grid_m",
        "grid_distance_m",
    ),
    "distance_to_road": (
        "distance_to_road",
        "distance_to_road_m",
        "road_distance_m",
    ),
    "recommended_capacity": (
        "recommended_capacity",
        "recommended_capacity_kwp",
        "capacity_kwp",
        "target_capacity_kw",
    ),
    "tilt": ("tilt", "tilt_deg"),
    "azimuth": ("azimuth", "azimuth_deg"),
    "expected_annual_energy": (
        "expected_annual_energy",
        "annual_energy",
        "annual_energy_kwh",
        "e_y",
    ),
    "specific_yield": ("specific_yield", "specific_yield_kwh_kwp"),
}

_UNIT_ALIASES: dict[str, tuple[str, ...]] = {
    "ghi": ("ghi_unit",),
    "dni": ("dni_unit",),
    "slope": ("slope_unit",),
    "aspect": ("aspect_unit",),
    "shading_impact": ("shading_unit", "shading_impact_unit"),
    "roof_suitability": ("roof_suitability_unit",),
    "available_area": ("available_area_unit", "area_unit"),
    "distance_to_grid": ("distance_to_grid_unit",),
    "distance_to_road": ("distance_to_road_unit",),
    "recommended_capacity": ("recommended_capacity_unit", "capacity_unit"),
    "tilt": ("tilt_unit",),
    "azimuth": ("azimuth_unit",),
    "expected_annual_energy": ("annual_energy_unit", "energy_unit"),
    "specific_yield": ("specific_yield_unit",),
}


class SolarDisplayProvenance(BaseModel):
    """Where the copied SolarDSS values came from. Never chain-of-thought."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine: str = SOLAR_ENGINE_NAME
    source: str = "integration_layer"
    capability_id: str | None = None
    plugin_id: str | None = None
    solardss_version: str | None = None
    request_fingerprint: str | None = None
    detail: str | None = None


class SolarCandidateComparisonRow(BaseModel):
    """One candidate row for the comparison table. Copied, never ranked here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1, max_length=120)
    name: str | None = None
    rank: int | None = Field(default=None, ge=1)
    score: float | None = None
    ghi: float | None = None
    slope: float | None = None
    distance_to_grid: float | None = None
    annual_energy: float | None = None
    selected: bool = False


class SolarDecisionSupportReport(BaseModel):
    """Structured SolarDSS indicators for the UI. Empty means the engine omitted them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine: str = SOLAR_ENGINE_NAME
    capability_id: str | None = None
    decision_mode: str | None = None
    project_type: str | None = None
    selected_site: str | None = None
    suitability_score: float | None = None
    ghi: float | None = None
    dni: float | None = None
    slope: float | None = None
    aspect: float | None = None
    shading_impact: float | None = None
    roof_suitability: float | None = None
    available_area: float | None = None
    distance_to_grid: float | None = None
    distance_to_road: float | None = None
    protected_area_status: str | None = None
    recommended_capacity: float | None = None
    module_technology: str | None = None
    tilt: float | None = None
    azimuth: float | None = None
    expected_annual_energy: float | None = None
    specific_yield: float | None = None
    units: dict[str, str] = Field(default_factory=dict)
    candidates: tuple[SolarCandidateComparisonRow, ...] = ()
    provenance: SolarDisplayProvenance = Field(default_factory=SolarDisplayProvenance)

    @field_validator(
        "suitability_score",
        "ghi",
        "dni",
        "slope",
        "aspect",
        "shading_impact",
        "roof_suitability",
        "available_area",
        "distance_to_grid",
        "distance_to_road",
        "recommended_capacity",
        "tilt",
        "azimuth",
        "expected_annual_energy",
        "specific_yield",
    )
    @classmethod
    def _finite_optional(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("indicator must be a finite number")
        return float(value)


def copy_solar_charts(raw: object) -> list[AnalysisChart]:
    """Keep well-formed engine charts. Drop malformed items; never fill ``y``."""
    charts = parse_charts(raw)
    stamped: list[AnalysisChart] = []
    for chart in charts:
        metadata = chart.metadata
        if metadata.engine:
            stamped.append(chart)
            continue
        stamped.append(
            chart.model_copy(
                update={"metadata": metadata.model_copy(update={"engine": SOLAR_ENGINE_NAME})}
            )
        )
    return stamped


def copy_solar_geojson(raw: object) -> GeoJsonFeatureCollection | None:
    """Copy a FeatureCollection. Features without geometry are dropped, not invented."""
    collection = _as_feature_collection(raw)
    if collection is None:
        return None
    features: list[dict[str, Any]] = []
    for item in collection.get("features", []):
        copied = _copy_feature(item)
        if copied is not None:
            features.append(copied)
    if not features:
        return None
    return {"type": "FeatureCollection", "features": features}


def copy_solar_support(
    payload: object,
    *,
    routing_mode: str | None = None,
    routing_project_type: str | None = None,
    routing_capability: str | None = None,
) -> SolarDecisionSupportReport | None:
    """Copy SolarDSS indicators, candidates, and provenance from a plugin payload."""
    if not isinstance(payload, Mapping):
        return None
    bags = _value_bags(payload)
    strings = {key: _first_str(bags, aliases) for key, aliases in _STRING_FIELDS.items()}
    numbers = {key: _first_float(bags, aliases) for key, aliases in _FLOAT_FIELDS.items()}
    units = _copy_units(bags)
    candidates = _copy_candidates(payload)
    provenance = _copy_provenance(payload, routing_capability)
    capability = _first_str(bags, ("capability_id", "capability")) or routing_capability
    decision_mode = strings["decision_mode"] or routing_mode
    project_type = strings["project_type"] or routing_project_type
    selected = strings["selected_site"]
    if selected is None:
        for row in candidates:
            if row.selected:
                selected = row.candidate_id
                break
    has_values = any(value is not None for value in strings.values()) or any(
        value is not None for value in numbers.values()
    )
    if (
        not has_values
        and not candidates
        and payload.get("provenance") is None
        and payload.get("charts") is None
        and payload.get("geojson") is None
        and payload.get("features") is None
        and payload.get("statistics") is None
    ):
        return None
    return SolarDecisionSupportReport(
        engine=SOLAR_ENGINE_NAME,
        capability_id=capability,
        decision_mode=decision_mode,
        project_type=project_type,
        selected_site=selected,
        suitability_score=numbers["suitability_score"],
        ghi=numbers["ghi"],
        dni=numbers["dni"],
        slope=numbers["slope"],
        aspect=numbers["aspect"],
        shading_impact=numbers["shading_impact"],
        roof_suitability=numbers["roof_suitability"],
        available_area=numbers["available_area"],
        distance_to_grid=numbers["distance_to_grid"],
        distance_to_road=numbers["distance_to_road"],
        protected_area_status=strings["protected_area_status"],
        recommended_capacity=numbers["recommended_capacity"],
        module_technology=strings["module_technology"],
        tilt=numbers["tilt"],
        azimuth=numbers["azimuth"],
        expected_annual_energy=numbers["expected_annual_energy"],
        specific_yield=numbers["specific_yield"],
        units=units,
        candidates=candidates,
        provenance=provenance,
    )


def copy_solar_outputs(
    payload: object,
    *,
    routing_mode: str | None = None,
    routing_project_type: str | None = None,
    routing_capability: str | None = None,
) -> tuple[SolarDecisionSupportReport | None, GeoJsonFeatureCollection | None, list[AnalysisChart]]:
    """Split a plugin-shaped dict into UI report, GeoJSON, and charts."""
    geojson = copy_solar_geojson(_geojson_from_payload(payload))
    charts = copy_solar_charts(_charts_from_payload(payload))
    support = copy_solar_support(
        payload,
        routing_mode=routing_mode,
        routing_project_type=routing_project_type,
        routing_capability=routing_capability,
    )
    if support is None and (geojson is not None or charts):
        support = SolarDecisionSupportReport(
            capability_id=routing_capability,
            decision_mode=routing_mode,
            project_type=routing_project_type,
            provenance=SolarDisplayProvenance(
                engine=SOLAR_ENGINE_NAME,
                source="integration_layer",
                capability_id=routing_capability,
            ),
        )
    return support, geojson, charts


def overlay_routing_slots(
    report: SolarDecisionSupportReport,
    *,
    mode: str | None,
    project_type: str | None,
    capability_id: str | None,
) -> SolarDecisionSupportReport:
    """Fill empty display slots from routing. Never overwrite SolarDSS numbers."""
    updates: dict[str, Any] = {}
    if report.decision_mode is None and mode:
        updates["decision_mode"] = mode
    if report.project_type is None and project_type:
        updates["project_type"] = project_type
    if report.capability_id is None and capability_id:
        updates["capability_id"] = capability_id
    if not updates:
        return report
    return report.model_copy(update=updates)


def _geojson_from_payload(payload: object) -> object:
    if not isinstance(payload, Mapping):
        return None
    for key in ("geojson", "spatial_geojson"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    spatial = payload.get("spatial")
    if isinstance(spatial, Mapping):
        geojson = spatial.get("geojson")
        if isinstance(geojson, Mapping):
            return geojson
    outputs = payload.get("outputs")
    if isinstance(outputs, Mapping):
        geojson = outputs.get("geojson")
        if isinstance(geojson, Mapping):
            return geojson
        features = outputs.get("geojson_features")
        if isinstance(features, list):
            return {"type": "FeatureCollection", "features": features}
    features = payload.get("features")
    if isinstance(features, list):
        return {"type": "FeatureCollection", "features": features}
    return None


def _charts_from_payload(payload: object) -> object:
    if not isinstance(payload, Mapping):
        return None
    charts = payload.get("charts")
    if charts is not None:
        return charts
    outputs = payload.get("outputs")
    if isinstance(outputs, Mapping):
        return outputs.get("charts")
    return None


def _as_feature_collection(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    if raw.get("type") != "FeatureCollection":
        return None
    features = raw.get("features")
    if not isinstance(features, list):
        return None
    return {"type": "FeatureCollection", "features": features}


def _copy_feature(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    geometry = raw.get("geometry")
    if not _usable_geometry(geometry):
        return None
    properties = raw.get("properties")
    copied_props = dict(properties) if isinstance(properties, Mapping) else {}
    layer_id = None
    for key in ("analysis_layer", "solar_layer", "solar_map_layer", "layer_type"):
        value = copied_props.get(key)
        if isinstance(value, str) and value.strip():
            layer_id = value.strip()
            break
    if layer_id is None:
        name = copied_props.get("layer_name")
        if isinstance(name, str) and name.strip().lower() in {
            "solardss candidates",
            "candidate_polygon",
        }:
            layer_id = "candidate_sites"
    if layer_id and "analysis_layer" not in copied_props:
        copied_props["analysis_layer"] = layer_id
    feature: dict[str, Any] = {
        "type": "Feature",
        "geometry": geometry,
        "properties": copied_props,
    }
    ident = raw.get("id")
    if ident is not None:
        feature["id"] = ident
    return feature


def _usable_geometry(raw: object) -> bool:
    if not isinstance(raw, Mapping):
        return False
    geom_type = raw.get("type")
    if geom_type not in _GEOMETRY_TYPES:
        return False
    coordinates = raw.get("coordinates")
    return isinstance(coordinates, list) and bool(coordinates)


def _value_bags(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    bags: list[Mapping[str, Any]] = [payload]
    for key in ("statistics", "indicators", "selected", "detail"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            bags.append(value)
    outputs = payload.get("outputs")
    if isinstance(outputs, Mapping):
        bags.append(outputs)
        nested = outputs.get("statistics")
        if isinstance(nested, Mapping):
            bags.append(nested)
        selected = outputs.get("selected")
        if isinstance(selected, Mapping):
            bags.append(selected)
    return tuple(bags)


def _copy_units(bags: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    units: dict[str, str] = {}
    explicit = None
    for bag in bags:
        raw = bag.get("units")
        if isinstance(raw, Mapping):
            explicit = raw
            break
    if isinstance(explicit, Mapping):
        for key, value in explicit.items():
            if isinstance(key, str) and isinstance(value, str) and value.strip():
                units[key] = value.strip()
    for field, aliases in _UNIT_ALIASES.items():
        if field in units:
            continue
        text = _first_str(bags, aliases)
        if text:
            units[field] = text
    return units


def _copy_provenance(
    payload: Mapping[str, Any], routing_capability: str | None
) -> SolarDisplayProvenance:
    raw = payload.get("provenance")
    bag: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    capability = _as_str(bag.get("capability_id")) or routing_capability
    detail = _as_str(payload.get("detail")) or _as_str(bag.get("detail"))
    return SolarDisplayProvenance(
        engine=_as_str(bag.get("engine")) or SOLAR_ENGINE_NAME,
        source=_as_str(bag.get("source")) or "integration_layer",
        capability_id=capability,
        plugin_id=_as_str(bag.get("plugin_id")),
        solardss_version=_as_str(bag.get("solardss_version")),
        request_fingerprint=_as_str(bag.get("request_fingerprint")),
        detail=detail,
    )


def _copy_candidates(payload: Mapping[str, Any]) -> tuple[SolarCandidateComparisonRow, ...]:
    rows: list[SolarCandidateComparisonRow] = []
    seen: set[str] = set()
    for item in _candidate_items(payload):
        row = _candidate_from_mapping(item)
        if row is None or row.candidate_id in seen:
            continue
        seen.add(row.candidate_id)
        rows.append(row)
    return tuple(rows)


def _candidate_items(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items: list[Mapping[str, Any]] = []
    for key in ("candidates", "ranked"):
        raw = payload.get(key)
        if isinstance(raw, list):
            items.extend(item for item in raw if isinstance(item, Mapping))
    outputs = payload.get("outputs")
    if isinstance(outputs, Mapping):
        for key in ("candidates", "ranked"):
            raw = outputs.get(key)
            if isinstance(raw, list):
                items.extend(item for item in raw if isinstance(item, Mapping))
    features = payload.get("features")
    if isinstance(features, list):
        for feature in features:
            if not isinstance(feature, Mapping):
                continue
            props = feature.get("properties")
            if isinstance(props, Mapping) and props.get("candidate_id"):
                items.append(props)
    return items


def _candidate_from_mapping(raw: Mapping[str, Any]) -> SolarCandidateComparisonRow | None:
    ident = _as_str(raw.get("candidate_id")) or _as_str(raw.get("id"))
    if not ident:
        return None
    selected = raw.get("selected")
    return SolarCandidateComparisonRow(
        candidate_id=ident,
        name=_as_str(raw.get("name")) or _as_str(raw.get("label")),
        rank=_as_rank(raw.get("rank")),
        score=_as_finite_float(
            raw.get("score") if "score" in raw else raw.get("suitability_score")
        ),
        ghi=_as_finite_float(raw.get("ghi") if "ghi" in raw else raw.get("ghi_annual")),
        slope=_as_finite_float(raw.get("slope")),
        distance_to_grid=_as_finite_float(
            raw.get("distance_to_grid")
            if "distance_to_grid" in raw
            else raw.get("distance_to_grid_m")
        ),
        annual_energy=_as_finite_float(
            raw.get("annual_energy")
            if "annual_energy" in raw
            else raw.get("expected_annual_energy")
        ),
        selected=selected is True or selected == "true",
    )


def _first_str(bags: Sequence[Mapping[str, Any]], aliases: Sequence[str]) -> str | None:
    for bag in bags:
        for alias in aliases:
            if alias not in bag:
                continue
            text = _as_str(bag.get(alias))
            if text is not None:
                return text
    return None


def _first_float(bags: Sequence[Mapping[str, Any]], aliases: Sequence[str]) -> float | None:
    for bag in bags:
        for alias in aliases:
            if alias not in bag:
                continue
            number = _as_finite_float(bag.get(alias))
            if number is not None:
                return number
    return None


def _as_str(value: object) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _as_finite_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _as_rank(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and value >= 1:
        return value
    if isinstance(value, float) and math.isfinite(value) and value >= 1 and value == int(value):
        return int(value)
    return None
