"""Shared scientific-plugin result contract.

OSM ``analyze_features`` keeps ``app.analytics.contracts.AnalysisResult``.
This module is the plugin-facing envelope (GeoLoadST, SolarDSS, future engines).
Every number, geometry, and chart is copied from an engine. Nothing is invented.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.analytics.charts import AnalysisChart
from app.osm.contracts import GeoJsonFeatureCollection
from app.solar.contracts import SolarFollowUpQuestion
from app.solar.display import SOLAR_MAP_LAYER_IDS, SolarDecisionSupportReport
from app.tools.energy_report import STATISTIC_LABELS, EnergyAnalysisReport

AnalysisLayerStyle = dict[str, str | int | float | bool]


class AnalysisMetric(BaseModel):
    """One copied KPI. ``status='missing'`` when the engine omitted the value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=160)
    value: float | int | str | None = None
    unit: str | None = None
    status: Literal["copied", "missing"] = "copied"

    @field_validator("value")
    @classmethod
    def _reject_nonfinite_floats(cls, value: float | int | str | None) -> float | int | str | None:
        if isinstance(value, bool):
            raise ValueError("metric value must not be a boolean")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("metric value must be finite")
        return value


class AnalysisLayer(BaseModel):
    """Named map layer metadata. Geometry is optional; features may live on geojson."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=160)
    geometry: GeoJsonFeatureCollection | None = None
    style: AnalysisLayerStyle = Field(default_factory=dict)
    visible: bool = True
    feature_count: int = Field(default=0, ge=0)


class AnalysisRecommendation(BaseModel):
    """Deterministic engine recommendation. Never model-authored advice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(min_length=1, max_length=160)
    detail: str = Field(min_length=1, max_length=800)
    source: str | None = None


class AnalysisProvenance(BaseModel):
    """Traceable origin of copied scientific outputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine: str = Field(min_length=1, max_length=80)
    data_source: str | None = None
    version: str | None = None
    methodology: str | None = None
    capability_id: str | None = None
    plugin_id: str | None = None


class AgentQuestion(BaseModel):
    """One structured clarification prompt for the Agent Questions panel."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str = Field(min_length=1, max_length=64)
    prompt: str = Field(min_length=4, max_length=200)
    required: bool = True
    source: str | None = Field(default=None, max_length=40)


class ScientificAnalysisResult(BaseModel):
    """Generic plugin analysis envelope. Backward-compatible with energy/solar cards."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_type: str = Field(min_length=1, max_length=80)
    summary: str = Field(default="", max_length=2000)
    metrics: list[AnalysisMetric] = Field(default_factory=list)
    features: GeoJsonFeatureCollection | None = None
    layers: list[AnalysisLayer] = Field(default_factory=list)
    charts: list[AnalysisChart] = Field(default_factory=list)
    recommendations: list[AnalysisRecommendation] = Field(default_factory=list)
    provenance: AnalysisProvenance


_SOLAR_LAYER_TITLES: dict[str, str] = {
    "search_area": "Search area",
    "excluded_areas": "Exclusion areas",
    "candidate_sites": "Candidate locations",
    "top_ranked_sites": "Top-ranked sites",
    "solar_resource": "Solar resource",
    "land_cover": "Land cover",
    "protected_areas": "Protected areas",
    "roads": "Roads",
    "power_lines": "Power lines",
    "substations": "Substations",
    "buildings": "Building footprints",
    "selected_site": "Selected site",
    "suitability": "Suitability polygons",
    "constraints": "Constraints",
}

_SOLAR_LAYER_STYLES: dict[str, AnalysisLayerStyle] = {
    "search_area": {"fill": "#38bdf8"},
    "excluded_areas": {"fill": "#dc2626"},
    "candidate_sites": {"fill": "#0f766e"},
    "top_ranked_sites": {"fill": "#eab308"},
    "solar_resource": {"fill": "#d97706"},
    "land_cover": {"fill": "#65a30d"},
    "protected_areas": {"fill": "#7c3aed"},
    "roads": {"fill": "#78716c"},
    "power_lines": {"fill": "#1d4ed8"},
    "substations": {"fill": "#b45309"},
    "buildings": {"fill": "#64748b"},
    "selected_site": {"fill": "#f43f5e"},
    "suitability": {"fill": "#0f766e"},
    "constraints": {"fill": "#dc2626"},
}

_SOLAR_METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("ghi", "GHI", "ghi"),
    ("dni", "DNI", "dni"),
    ("slope", "Slope", "slope"),
    ("aspect", "Aspect", "aspect"),
    ("shading_impact", "Shading impact", "shading_impact"),
    ("roof_suitability", "Roof suitability", "roof_suitability"),
    ("suitability_score", "Suitability score", "suitability_score"),
    ("available_area", "Available area", "available_area"),
    ("distance_to_grid", "Distance to grid", "distance_to_grid"),
    ("distance_to_road", "Distance to road", "distance_to_road"),
    ("recommended_capacity", "Recommended capacity", "recommended_capacity"),
    ("tilt", "Tilt", "tilt"),
    ("azimuth", "Azimuth", "azimuth"),
    ("expected_annual_energy", "Annual PV yield", "expected_annual_energy"),
    ("specific_yield", "Specific yield", "specific_yield"),
)


def agent_questions_from_solar(
    questions: list[SolarFollowUpQuestion] | tuple[SolarFollowUpQuestion, ...] | None,
) -> list[AgentQuestion]:
    """Copy SolarDSS follow-up questions into the generic clarification contract."""
    if not questions:
        return []
    return [
        AgentQuestion(field=item.field, prompt=item.prompt, required=item.required, source="solar")
        for item in questions
    ]


def scientific_result_from_energy(
    report: EnergyAnalysisReport | None,
    *,
    geojson: GeoJsonFeatureCollection | None = None,
    charts: list[AnalysisChart] | None = None,
) -> ScientificAnalysisResult | None:
    """Wrap a GeoLoadST report. Statistics stay the plugin's copied values."""
    if report is None:
        return None
    metrics: list[AnalysisMetric] = []
    for item in report.statistic_items:
        metrics.append(
            AnalysisMetric(
                id=item.key,
                label=item.label or STATISTIC_LABELS.get(item.key, item.key),
                value=item.value,
                status="copied",
            )
        )
    for metric_summary in report.metric_summaries:
        if metric_summary.mean is not None:
            metrics.append(
                AnalysisMetric(
                    id=f"{metric_summary.metric_id}_mean",
                    label=f"{metric_summary.label} (mean)",
                    value=metric_summary.mean,
                    status="copied",
                )
            )
    features = geojson if geojson is not None else None
    feature_count = 0
    if isinstance(features, dict):
        raw = features.get("features")
        if isinstance(raw, list):
            feature_count = len(raw)
    layer_id = report.kind or report.capability_id
    layers = [
        AnalysisLayer(
            id=layer_id,
            title=report.layer_name,
            geometry=features,
            style={"fill": "#dc2626"},
            visible=True,
            feature_count=feature_count,
        )
    ]
    recommendations: list[AnalysisRecommendation] = []
    if report.ranked_entities:
        top = report.ranked_entities[0]
        score = None
        if report.ranked_by and report.ranked_by in top.scores:
            score = top.scores[report.ranked_by]
        detail = f"Top-ranked bus {top.entity_id}"
        if score is not None:
            detail = f"{detail} ({report.ranked_by}={score})"
        recommendations.append(
            AnalysisRecommendation(title="Highest-ranked entity", detail=detail, source="geoloadst")
        )
    summary = report.analysis_name
    if report.network_id:
        summary = f"{report.analysis_name} on SimBench {report.network_id}"
    return ScientificAnalysisResult(
        analysis_type=report.kind or "energy_analysis",
        summary=summary,
        metrics=metrics,
        features=features,
        layers=layers,
        charts=list(charts or []),
        recommendations=recommendations,
        provenance=AnalysisProvenance(
            engine="GeoLoadST",
            data_source=f"SimBench {report.network_id}" if report.network_id else "SimBench",
            methodology=report.analysis_name,
            capability_id=report.capability_id,
            plugin_id="ariadne_geoloadst",
        ),
    )


def scientific_result_from_solar(
    support: SolarDecisionSupportReport | None,
    *,
    geojson: GeoJsonFeatureCollection | None = None,
    charts: list[AnalysisChart] | None = None,
    capability_id: str | None = None,
) -> ScientificAnalysisResult | None:
    """Wrap copied SolarDSS indicators, layers, and charts. Empty fields stay empty."""
    if support is None and geojson is None and not charts:
        return None
    units = dict(support.units) if support is not None else {}
    metrics: list[AnalysisMetric] = []
    if support is not None:
        for field, label, unit_key in _SOLAR_METRIC_SPECS:
            value = getattr(support, field, None)
            metrics.append(
                AnalysisMetric(
                    id=field,
                    label=label,
                    value=value,
                    unit=units.get(unit_key) or units.get(field),
                    status="copied" if value is not None else "missing",
                )
            )
        if support.selected_site:
            metrics.append(
                AnalysisMetric(
                    id="selected_site",
                    label="Selected site",
                    value=support.selected_site,
                    status="copied",
                )
            )
        if support.protected_area_status:
            metrics.append(
                AnalysisMetric(
                    id="protected_area_status",
                    label="Protected area status",
                    value=support.protected_area_status,
                    status="copied",
                )
            )
        metrics.extend(_monthly_metrics_from_charts(charts or []))
    layers = layers_from_geojson(geojson)
    recommendations = _solar_recommendations(support)
    provenance = AnalysisProvenance(
        engine=(support.engine if support is not None else "SolarDSS") or "SolarDSS",
        data_source=support.provenance.source if support is not None else "integration_layer",
        version=support.provenance.solardss_version if support is not None else None,
        methodology="SolarDSS deterministic screening; values copied through the integration layer",
        capability_id=((support.capability_id if support is not None else None) or capability_id),
        plugin_id=support.provenance.plugin_id if support is not None else "ariadne_solar",
    )
    summary = _solar_summary(support, layers, charts or [])
    analysis_type = (
        (support.capability_id if support is not None else None)
        or capability_id
        or "solar_analysis"
    )
    return ScientificAnalysisResult(
        analysis_type=analysis_type,
        summary=summary,
        metrics=metrics,
        features=geojson,
        layers=layers,
        charts=list(charts or []),
        recommendations=recommendations,
        provenance=provenance,
    )


def layers_from_geojson(geojson: GeoJsonFeatureCollection | None) -> list[AnalysisLayer]:
    """Group copied features by ``analysis_layer`` / ``solar_layer``. No new geometries."""
    if not isinstance(geojson, dict):
        return []
    raw_features = geojson.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        return []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in raw_features:
        if not isinstance(item, dict):
            continue
        layer_id = _feature_layer_id(item)
        buckets.setdefault(layer_id, []).append(item)
    layers: list[AnalysisLayer] = []
    preferred = list(SOLAR_MAP_LAYER_IDS) + [
        key for key in buckets if key not in SOLAR_MAP_LAYER_IDS
    ]
    for layer_id in preferred:
        features = buckets.get(layer_id)
        if not features:
            continue
        layers.append(
            AnalysisLayer(
                id=layer_id,
                title=_SOLAR_LAYER_TITLES.get(layer_id, layer_id.replace("_", " ").title()),
                geometry={"type": "FeatureCollection", "features": features},
                style=dict(_SOLAR_LAYER_STYLES.get(layer_id, {"fill": "#0e7490"})),
                visible=True,
                feature_count=len(features),
            )
        )
    return layers


def _feature_layer_id(feature: dict[str, Any]) -> str:
    props = feature.get("properties")
    bag = props if isinstance(props, dict) else {}
    for key in ("analysis_layer", "solar_layer", "solar_map_layer", "layer_type"):
        value = bag.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    name = bag.get("layer_name")
    if isinstance(name, str) and name.strip():
        token = name.strip().lower().replace(" ", "_")
        aliases = {
            "solardss_candidates": "candidate_sites",
            "candidate_polygon": "candidate_sites",
            "ranked_sites": "top_ranked_sites",
            "site_point": "candidate_sites",
            "suitability_polygons": "suitability",
            "building_footprint": "buildings",
            "exclusion": "excluded_areas",
        }
        return aliases.get(token, token)
    return "analysis"


def _monthly_metrics_from_charts(charts: list[AnalysisChart]) -> list[AnalysisMetric]:
    metrics: list[AnalysisMetric] = []
    for chart in charts:
        if chart.chart_id not in {"monthly_pv_generation", "monthly_energy"}:
            continue
        if not chart.series:
            continue
        for point in chart.series[0].data:
            metrics.append(
                AnalysisMetric(
                    id=f"monthly_pv_{point.x}",
                    label=f"Monthly PV ({point.x})",
                    value=point.y,
                    unit=chart.y_label or None,
                    status="copied",
                )
            )
        break
    return metrics


def _solar_recommendations(
    support: SolarDecisionSupportReport | None,
) -> list[AnalysisRecommendation]:
    if support is None:
        return []
    recs: list[AnalysisRecommendation] = []
    if support.selected_site:
        bits = [f"SolarDSS selected {support.selected_site}"]
        if support.suitability_score is not None:
            bits.append(f"suitability score {support.suitability_score}")
        if support.expected_annual_energy is not None:
            bits.append(f"annual yield {support.expected_annual_energy}")
        recs.append(
            AnalysisRecommendation(
                title="Selected candidate",
                detail="; ".join(bits),
                source="solardss",
            )
        )
    ranked = [row for row in support.candidates if row.rank is not None]
    if len(ranked) >= 2 and not recs:
        top = min(ranked, key=lambda row: row.rank or 10**9)
        recs.append(
            AnalysisRecommendation(
                title="Highest-ranked candidate",
                detail=f"{top.candidate_id} rank {top.rank}",
                source="solardss",
            )
        )
    return recs


def _solar_summary(
    support: SolarDecisionSupportReport | None,
    layers: list[AnalysisLayer],
    charts: list[AnalysisChart],
) -> str:
    if support is None:
        parts = []
        if layers:
            parts.append(f"{sum(item.feature_count for item in layers)} copied geometries")
        if charts:
            parts.append(f"{len(charts)} charts")
        if not parts:
            return "SolarDSS outputs copied."
        return "SolarDSS outputs copied. " + "; ".join(parts)
    mode = support.decision_mode or "solar"
    if support.selected_site:
        return f"SolarDSS {mode}: selected {support.selected_site}."
    if support.candidates:
        return f"SolarDSS {mode}: {len(support.candidates)} ranked candidates copied."
    return f"SolarDSS {mode} results copied from the integration layer."
