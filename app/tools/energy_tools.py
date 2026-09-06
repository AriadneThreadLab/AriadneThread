"""``analyze_energy_grid`` — host contract for GeoLoadST analysis.

The LLM may only pass string ids. This module loads the SimBench network, then
dispatches those ids to the GeoLoadST adapter. It does not compute Moran, LISA,
or other scientific statistics itself.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.analytics.charts import AnalysisChart, parse_charts
from app.core.errors import (
    EnergyAnalysisFailedError,
    EnergyAnalysisInfeasibleError,
    EnergyNetworkLoadError,
    EnergyPluginInternalError,
    EnergyPluginUnavailableError,
    ToolError,
)
from app.osm.contracts import GeoJsonFeatureCollection
from app.tools.context import ToolContext
from app.tools.contracts import ToolOutcome
from app.tools.energy_capabilities import (
    allowed_host_capability_ids,
    analyze_energy_grid_tool_description,
    capability_id_field_description,
    capability_id_looks_valid,
    capability_json_schema,
    resolve_capability,
    safe_logged_energy_ids,
)
from app.tools.energy_report import (
    LAYER_LISA,
    LAYER_PCA,
    LAYER_TOPOLOGY,
    PCA_KIND,
    TOPOLOGY_KIND,
    EnergyAnalysisReport,
    build_energy_analysis_report,
    stamp_ranked_entities,
)
from app.tools.simbench_tools import (
    MAX_MAPPED_FEATURES,
    NETWORK_ID_PATTERN,
    SimBenchConnector,
    SimBenchLoadedNetwork,
    network_to_geojson,
)

logger = logging.getLogger(__name__)

TOOL_NAME = "analyze_energy_grid"
TOOL_DESCRIPTION = analyze_energy_grid_tool_description()
ENERGY_ANALYSIS_ATTRIBUTION = "Energy-grid analysis via GeoLoadST and SimBench. Not OpenStreetMap."
LISA_CANONICAL_ID = "lisa_instability"

_EMPTY_LAYER: GeoJsonFeatureCollection = {"type": "FeatureCollection", "features": []}
LISA_CLUSTER_TYPES = frozenset({"HIGH_HIGH", "LOW_LOW", "HIGH_LOW", "LOW_HIGH"})
LISA_LAYER_NAME = LAYER_LISA
TOPOLOGY_METRIC_KEYS = (
    "degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
)


class AnalyzeEnergyGridArgs(BaseModel):
    """Closed string contract. The model must not send Python objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    network_id: str = Field(description="SimBench network code to load, then analyse.")
    capability_id: str = Field(
        description=capability_id_field_description(),
    )

    @field_validator("network_id")
    @classmethod
    def _normalize_network_id(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("network_id must be a string")
        text = value.strip()
        if re.fullmatch(NETWORK_ID_PATTERN, text) is None:
            raise ValueError("network_id must be a SimBench-style code")
        return text

    @field_validator("capability_id")
    @classmethod
    def _normalize_capability(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("capability_id must be a string")
        text = value.strip()
        if not capability_id_looks_valid(text):
            raise ValueError("capability_id must be a closed catalog identifier")
        if text not in allowed_host_capability_ids():
            raise ValueError(
                "unsupported capability_id; "
                f"registered: {', '.join(capability_json_schema()['enum'])}"
            )
        return text

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        schema = super().model_json_schema(*args, **kwargs)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            properties["capability_id"] = capability_json_schema()
        return schema


class AnalyzeEnergyGridResult(BaseModel):
    """Structured analysis payload for the API, observation, and map viewer."""

    model_config = ConfigDict(frozen=True)

    status: Literal["success"]
    capability: str
    requested_capability: str = ""
    network_id: str
    statistics: dict[str, float] = Field(default_factory=dict)
    spatial_layer: GeoJsonFeatureCollection
    geojson: GeoJsonFeatureCollection | None = None
    feature_count: int = Field(default=0, ge=0)
    scope_summary: str
    warnings: list[str] = Field(default_factory=list)
    source_title: str
    retrieved_at: datetime
    detail: str = ""
    analysis: EnergyAnalysisReport | None = None
    charts: list[AnalysisChart] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EnergyAnalysisSnapshot:
    """Normalized adapter result. Statistics are copied, never invented."""

    capability_id: str
    network_id: str
    status: str
    detail: str
    statistics: dict[str, float]
    spatial_layer: GeoJsonFeatureCollection
    warnings: tuple[str, ...]
    charts: tuple[AnalysisChart, ...] = ()


class EnergyAnalysisBackend(Protocol):
    """Adapter port. Receives string ids plus the already-loaded SimBench snapshot."""

    def analyze(
        self,
        *,
        capability_id: str,
        network_id: str,
        loaded: SimBenchLoadedNetwork,
    ) -> EnergyAnalysisSnapshot: ...


class AnalyzeEnergyGridTool:
    """Load a SimBench network by id, then run the GeoLoadST adapter."""

    def __init__(
        self,
        backend: EnergyAnalysisBackend | None = None,
        connector: SimBenchConnector | None = None,
    ) -> None:
        self._backend = backend if backend is not None else PluginEnergyAnalysisBackend()
        self._connector = connector if connector is not None else SimBenchConnector()

    @property
    def name(self) -> str:
        return TOOL_NAME

    @property
    def description(self) -> str:
        return analyze_energy_grid_tool_description()

    @property
    def args_model(self) -> type[AnalyzeEnergyGridArgs]:
        return AnalyzeEnergyGridArgs

    async def execute(
        self, args: AnalyzeEnergyGridArgs, context: ToolContext
    ) -> ToolOutcome[AnalyzeEnergyGridResult]:
        resolution = resolve_capability(args.capability_id)
        context.analysis.energy_capability_requested = resolution.requested
        context.analysis.energy_capability_canonical = resolution.canonical
        logged = safe_logged_energy_ids(
            network_id=args.network_id,
            capability_id=resolution.requested,
        )
        logger.info(
            "analyze_energy_grid network_id=%s capability_id=%s canonical=%s alias=%s",
            logged.get("network_id", "-"),
            logged.get("capability_id", "-"),
            resolution.canonical,
            resolution.alias_used or "-",
        )
        try:
            loaded = self._connector.load_network(args.network_id)
        except ToolError as exc:
            logger.exception(
                "analyze_energy_grid SimBench load failed network_id=%s",
                args.network_id,
            )
            raise EnergyNetworkLoadError(
                f"SimBench Connector could not load network_id={args.network_id!r}: {exc.message}"
            ) from exc
        except Exception as exc:
            logger.exception(
                "analyze_energy_grid SimBench load failed network_id=%s",
                args.network_id,
            )
            raise EnergyNetworkLoadError(
                f"SimBench Connector could not load network_id={args.network_id!r}: {exc}"
            ) from exc
        context.analysis.energy_network_id = loaded.network_id
        try:
            snapshot = self._backend.analyze(
                capability_id=resolution.canonical,
                network_id=loaded.network_id,
                loaded=loaded,
            )
        except ToolError:
            logger.exception(
                "analyze_energy_grid adapter failed capability_id=%s network_id=%s",
                resolution.canonical,
                loaded.network_id,
            )
            raise
        except Exception as exc:
            logger.exception(
                "analyze_energy_grid adapter failed capability_id=%s network_id=%s",
                resolution.canonical,
                loaded.network_id,
            )
            raise EnergyPluginInternalError(
                f"GeoLoadST adapter failed for capability_id={resolution.canonical!r}: "
                f"{type(exc).__name__}",
                capability_id=resolution.canonical,
            ) from exc
        if snapshot.status != "completed":
            raise classify_energy_status(
                snapshot.status,
                snapshot.detail,
                capability_id=resolution.canonical,
            )
        layer = snapshot.spatial_layer
        features = layer.get("features") if isinstance(layer, dict) else None
        warnings = list(snapshot.warnings)
        if resolution.canonical == LISA_CANONICAL_ID:
            layer = _lisa_cluster_collection(features)
            if not layer["features"]:
                warnings.append(
                    "GeoLoadST returned no LISA cluster geometries. "
                    "The SimBench topology layer was not reused."
                )
        elif resolution.canonical == TOPOLOGY_KIND:
            layer = _topology_overlay_collection(features)
            if not layer["features"]:
                warnings.append(
                    "GeoLoadST returned no topology-centrality geometries. "
                    "The SimBench topology layer was not reused."
                )
        elif resolution.canonical == PCA_KIND:
            layer = _pca_overlay_collection(features)
            if not layer["features"]:
                warnings.append(
                    "GeoLoadST returned no PCA cluster geometries. "
                    "The SimBench topology layer was not reused."
                )
        elif not isinstance(features, list) or not features:
            layer, geo_warnings, _truncated = network_to_geojson(loaded, limit=MAX_MAPPED_FEATURES)
            warnings = warnings + geo_warnings
        overlay_features = [
            item for item in (layer.get("features") or []) if isinstance(item, dict)
        ]
        report = build_energy_analysis_report(
            capability_id=resolution.canonical,
            network_id=loaded.network_id,
            statistics=snapshot.statistics,
            features=overlay_features,
        )
        stamp_ranked_entities(overlay_features, report)
        feature_count = len(overlay_features)
        retrieved_at = datetime.now(tz=timezone.utc)
        chart_ids = [item.chart_id for item in snapshot.charts]
        observation = (
            f"status=success source=geoloadst_analysis "
            f"capability={resolution.canonical} network_id={loaded.network_id} "
            f"requested_capability={resolution.requested} "
            f"alias={resolution.alias_used or '-'} "
            f"statistics={snapshot.statistics} mapped_features={feature_count} "
            f"charts={len(chart_ids)} chart_ids={chart_ids} "
            f"analysis={report.kind} ranked={len(report.ranked_entities)}. "
            f"{snapshot.detail} Numbers and chart curves come from GeoLoadST, "
            f"not from this host."
        )
        payload = AnalyzeEnergyGridResult(
            status="success",
            capability=resolution.canonical,
            requested_capability=resolution.requested,
            network_id=loaded.network_id,
            statistics=snapshot.statistics,
            spatial_layer=layer,
            geojson=layer if feature_count else None,
            feature_count=feature_count,
            scope_summary=f"GeoLoadST {resolution.canonical} on SimBench {loaded.network_id}",
            warnings=warnings,
            source_title=f"GeoLoadST {resolution.canonical}",
            retrieved_at=retrieved_at,
            detail=snapshot.detail,
            analysis=report,
            charts=list(snapshot.charts),
        )
        return ToolOutcome(observation=observation, payload=payload)


class PluginEnergyAnalysisBackend:
    """Dispatch string ids to ``ariadne_geoloadst``. Never vendors algorithms."""

    def analyze(
        self,
        *,
        capability_id: str,
        network_id: str,
        loaded: SimBenchLoadedNetwork,
    ) -> EnergyAnalysisSnapshot:
        module = _require_plugin()
        analyze_fn = getattr(module, "analyze", None)
        if not callable(analyze_fn):
            raise EnergyPluginUnavailableError(
                "installed ariadne_geoloadst package is missing analyze()"
            )
        network_data = {
            "network_id": loaded.network_id,
            "name": loaded.name,
            "bus_count": len(loaded.buses),
            "line_count": len(loaded.lines),
            "load_count": loaded.load_count,
            "has_load_profiles": loaded.has_load_profiles,
        }
        logged = safe_logged_energy_ids(network_id=network_id, capability_id=capability_id)
        logger.info(
            "geoloadst_analyze network_id=%s capability_id=%s",
            logged.get("network_id", "-"),
            logged.get("capability_id", "-"),
        )
        try:
            payload = analyze_fn(
                network_id,
                capability_id,
                network_data,
            )
        except ToolError:
            raise
        except Exception as exc:
            logger.exception(
                "GeoLoadST plugin analyze raised capability_id=%s network_id=%s",
                capability_id,
                network_id,
            )
            raise EnergyPluginInternalError(
                f"GeoLoadST plugin raised {type(exc).__name__} for capability_id={capability_id!r}",
                capability_id=capability_id,
            ) from exc
        if not isinstance(payload, dict):
            raise EnergyPluginInternalError(
                "GeoLoadST plugin returned a non-object result for "
                f"capability_id={capability_id!r}",
                capability_id=capability_id,
            )
        host_status = str(payload.get("status", ""))
        detail = str(payload.get("detail", "") or "GeoLoadST returned no detail.")
        status = "completed" if host_status in {"success", "completed"} else host_status
        if status != "completed":
            raise classify_energy_status(
                status,
                detail,
                capability_id=capability_id,
            )
        raw_stats = payload.get("statistics")
        statistics = extract_statistics(raw_stats) if isinstance(raw_stats, dict) else {}
        features = payload.get("features")
        if isinstance(features, list | tuple) and features:
            layer = {
                "type": "FeatureCollection",
                "features": [item for item in features if isinstance(item, dict)],
            }
        else:
            layer = dict(_EMPTY_LAYER)
        warnings = tuple(str(item) for item in payload.get("warnings", ()) or ())
        charts = tuple(parse_charts(payload.get("charts")))
        return EnergyAnalysisSnapshot(
            capability_id=capability_id,
            network_id=network_id,
            status=status,
            detail=detail,
            statistics=statistics,
            spatial_layer=layer,
            warnings=warnings,
            charts=charts,
        )


def _pca_overlay_collection(features: object) -> GeoJsonFeatureCollection:
    """Keep PCA cluster points only. Never substitute the SimBench graph."""
    if not isinstance(features, list | tuple):
        return dict(_EMPTY_LAYER)
    kept: list[Any] = []
    for raw in features:
        feature = _as_pca_overlay_feature(raw)
        if feature is not None:
            kept.append(feature)
    return {"type": "FeatureCollection", "features": kept}


def _as_pca_overlay_feature(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or raw.get("type") != "Feature":
        return None
    geometry = raw.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") != "Point":
        return None
    props = raw.get("properties")
    if not isinstance(props, dict):
        return None
    cluster_id = _cluster_id(props.get("cluster_id"))
    if cluster_id is None:
        cluster_id = _cluster_id(props.get("value"))
    if cluster_id is None:
        return None
    bus_id = props.get("bus_id")
    name = props.get("name")
    return {
        "type": "Feature",
        "id": raw.get("id"),
        "geometry": geometry,
        "properties": {
            "bus_id": bus_id,
            "cluster_id": cluster_id,
            "analysis": PCA_KIND,
            "layer_name": LAYER_PCA,
            "name": name if isinstance(name, str) else LAYER_PCA,
            "target_label": LAYER_PCA,
            "analysis_target": LAYER_PCA,
            "analysis_target_label": LAYER_PCA,
            "value": cluster_id,
        },
    }


def _cluster_id(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _topology_overlay_collection(features: object) -> GeoJsonFeatureCollection:
    """Keep topology-centrality points only. Never substitute the SimBench graph."""
    if not isinstance(features, list | tuple):
        return dict(_EMPTY_LAYER)
    kept: list[Any] = []
    for raw in features:
        feature = _as_topology_overlay_feature(raw)
        if feature is not None:
            kept.append(feature)
    return {"type": "FeatureCollection", "features": kept}


def _as_topology_overlay_feature(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or raw.get("type") != "Feature":
        return None
    geometry = raw.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") != "Point":
        return None
    props = raw.get("properties")
    if not isinstance(props, dict):
        return None
    metrics: dict[str, float] = {}
    for key in TOPOLOGY_METRIC_KEYS:
        number = _as_finite_float(props.get(key))
        if number is not None:
            metrics[key] = number
    if "degree_centrality" not in metrics:
        fallback = _as_finite_float(props.get("value"))
        if fallback is not None:
            metrics["degree_centrality"] = fallback
    if not metrics:
        return None
    bus_id = props.get("bus_id")
    name = props.get("name")
    visualization_metric = props.get("visualization_metric")
    if visualization_metric not in metrics:
        visualization_metric = next(
            (
                key
                for key in (
                    "betweenness_centrality",
                    "degree_centrality",
                    "closeness_centrality",
                )
                if key in metrics
            ),
            None,
        )
    viz_value = metrics[visualization_metric] if visualization_metric is not None else None
    out_props: dict[str, Any] = {
        "bus_id": bus_id,
        "analysis": TOPOLOGY_KIND,
        "layer_name": LAYER_TOPOLOGY,
        "name": name if isinstance(name, str) else LAYER_TOPOLOGY,
        "target_label": LAYER_TOPOLOGY,
        "analysis_target": LAYER_TOPOLOGY,
        "analysis_target_label": LAYER_TOPOLOGY,
        **metrics,
    }
    if visualization_metric is not None:
        out_props["visualization_metric"] = visualization_metric
    if viz_value is not None:
        out_props["value"] = viz_value
    return {
        "type": "Feature",
        "id": raw.get("id"),
        "geometry": geometry,
        "properties": out_props,
    }


def _lisa_cluster_collection(features: object) -> GeoJsonFeatureCollection:
    """Keep LISA cluster points only. Never substitute the SimBench topology."""
    if not isinstance(features, list):
        return dict(_EMPTY_LAYER)
    kept: list[Any] = []
    for raw in features:
        feature = _as_lisa_cluster_feature(raw)
        if feature is not None:
            kept.append(feature)
    return {"type": "FeatureCollection", "features": kept}


def _as_lisa_cluster_feature(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or raw.get("type") != "Feature":
        return None
    geometry = raw.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") != "Point":
        return None
    props = raw.get("properties")
    if not isinstance(props, dict):
        return None
    cluster_type = _lisa_cluster_type(props)
    if cluster_type is None:
        return None
    value = _as_finite_float(props.get("value"))
    p_value = _as_finite_float(props.get("p_value"))
    if value is None or p_value is None:
        return None
    name = props.get("name")
    return {
        "type": "Feature",
        "id": raw.get("id"),
        "geometry": geometry,
        "properties": {
            "cluster_type": cluster_type,
            "indicator": "LISA",
            "value": value,
            "p_value": p_value,
            "layer_name": LISA_LAYER_NAME,
            "name": name if isinstance(name, str) else LISA_LAYER_NAME,
            "color": _lisa_color(cluster_type),
            "bus_id": props.get("bus_id"),
        },
    }


def _lisa_cluster_type(props: dict[str, Any]) -> str | None:
    raw = props.get("cluster_type")
    if isinstance(raw, str) and raw in LISA_CLUSTER_TYPES:
        return raw
    label = props.get("severity") or props.get("class_label")
    if isinstance(label, str):
        mapped = {
            "high-high": "HIGH_HIGH",
            "high_high": "HIGH_HIGH",
            "low-low": "LOW_LOW",
            "low_low": "LOW_LOW",
            "high-low": "HIGH_LOW",
            "high_low": "HIGH_LOW",
            "low-high": "LOW_HIGH",
            "low_high": "LOW_HIGH",
        }.get(label.strip().lower().replace(" ", "-"))
        if mapped in LISA_CLUSTER_TYPES:
            return mapped
    return None


def _lisa_color(cluster_type: str) -> str:
    return {
        "HIGH_HIGH": "#dc2626",
        "LOW_LOW": "#2563eb",
        "HIGH_LOW": "#ea580c",
        "LOW_HIGH": "#7c3aed",
    }[cluster_type]


def extract_statistics(outputs: dict[str, Any]) -> dict[str, float]:
    """Copy numeric scalars the adapter already computed. Never invent values."""
    stats: dict[str, float] = {}
    _take(stats, outputs, "moran_i", ("moran_i", "morans_i", "I", "Moran_I"))
    _take(stats, outputs, "p_value", ("p_value", "p_sim", "pvalue", "p"))
    nested = outputs.get("moran_instability")
    if isinstance(nested, dict):
        _take(stats, nested, "moran_i", ("moran_i", "morans_i", "I", "Moran_I", "I_obs"))
        _take(stats, nested, "p_value", ("p_value", "p_sim", "pvalue", "p"))
    elif "moran_i" not in stats:
        number = _as_finite_float(nested)
        if number is not None:
            stats["moran_i"] = number
    _take(
        stats,
        outputs,
        "cumulative_variance",
        ("pca_results_cumulative_variance", "cumulative_variance"),
    )
    _take(stats, outputs, "space_range", ("stv_space_range", "space_range"))
    _take(stats, outputs, "time_range_steps", ("stv_time_range_steps", "time_range_steps"))
    _take(stats, outputs, "time_range_hours", ("stv_time_range_hours", "time_range_hours"))
    _take(
        stats,
        outputs,
        "max_valid_space_lag",
        ("stv_max_valid_space_lag", "max_valid_space_lag"),
    )
    _take(
        stats,
        outputs,
        "max_valid_time_lag",
        ("stv_max_valid_time_lag", "max_valid_time_lag"),
    )
    _take(stats, outputs, "analyzed_bus_count", ("analyzed_bus_count",))
    if "analyzed_bus_count" not in stats and (
        "space_range" in stats or "time_range_hours" in stats
    ):
        raw_ids = outputs.get("bus_ids_used")
        values = None
        if isinstance(raw_ids, dict) and raw_ids.get("kind") in {"array", "series"}:
            values = raw_ids.get("values")
        elif isinstance(raw_ids, list | tuple):
            values = raw_ids
        if isinstance(values, list | tuple) and values:
            stats["analyzed_bus_count"] = float(len(values))
    return stats


def classify_energy_status(
    status: str,
    detail: str,
    *,
    capability_id: str,
) -> ToolError:
    """Map a plugin status onto a typed host error. Never invent analysis values."""
    message = f"GeoLoadST plugin status={status} capability_id={capability_id!r}: {detail}"
    if status == "engine_error":
        return EnergyPluginInternalError(message, capability_id=capability_id)
    if status in {"rejected", "not_bound", "infeasible"}:
        return EnergyAnalysisInfeasibleError(message, capability_id=capability_id)
    if status == "unavailable":
        return EnergyPluginUnavailableError(message, capability_id=capability_id)
    return EnergyAnalysisFailedError(message, capability_id=capability_id)


def _require_plugin() -> Any:
    if find_spec("ariadne_geoloadst") is None:
        raise EnergyPluginUnavailableError(
            "ariadne_geoloadst is not installed. simbench_query remains available. "
            "Install the GeoLoadST plugin to enable analyze_energy_grid."
        )
    return import_module("ariadne_geoloadst")


def _take(
    dest: dict[str, float],
    source: dict[str, Any],
    dest_key: str,
    aliases: tuple[str, ...],
) -> None:
    if dest_key in dest:
        return
    for alias in aliases:
        number = _as_finite_float(source.get(alias))
        if number is not None:
            dest[dest_key] = number
            return


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def build_analyze_energy_grid_tool() -> AnalyzeEnergyGridTool:
    """Composition helper. Does not import GeoLoadST at call time."""
    return AnalyzeEnergyGridTool()
