"""Generic structured GeoLoadST analysis results for the Ariadne UI.

The host copies values the plugin already returned. It may summarise and rank
those copied numbers. It must not invent scientific metrics or ask the LLM
to calculate them.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

LAYER_SIMBENCH = "SimBench Network"
LAYER_TOPOLOGY = "GeoLoadST Topology Centrality"
LAYER_LISA = "GeoLoadST LISA Clusters"
LAYER_INSTABILITY = "GeoLoadST Load Instability"
LAYER_VARIOGRAM = "GeoLoadST Space-Time Variogram"
LAYER_PCA = "GeoLoadST PCA Clusters"

TOPOLOGY_KIND = "topology_centrality"
LISA_KIND = "moran_lisa"
INSTABILITY_KIND = "load_instability_rms"
VARIOGRAM_KIND = "space_time_variogram"
PCA_KIND = "multidim_pca_clustering"
GENERIC_KIND = "generic"

TOP_N = 10
TOPOLOGY_RANK_PREFERENCE = (
    "betweenness_centrality",
    "degree_centrality",
    "closeness_centrality",
)
TOPOLOGY_METRIC_LABELS = {
    "degree_centrality": "Degree centrality",
    "betweenness_centrality": "Betweenness centrality",
    "closeness_centrality": "Closeness centrality",
}
LISA_CANONICAL_IDS = frozenset({"lisa_instability", "moran_lisa", "lisa"})
STATISTIC_LABELS = {
    "moran_i": "Moran I",
    "p_value": "p-value",
    "cumulative_variance": "Cumulative explained variance",
    "space_range": "Spatial range",
    "time_range": "Temporal range",
    "time_range_steps": "Temporal range",
    "time_range_hours": "Temporal range (hours)",
    "analyzed_bus_count": "Number of analyzed buses",
    "max_valid_space_lag": "Valid spatial lag range",
    "max_valid_time_lag": "Valid temporal lag range",
}

ANALYSIS_NAMES = {
    TOPOLOGY_KIND: "Topology centrality",
    "lisa_instability": "LISA local spatial autocorrelation",
    "moran_lisa": "LISA local spatial autocorrelation",
    "lisa": "LISA local spatial autocorrelation",
    INSTABILITY_KIND: "Load instability (RMS)",
    VARIOGRAM_KIND: "Space-time variogram",
    PCA_KIND: "Multidimensional PCA and clustering",
}

LAYER_NAMES = {
    TOPOLOGY_KIND: LAYER_TOPOLOGY,
    "lisa_instability": LAYER_LISA,
    "moran_lisa": LAYER_LISA,
    "lisa": LAYER_LISA,
    INSTABILITY_KIND: LAYER_INSTABILITY,
    VARIOGRAM_KIND: LAYER_VARIOGRAM,
    PCA_KIND: LAYER_PCA,
}


class EnergyMetricSummary(BaseModel):
    """Deterministic summary of one copied per-entity metric."""

    model_config = ConfigDict(frozen=True)

    metric_id: str
    label: str
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    minimum: float | None = None
    top_entity_id: str | None = None
    top_entity_score: float | None = None


class EnergyRankedEntity(BaseModel):
    """One ranked bus/entity. Scores are copied GeoLoadST values only."""

    model_config = ConfigDict(frozen=True)

    entity_id: str
    label: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    rank: int | None = Field(default=None, ge=1)


class EnergyStatisticItem(BaseModel):
    """One scalar already present in the GeoLoadST/host statistics dict."""

    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    value: float


class EnergyAnalysisReport(BaseModel):
    """Common Energy Analysis Results contract for current and future capabilities."""

    model_config = ConfigDict(frozen=True)

    capability_id: str
    analysis_name: str
    network_id: str
    kind: str
    entity_count: int = Field(default=0, ge=0)
    statistics: dict[str, float] = Field(default_factory=dict)
    statistic_items: list[EnergyStatisticItem] = Field(default_factory=list)
    metric_summaries: list[EnergyMetricSummary] = Field(default_factory=list)
    ranked_entities: list[EnergyRankedEntity] = Field(default_factory=list)
    ranked_by: str | None = None
    columns: list[str] = Field(default_factory=list)
    layer_name: str
    visualization_metric: str | None = None
    cluster_counts: dict[str, int] = Field(default_factory=dict)
    top_n: int = Field(default=TOP_N, ge=1)


def build_energy_analysis_report(
    *,
    capability_id: str,
    network_id: str,
    statistics: dict[str, float],
    features: list[dict[str, Any]],
) -> EnergyAnalysisReport:
    """Build a UI report from copied plugin values. Missing metrics stay omitted."""
    kind = _kind_for(capability_id)
    if kind == TOPOLOGY_KIND:
        return _topology_report(capability_id, network_id, statistics, features)
    if kind == LISA_KIND:
        return _lisa_report(capability_id, network_id, statistics, features)
    if kind == INSTABILITY_KIND:
        return _value_overlay_report(
            capability_id,
            network_id,
            statistics,
            features,
            kind=INSTABILITY_KIND,
            metric_id="value",
            metric_label="Instability index",
            layer_name=LAYER_INSTABILITY,
        )
    if kind == VARIOGRAM_KIND:
        return _scalar_report(capability_id, network_id, statistics, features, kind=VARIOGRAM_KIND)
    if kind == PCA_KIND:
        return _pca_report(capability_id, network_id, statistics, features)
    return _scalar_report(capability_id, network_id, statistics, features, kind=GENERIC_KIND)


def stamp_ranked_entities(features: list[dict[str, Any]], report: EnergyAnalysisReport) -> None:
    """Mark overlay features that belong to the current top-N ranking."""
    ranked = {
        item.entity_id: item.rank
        for item in report.ranked_entities
        if item.entity_id and item.rank is not None
    }
    for feature in features:
        props = feature.get("properties")
        if not isinstance(props, dict):
            continue
        entity_id = _entity_id(props)
        if entity_id is None:
            continue
        rank = ranked.get(entity_id)
        props["in_top_n"] = rank is not None
        props["top_n"] = report.top_n
        if rank is not None:
            props["rank"] = rank


def layer_name_for(capability_id: str, report: EnergyAnalysisReport | None = None) -> str:
    if report is not None and report.layer_name:
        return report.layer_name
    return LAYER_NAMES.get(capability_id, f"GeoLoadST {capability_id}")


def _topology_report(
    capability_id: str,
    network_id: str,
    statistics: dict[str, float],
    features: list[dict[str, Any]],
) -> EnergyAnalysisReport:
    rows = _topology_rows(features)
    summaries: list[EnergyMetricSummary] = []
    for metric_id, label in TOPOLOGY_METRIC_LABELS.items():
        summary = _metric_summary(metric_id, label, rows)
        if summary is not None:
            summaries.append(summary)
    ranked_by = next(
        (key for key in TOPOLOGY_RANK_PREFERENCE if any(key in row.scores for row in rows)),
        None,
    )
    ranked = _rank_rows(rows, ranked_by, TOP_N)
    columns = [
        key
        for key in TOPOLOGY_METRIC_LABELS
        if any(key in item.scores for item in ranked) or any(key in row.scores for row in rows)
    ]
    return EnergyAnalysisReport(
        capability_id=capability_id,
        analysis_name=ANALYSIS_NAMES[TOPOLOGY_KIND],
        network_id=network_id,
        kind=TOPOLOGY_KIND,
        entity_count=len(rows),
        statistics=dict(statistics),
        statistic_items=_statistic_items(statistics),
        metric_summaries=summaries,
        ranked_entities=ranked,
        ranked_by=ranked_by,
        columns=columns,
        layer_name=LAYER_TOPOLOGY,
        visualization_metric=ranked_by,
        top_n=TOP_N,
    )


def _pca_report(
    capability_id: str,
    network_id: str,
    statistics: dict[str, float],
    features: list[dict[str, Any]],
) -> EnergyAnalysisReport:
    counts: dict[str, int] = {}
    for feature in features:
        props = feature.get("properties")
        if not isinstance(props, dict):
            continue
        cluster = props.get("cluster_id")
        if cluster is None:
            continue
        key = str(cluster)
        if key:
            counts[key] = counts.get(key, 0) + 1
    return EnergyAnalysisReport(
        capability_id=capability_id,
        analysis_name=ANALYSIS_NAMES[PCA_KIND],
        network_id=network_id,
        kind=PCA_KIND,
        entity_count=sum(counts.values()),
        statistics=dict(statistics),
        statistic_items=_statistic_items(statistics),
        cluster_counts=counts,
        layer_name=LAYER_PCA,
    )


def _lisa_report(
    capability_id: str,
    network_id: str,
    statistics: dict[str, float],
    features: list[dict[str, Any]],
) -> EnergyAnalysisReport:
    counts: dict[str, int] = {}
    for feature in features:
        props = feature.get("properties")
        if not isinstance(props, dict):
            continue
        cluster = props.get("cluster_type")
        if isinstance(cluster, str) and cluster:
            counts[cluster] = counts.get(cluster, 0) + 1
    return EnergyAnalysisReport(
        capability_id=capability_id,
        analysis_name=ANALYSIS_NAMES.get(capability_id, ANALYSIS_NAMES[LISA_KIND]),
        network_id=network_id,
        kind=LISA_KIND,
        entity_count=sum(counts.values()),
        statistics=dict(statistics),
        statistic_items=_statistic_items(statistics),
        cluster_counts=counts,
        layer_name=LAYER_LISA,
    )


def _value_overlay_report(
    capability_id: str,
    network_id: str,
    statistics: dict[str, float],
    features: list[dict[str, Any]],
    *,
    kind: str,
    metric_id: str,
    metric_label: str,
    layer_name: str,
) -> EnergyAnalysisReport:
    rows = _value_rows(features, metric_id)
    summary = _metric_summary(metric_id, metric_label, rows)
    ranked_by = metric_id if any(metric_id in row.scores for row in rows) else None
    ranked = _rank_rows(rows, ranked_by, TOP_N)
    return EnergyAnalysisReport(
        capability_id=capability_id,
        analysis_name=ANALYSIS_NAMES.get(capability_id, capability_id),
        network_id=network_id,
        kind=kind,
        entity_count=len(rows),
        statistics=dict(statistics),
        statistic_items=_statistic_items(statistics),
        metric_summaries=[summary] if summary is not None else [],
        ranked_entities=ranked,
        ranked_by=metric_id if ranked else None,
        columns=[metric_id] if ranked else [],
        layer_name=layer_name,
        visualization_metric=metric_id if summary is not None else None,
        top_n=TOP_N,
    )


def _scalar_report(
    capability_id: str,
    network_id: str,
    statistics: dict[str, float],
    features: list[dict[str, Any]],
    *,
    kind: str,
) -> EnergyAnalysisReport:
    count = 0
    for feature in features:
        if isinstance(feature, dict):
            count += 1
    return EnergyAnalysisReport(
        capability_id=capability_id,
        analysis_name=ANALYSIS_NAMES.get(capability_id, capability_id.replace("_", " ")),
        network_id=network_id,
        kind=kind,
        entity_count=count,
        statistics=dict(statistics),
        statistic_items=_statistic_items(statistics),
        layer_name=LAYER_NAMES.get(capability_id, f"GeoLoadST {capability_id}"),
    )


class _Row:
    def __init__(self, entity_id: str, label: str | None, scores: dict[str, float]) -> None:
        self.entity_id = entity_id
        self.label = label
        self.scores = scores


def _topology_rows(features: list[dict[str, Any]]) -> list[_Row]:
    rows: list[_Row] = []
    seen: set[str] = set()
    for feature in features:
        props = feature.get("properties") if isinstance(feature, dict) else None
        if not isinstance(props, dict):
            continue
        entity_id = _entity_id(props)
        if entity_id is None or entity_id in seen:
            continue
        scores: dict[str, float] = {}
        for key in TOPOLOGY_METRIC_LABELS:
            number = _as_finite_float(props.get(key))
            if number is not None:
                scores[key] = number
        if not scores:
            degree_alias = _as_finite_float(props.get("value"))
            if degree_alias is not None and props.get("analysis") in {None, TOPOLOGY_KIND}:
                scores["degree_centrality"] = degree_alias
        if not scores:
            continue
        seen.add(entity_id)
        name = props.get("name")
        rows.append(_Row(entity_id, name if isinstance(name, str) else None, scores))
    return rows


def _value_rows(features: list[dict[str, Any]], metric_id: str) -> list[_Row]:
    rows: list[_Row] = []
    seen: set[str] = set()
    for feature in features:
        props = feature.get("properties") if isinstance(feature, dict) else None
        if not isinstance(props, dict):
            continue
        entity_id = _entity_id(props)
        number = _as_finite_float(props.get(metric_id))
        if entity_id is None or number is None or entity_id in seen:
            continue
        seen.add(entity_id)
        name = props.get("name")
        rows.append(_Row(entity_id, name if isinstance(name, str) else None, {metric_id: number}))
    return rows


def _metric_summary(metric_id: str, label: str, rows: list[_Row]) -> EnergyMetricSummary | None:
    pairs = [(row.entity_id, row.scores[metric_id]) for row in rows if metric_id in row.scores]
    if not pairs:
        return None
    values = [score for _entity, score in pairs]
    top_id, top_score = max(pairs, key=lambda item: (item[1], item[0]))
    return EnergyMetricSummary(
        metric_id=metric_id,
        label=label,
        maximum=max(values),
        mean=_mean(values),
        median=_median(values),
        minimum=min(values),
        top_entity_id=top_id,
        top_entity_score=top_score,
    )


def _rank_rows(rows: list[_Row], ranked_by: str | None, top_n: int) -> list[EnergyRankedEntity]:
    if ranked_by is None:
        return []
    eligible = [row for row in rows if ranked_by in row.scores]
    eligible.sort(key=lambda row: (-row.scores[ranked_by], row.entity_id))
    ranked: list[EnergyRankedEntity] = []
    for index, row in enumerate(eligible[:top_n], start=1):
        ranked.append(
            EnergyRankedEntity(
                entity_id=row.entity_id,
                label=row.label,
                scores=dict(row.scores),
                rank=index,
            )
        )
    return ranked


def _statistic_items(statistics: dict[str, float]) -> list[EnergyStatisticItem]:
    items: list[EnergyStatisticItem] = []
    for key, value in statistics.items():
        number = _as_finite_float(value)
        if number is None:
            continue
        items.append(
            EnergyStatisticItem(
                key=key,
                label=STATISTIC_LABELS.get(key, key.replace("_", " ")),
                value=number,
            )
        )
    return items


def _kind_for(capability_id: str) -> str:
    if capability_id == TOPOLOGY_KIND:
        return TOPOLOGY_KIND
    if capability_id in LISA_CANONICAL_IDS:
        return LISA_KIND
    if capability_id == INSTABILITY_KIND:
        return INSTABILITY_KIND
    if capability_id in {VARIOGRAM_KIND, "directional_variogram"}:
        return VARIOGRAM_KIND
    if capability_id == PCA_KIND:
        return PCA_KIND
    return GENERIC_KIND


def _entity_id(props: dict[str, Any]) -> str | None:
    raw = props.get("bus_id")
    if raw is None:
        raw = props.get("name")
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return str(raw)
    if isinstance(raw, float) and raw.is_integer():
        return str(int(raw))
    text = str(raw).strip()
    return text or None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        number = float(value)
        if math.isfinite(number):
            return number
    return None
