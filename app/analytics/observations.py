"""Extract typed observation sets from GeoJSON features for analytics."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from app.analytics.contracts import (
    MAX_ANALYSIS_FEATURES,
    AnalysisTarget,
    GeoPoint,
    MetricRequest,
)
from app.analytics.datasets import DatasetRecord
from app.analytics.geometry import (
    haversine_m,
    polygon_area_from_geometry,
    representative_point,
)
from app.core.errors import ToolArgumentError

_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


@dataclass(frozen=True, slots=True)
class CountObservations:
    candidate_count: int
    geometry_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class NumericObservations:
    values: tuple[float, ...]
    candidate_count: int
    missing_count: int
    invalid_count: int
    geometry_counts: dict[str, int]

    @property
    def observation_count(self) -> int:
        return len(self.values)


@dataclass(frozen=True, slots=True)
class AreaObservations:
    areas_m2: tuple[float, ...]
    candidate_count: int
    missing_count: int
    invalid_count: int
    geometry_counts: dict[str, int]
    multi_ring_warnings: int = 0

    @property
    def observation_count(self) -> int:
        return len(self.areas_m2)


@dataclass(frozen=True, slots=True)
class DistanceObservations:
    access_distances_m: tuple[float, ...]
    candidate_count: int
    missing_count: int
    invalid_count: int
    geometry_counts: dict[str, int]
    reference_point_count: int

    @property
    def observation_count(self) -> int:
        return len(self.access_distances_m)


@dataclass(slots=True)
class _GeometryTally:
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, kind: str) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1


def _features(record: DatasetRecord) -> list[dict[str, Any]]:
    raw = record.feature_collection.get("features", [])
    if not isinstance(raw, list):
        return []
    features = [f for f in raw if isinstance(f, dict)]
    if len(features) > MAX_ANALYSIS_FEATURES:
        raise ToolArgumentError(
            f"analysis_input_too_large: dataset has {len(features)} features; "
            f"maximum is {MAX_ANALYSIS_FEATURES}"
        )
    return features


def parse_numeric_tag(raw: Any) -> float | None | str:
    """Return float, None (missing), or 'invalid'."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return "invalid"
    if isinstance(raw, int | float):
        value = float(raw)
        if not math.isfinite(value):
            return "invalid"
        return value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if not _NUMERIC_RE.match(text):
            return "invalid"
        value = float(text)
        if not math.isfinite(value):
            return "invalid"
        return value
    return "invalid"


def extract_count(record: DatasetRecord) -> CountObservations:
    features = _features(record)
    tally = _GeometryTally()
    for feature in features:
        geometry = feature.get("geometry")
        kind = (
            str(geometry.get("type"))
            if isinstance(geometry, dict) and geometry.get("type")
            else "None"
        )
        tally.add(kind)
    return CountObservations(candidate_count=len(features), geometry_counts=dict(tally.counts))


def extract_numeric(record: DatasetRecord, property_key: str) -> NumericObservations:
    features = _features(record)
    values: list[float] = []
    missing = 0
    invalid = 0
    tally = _GeometryTally()
    for feature in features:
        geometry = feature.get("geometry")
        kind = (
            str(geometry.get("type"))
            if isinstance(geometry, dict) and geometry.get("type")
            else "None"
        )
        tally.add(kind)
        props = feature.get("properties")
        tags = props.get("tags") if isinstance(props, dict) else None
        if not isinstance(tags, dict) or property_key not in tags:
            missing += 1
            continue
        parsed = parse_numeric_tag(tags.get(property_key))
        if parsed is None:
            missing += 1
        elif parsed == "invalid":
            invalid += 1
        else:
            values.append(float(parsed))
    return NumericObservations(
        values=tuple(values),
        candidate_count=len(features),
        missing_count=missing,
        invalid_count=invalid,
        geometry_counts=dict(tally.counts),
    )


def extract_areas(record: DatasetRecord) -> AreaObservations:
    features = _features(record)
    areas: list[float] = []
    missing = 0
    invalid = 0
    multi_ring = 0
    tally = _GeometryTally()
    for feature in features:
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            missing += 1
            tally.add("None")
            continue
        kind = str(geometry.get("type") or "None")
        tally.add(kind)
        if kind != "Polygon":
            missing += 1
            continue
        coords = geometry.get("coordinates")
        if isinstance(coords, list) and len(coords) > 1:
            multi_ring += 1
        area = polygon_area_from_geometry(geometry)
        if area is None:
            invalid += 1
        else:
            areas.append(area)
    return AreaObservations(
        areas_m2=tuple(areas),
        candidate_count=len(features),
        missing_count=missing,
        invalid_count=invalid,
        geometry_counts=dict(tally.counts),
        multi_ring_warnings=multi_ring,
    )


def extract_distances(
    record: DatasetRecord,
    reference_points: list[GeoPoint],
) -> DistanceObservations:
    features = _features(record)
    positions: list[tuple[float, float]] = []
    missing = 0
    invalid = 0
    tally = _GeometryTally()
    for feature in features:
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            missing += 1
            tally.add("None")
            continue
        kind = str(geometry.get("type") or "None")
        tally.add(kind)
        point = representative_point(geometry)
        if point is None:
            missing += 1
        else:
            positions.append(point)

    distances: list[float] = []
    for ref in reference_points:
        if not positions:
            break
        nearest = min(haversine_m(ref.lat, ref.lon, lat, lon) for lat, lon in positions)
        distances.append(nearest)

    return DistanceObservations(
        access_distances_m=tuple(distances),
        candidate_count=len(features),
        missing_count=missing,
        invalid_count=invalid,
        geometry_counts=dict(tally.counts),
        reference_point_count=len(reference_points),
    )


def extract_for_metric(
    record: DatasetRecord,
    metric_req: MetricRequest,
    target: AnalysisTarget,
) -> CountObservations | NumericObservations | AreaObservations | DistanceObservations:
    """Dispatch observation extraction for a metric request."""
    metric = metric_req.metric
    if metric in {"count", "density"}:
        return extract_count(record)
    if metric in {"sum", "mean", "median", "standard_deviation", "min", "max"}:
        assert metric_req.property_key is not None
        return extract_numeric(record, metric_req.property_key)
    if metric in {"total_area", "mean_area", "median_area", "coverage_percentage"}:
        return extract_areas(record)
    if metric in {
        "nearest_distance",
        "mean_nearest_distance",
        "median_nearest_distance",
    }:
        return extract_distances(record, list(target.reference_points))
    if metric == "ratio":
        # Ratio observations are built from operand metrics in the engine.
        return extract_count(record)
    raise ToolArgumentError(f"unsupported metric for observation extraction: {metric}")
