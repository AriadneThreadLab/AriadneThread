"""Deterministic indicator executors.

Each catalog ``method_id`` that is marked implemented is computed here from
already-retrieved GeoJSON. Formulas are not parsed from YAML. This module does
not call Overpass, Nominatim, or the agent loop.
"""

from __future__ import annotations

from typing import Any

from app.analytics.contracts import GeoPoint
from app.analytics.datasets import DatasetRecord
from app.analytics.engine import SpatialAnalyticsEngine
from app.analytics.geometry import (
    clipped_geometry_length_m,
    geometry_area_m2,
    geometry_length_m,
    nearest_distance_m,
)
from app.analytics.methods import METHOD_REGISTRY
from app.analytics.network import count_network_intersections
from app.analytics.observations import (
    AreaObservations,
    CategoryObservations,
    CountObservations,
    DistanceObservations,
    LengthObservations,
)
from app.core.errors import IndicatorComputeError, UnknownIndicatorError
from app.indicators.catalog import INDICATOR_CATALOG, IndicatorCatalog
from app.indicators.categorize import categorize_features
from app.indicators.contracts import (
    FeatureSpecification,
    IndicatorComputationResult,
    IndicatorComputeRequest,
    IndicatorComputeStatus,
    IndicatorDefinition,
)

_GREEN_ACCESSIBILITY_REASON = (
    "green_accessibility is unimplemented: distance_decay_sum is a family of "
    "Hansen-style measures, not a pinned methodology. The catalog formula does "
    "not specify whether distance_i is to the polygon boundary, centroid, or "
    "vertex, nor whether area_i is clipped to the analysis scope."
)


class IndicatorExecutor:
    """Dispatch a catalog indicator to a closed engine method."""

    def __init__(
        self,
        *,
        catalog: IndicatorCatalog | None = None,
        engine: SpatialAnalyticsEngine | None = None,
    ) -> None:
        self._catalog = catalog if catalog is not None else INDICATOR_CATALOG
        self._engine = engine if engine is not None else SpatialAnalyticsEngine()

    def compute(self, request: IndicatorComputeRequest) -> IndicatorComputationResult:
        """Compute one indicator. Unknown ids fail; unimplemented methods are reported."""
        try:
            definition = self._catalog.get_indicator(request.indicator_id)
        except UnknownIndicatorError:
            raise
        spec = METHOD_REGISTRY[definition.method_id]
        if not spec.implemented:
            reason = (
                _GREEN_ACCESSIBILITY_REASON
                if definition.indicator_id == "green_accessibility"
                else (
                    f"indicator '{definition.indicator_id}' method "
                    f"'{definition.method_id}' is not implemented"
                )
            )
            return _result(
                definition,
                value=None,
                observation_count=0,
                missing_count=0,
                dataset_refs=_dataset_refs(request),
                warnings=(reason,),
                status="unimplemented",
            )

        record = _osm_record(definition, request)
        if definition.method_id == "count":
            return self._count(definition, record)
        if definition.method_id == "density":
            return self._density(definition, record)
        if definition.method_id == "area_share":
            return self._area_share(definition, record)
        if definition.method_id == "nearest_distance":
            return self._nearest_distance(definition, record, request.reference_points)
        if definition.method_id == "length_density":
            return self._length_density(definition, record)
        if definition.method_id == "shannon_entropy":
            return self._shannon_entropy(definition, record)
        if definition.method_id == "intersection_density":
            return self._intersection_density(definition, record)
        raise IndicatorComputeError(
            f"no executor is registered for implemented method '{definition.method_id}'"
        )

    def _count(
        self, definition: IndicatorDefinition, record: DatasetRecord
    ) -> IndicatorComputationResult:
        features = _unique_features(record)
        obs = CountObservations(
            candidate_count=len(features),
            geometry_counts=_geometry_counts(features),
        )
        computation = self._engine.count(obs)
        return _from_computation(definition, record, computation, extra_warnings=())

    def _density(
        self, definition: IndicatorDefinition, record: DatasetRecord
    ) -> IndicatorComputationResult:
        area_km2 = record.scope.area_km2
        if area_km2 is None:
            return _missing_area(definition, record)
        features = _unique_features(record)
        obs = CountObservations(
            candidate_count=len(features),
            geometry_counts=_geometry_counts(features),
        )
        computation = self._engine.density(obs, area_km2=area_km2)
        return _from_computation(definition, record, computation, extra_warnings=())

    def _area_share(
        self, definition: IndicatorDefinition, record: DatasetRecord
    ) -> IndicatorComputationResult:
        area_km2 = record.scope.area_km2
        if area_km2 is None:
            return _missing_area(definition, record)
        features = _unique_features(record)
        areas: list[float] = []
        missing = 0
        invalid = 0
        for feature in features:
            geometry = feature.get("geometry")
            if not isinstance(geometry, dict):
                missing += 1
                continue
            area = geometry_area_m2(geometry)
            if area is None:
                if geometry.get("type") in {"Polygon", "MultiPolygon"}:
                    invalid += 1
                else:
                    missing += 1
                continue
            areas.append(area)
        obs = AreaObservations(
            areas_m2=tuple(areas),
            candidate_count=len(features),
            missing_count=missing,
            invalid_count=invalid,
            geometry_counts=_geometry_counts(features),
        )
        computation = self._engine.coverage_percentage(obs, area_km2=area_km2)
        extras: tuple[str, ...] = ()
        if record.scope.scope_kind == "point":
            extras = ("polygon areas are not clipped to the circular analysis cap",)
        return _from_computation(definition, record, computation, extra_warnings=extras)

    def _nearest_distance(
        self,
        definition: IndicatorDefinition,
        record: DatasetRecord,
        reference_points: tuple[GeoPoint, ...],
    ) -> IndicatorComputationResult:
        if not reference_points:
            raise IndicatorComputeError(
                f"indicator '{definition.indicator_id}' requires a reference location"
            )
        features = _unique_features(record)
        distances: list[float] = []
        missing = 0
        invalid = 0
        for ref in reference_points:
            per_feature: list[float] = []
            for feature in features:
                geometry = feature.get("geometry")
                if not isinstance(geometry, dict):
                    missing += 1
                    continue
                distance = nearest_distance_m(ref.lat, ref.lon, geometry)
                if distance is None:
                    missing += 1
                    continue
                per_feature.append(distance)
            if per_feature:
                distances.append(min(per_feature))
            elif features:
                invalid += 1
        obs = DistanceObservations(
            access_distances_m=tuple(distances),
            candidate_count=len(features),
            missing_count=missing,
            invalid_count=invalid,
            geometry_counts=_geometry_counts(features),
            reference_point_count=len(reference_points),
        )
        computation = self._engine.nearest_distance(obs)
        extras = ("distances are geodesic, not routed",)
        return _from_computation(definition, record, computation, extra_warnings=extras)

    def _length_density(
        self, definition: IndicatorDefinition, record: DatasetRecord
    ) -> IndicatorComputationResult:
        area_km2 = record.scope.area_km2
        if area_km2 is None:
            return _missing_area(definition, record)
        features = _unique_features(record)
        lengths: list[float] = []
        missing = 0
        invalid = 0
        clipped = False
        for feature in features:
            geometry = feature.get("geometry")
            if not isinstance(geometry, dict):
                missing += 1
                continue
            length, was_clipped = _scoped_length_m(geometry, record)
            if was_clipped:
                clipped = True
            if length is None:
                invalid += 1
                continue
            lengths.append(length)
        obs = LengthObservations(
            lengths_m=tuple(lengths),
            candidate_count=len(features),
            missing_count=missing,
            invalid_count=invalid,
            geometry_counts=_geometry_counts(features),
        )
        computation = self._engine.length_density(obs, area_km2=area_km2)
        extras: tuple[str, ...] = ()
        if clipped:
            extras = ("linework was clipped to the analysis scope before length was summed",)
        elif record.scope.scope_kind == "place":
            extras = ("named-place scope has no clip polygon; full retrieved length is used",)
        return _from_computation(definition, record, computation, extra_warnings=extras)

    def _shannon_entropy(
        self, definition: IndicatorDefinition, record: DatasetRecord
    ) -> IndicatorComputationResult:
        spec = _osm_feature_spec(definition)
        features = _unique_features(record)
        counts, missing, unknown, multi_match = categorize_features(features, spec)
        obs = CategoryObservations(
            category_counts=tuple(counts[category.id] for category in spec.categories),
            candidate_count=len(features),
            missing_count=missing,
            unknown_count=unknown,
            multi_match_count=multi_match,
            geometry_counts=_geometry_counts(features),
        )
        log_base = definition.parameters.get("log_base")
        if log_base is None:
            raise IndicatorComputeError(
                f"indicator '{definition.indicator_id}' is missing parameter 'log_base'"
            )
        computation = self._engine.shannon_entropy(obs, log_base=float(log_base))
        extras: list[str] = []
        if missing:
            extras.append(f"{missing} features had no OSM tags and were excluded from entropy")
        if unknown:
            extras.append(f"{unknown} features matched no catalog category and were excluded")
        if multi_match:
            extras.append(
                f"{multi_match} features matched multiple categories; "
                "the first catalog category was used"
            )
        return _from_computation(definition, record, computation, extra_warnings=tuple(extras))

    def _intersection_density(
        self, definition: IndicatorDefinition, record: DatasetRecord
    ) -> IndicatorComputationResult:
        area_km2 = record.scope.area_km2
        if area_km2 is None:
            return _missing_area(definition, record)
        features = _unique_features(record)
        census = count_network_intersections(features, record.scope)
        computation = self._engine.intersection_density(census, area_km2=area_km2)
        extras: tuple[str, ...] = ()
        if record.scope.scope_kind == "place":
            extras = ("named-place scope has no clip polygon; full retrieved linework is used",)
        return _from_computation(definition, record, computation, extra_warnings=extras)


def compute_indicator(
    request: IndicatorComputeRequest,
    *,
    catalog: IndicatorCatalog | None = None,
    engine: SpatialAnalyticsEngine | None = None,
) -> IndicatorComputationResult:
    """Convenience entry: catalog indicator + GeoJSON datasets → structured result."""
    return IndicatorExecutor(catalog=catalog, engine=engine).compute(request)


def _osm_record(definition: IndicatorDefinition, request: IndicatorComputeRequest) -> DatasetRecord:
    osm_ids = [
        item.requirement_id for item in definition.requirements if item.kind == "osm_features"
    ]
    if len(osm_ids) != 1:
        raise IndicatorComputeError(
            f"indicator '{definition.indicator_id}' must declare exactly one "
            "osm_features requirement for this executor"
        )
    requirement_id = osm_ids[0]
    try:
        return request.datasets[requirement_id]
    except KeyError as exc:
        raise IndicatorComputeError(f"missing dataset for requirement '{requirement_id}'") from exc


def _unique_features(record: DatasetRecord) -> list[dict[str, Any]]:
    raw = record.feature_collection.get("features", [])
    if not isinstance(raw, list):
        return []
    seen: set[tuple[object, ...]] = set()
    unique: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        props = item.get("properties")
        osm_type = props.get("osm_type") if isinstance(props, dict) else None
        osm_id = props.get("osm_id") if isinstance(props, dict) else None
        if osm_type is not None and osm_id is not None:
            key: tuple[object, ...] = ("osm", osm_type, osm_id)
        else:
            geometry = item.get("geometry")
            key = ("anon", repr(geometry))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _osm_feature_spec(definition: IndicatorDefinition) -> FeatureSpecification:
    for item in definition.requirements:
        if item.kind == "osm_features":
            if item.feature_spec is None:
                break
            return item.feature_spec
    raise IndicatorComputeError(
        f"indicator '{definition.indicator_id}' is missing an OSM feature specification"
    )


def _geometry_counts(features: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for feature in features:
        geometry = feature.get("geometry")
        kind = (
            str(geometry.get("type"))
            if isinstance(geometry, dict) and geometry.get("type")
            else "None"
        )
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _scoped_length_m(geometry: dict[str, Any], record: DatasetRecord) -> tuple[float | None, bool]:
    scope = record.scope
    if scope.scope_kind == "point" and scope.center is not None and scope.radius_m is not None:
        return clipped_geometry_length_m(
            geometry,
            center=(scope.center.lat, scope.center.lon),
            radius_m=float(scope.radius_m),
        )
    if scope.scope_kind == "bbox" and scope.bbox is not None:
        return clipped_geometry_length_m(geometry, bbox=scope.bbox)
    return geometry_length_m(geometry), False


def _dataset_refs(request: IndicatorComputeRequest) -> tuple[str, ...]:
    refs: list[str] = []
    seen: set[str] = set()
    for record in request.datasets.values():
        if record.dataset_ref in seen:
            continue
        seen.add(record.dataset_ref)
        refs.append(record.dataset_ref)
    return tuple(refs)


def _missing_area(
    definition: IndicatorDefinition, record: DatasetRecord
) -> IndicatorComputationResult:
    return _result(
        definition,
        value=None,
        observation_count=0,
        missing_count=0,
        dataset_refs=(record.dataset_ref,),
        warnings=("analysis area is unavailable for this scope",),
        status="insufficient_data",
    )


def _from_computation(
    definition: IndicatorDefinition,
    record: DatasetRecord,
    computation: Any,
    *,
    extra_warnings: tuple[str, ...],
) -> IndicatorComputationResult:
    warnings = tuple(computation.notes) + extra_warnings
    status: IndicatorComputeStatus = computation.status
    value = computation.value
    if (
        definition.min_observations > 0
        and computation.observation_count < definition.min_observations
        and definition.method_id not in {"count", "density", "intersection_density"}
    ):
        status = "insufficient_data"
        value = None
    return _result(
        definition,
        value=value,
        observation_count=computation.observation_count,
        missing_count=computation.missing_count,
        dataset_refs=(record.dataset_ref,),
        warnings=warnings,
        status=status,
    )


def _result(
    definition: IndicatorDefinition,
    *,
    value: float | None,
    observation_count: int,
    missing_count: int,
    dataset_refs: tuple[str, ...],
    warnings: tuple[str, ...],
    status: IndicatorComputeStatus,
) -> IndicatorComputationResult:
    return IndicatorComputationResult(
        indicator_id=definition.indicator_id,
        value=value,
        unit=definition.unit,
        observation_count=observation_count,
        missing_count=missing_count,
        methodology=definition.method_id,
        dataset_refs=dataset_refs,
        warnings=warnings,
        status=status,
    )
