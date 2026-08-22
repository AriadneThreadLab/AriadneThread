"""Deterministic Spatial Analytics Engine — source of truth for all numbers."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from app.analytics.catalog import (
    METRIC_CATALOG_VERSION,
    get_metric_definition,
    resolve_unit,
)
from app.analytics.contracts import (
    AnalysisPlan,
    AnalysisResult,
    CalculationProvenance,
    DataProvenance,
    KnowledgeSourceRef,
    MetricDirection,
    MetricRequest,
    MetricResult,
    MetricStatus,
    MetricType,
    TargetAnalysisResult,
)
from app.analytics.datasets import DatasetRecord, DatasetRegistry
from app.analytics.geometry import (
    GEOMETRY_STRATEGY_CAP,
    GEOMETRY_STRATEGY_HAVERSINE,
    GEOMETRY_STRATEGY_SPHERICAL_AREA,
)
from app.analytics.limitations import select_limitations
from app.analytics.observations import (
    AreaObservations,
    CountObservations,
    DistanceObservations,
    NumericObservations,
    extract_areas,
    extract_count,
    extract_distances,
    extract_numeric,
)
from app.analytics.rules import RULESET_VERSION


@dataclass(frozen=True, slots=True)
class MetricComputation:
    """Raw engine output before provenance attachment."""

    value: float | None
    status: MetricStatus
    observation_count: int
    candidate_count: int
    missing_count: int
    invalid_count: int
    notes: tuple[str, ...] = ()
    geometry_strategy: str | None = None


SUPPORTED_RATIO_PAIRS: frozenset[tuple[MetricType, MetricType]] = frozenset(
    {
        ("count", "total_area"),
        ("total_area", "count"),
        ("sum", "count"),
    }
)


class SpatialAnalyticsEngine:
    """Pure, synchronous metric implementations."""

    def count(self, obs: CountObservations) -> MetricComputation:
        return MetricComputation(
            value=float(obs.candidate_count),
            status="computed",
            observation_count=obs.candidate_count,
            candidate_count=obs.candidate_count,
            missing_count=0,
            invalid_count=0,
        )

    def density(
        self,
        obs: CountObservations,
        *,
        area_km2: float,
    ) -> MetricComputation:
        if area_km2 <= 0:
            return MetricComputation(
                value=None,
                status="insufficient_data",
                observation_count=obs.candidate_count,
                candidate_count=obs.candidate_count,
                missing_count=0,
                invalid_count=0,
                notes=("analysis area is zero or unavailable",),
                geometry_strategy=GEOMETRY_STRATEGY_CAP,
            )
        return MetricComputation(
            value=obs.candidate_count / area_km2,
            status="computed",
            observation_count=obs.candidate_count,
            candidate_count=obs.candidate_count,
            missing_count=0,
            invalid_count=0,
            geometry_strategy=GEOMETRY_STRATEGY_CAP,
        )

    def sum(self, obs: NumericObservations) -> MetricComputation:
        if not obs.values:
            return self._insufficient_numeric(obs)
        return MetricComputation(
            value=math.fsum(obs.values),
            status="computed",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
        )

    def mean(self, obs: NumericObservations) -> MetricComputation:
        if not obs.values:
            return self._insufficient_numeric(obs)
        return MetricComputation(
            value=statistics.fmean(obs.values),
            status="computed",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
        )

    def median(self, obs: NumericObservations) -> MetricComputation:
        if not obs.values:
            return self._insufficient_numeric(obs)
        return MetricComputation(
            value=float(statistics.median(obs.values)),
            status="computed",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
        )

    def standard_deviation(self, obs: NumericObservations) -> MetricComputation:
        """Sample standard deviation (Bessel-corrected)."""
        if len(obs.values) < 2:
            return MetricComputation(
                value=None,
                status="insufficient_data",
                observation_count=obs.observation_count,
                candidate_count=obs.candidate_count,
                missing_count=obs.missing_count,
                invalid_count=obs.invalid_count,
                notes=("sample standard deviation requires at least two observations",),
            )
        return MetricComputation(
            value=statistics.stdev(obs.values),
            status="computed",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
        )

    def minimum(self, obs: NumericObservations) -> MetricComputation:
        if not obs.values:
            return self._insufficient_numeric(obs)
        return MetricComputation(
            value=min(obs.values),
            status="computed",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
        )

    def maximum(self, obs: NumericObservations) -> MetricComputation:
        if not obs.values:
            return self._insufficient_numeric(obs)
        return MetricComputation(
            value=max(obs.values),
            status="computed",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
        )

    def total_area(self, obs: AreaObservations) -> MetricComputation:
        if not obs.areas_m2:
            return self._insufficient_area(obs)
        return MetricComputation(
            value=math.fsum(obs.areas_m2),
            status="computed",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
            geometry_strategy=GEOMETRY_STRATEGY_SPHERICAL_AREA,
        )

    def mean_area(self, obs: AreaObservations) -> MetricComputation:
        if not obs.areas_m2:
            return self._insufficient_area(obs)
        return MetricComputation(
            value=statistics.fmean(obs.areas_m2),
            status="computed",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
            geometry_strategy=GEOMETRY_STRATEGY_SPHERICAL_AREA,
        )

    def median_area(self, obs: AreaObservations) -> MetricComputation:
        if not obs.areas_m2:
            return self._insufficient_area(obs)
        return MetricComputation(
            value=float(statistics.median(obs.areas_m2)),
            status="computed",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
            geometry_strategy=GEOMETRY_STRATEGY_SPHERICAL_AREA,
        )

    def nearest_distance(self, obs: DistanceObservations) -> MetricComputation:
        if not obs.access_distances_m:
            return self._insufficient_distance(obs)
        return MetricComputation(
            value=min(obs.access_distances_m),
            status="computed",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
            geometry_strategy=GEOMETRY_STRATEGY_HAVERSINE,
        )

    def mean_nearest_distance(self, obs: DistanceObservations) -> MetricComputation:
        if len(obs.access_distances_m) < 2:
            return MetricComputation(
                value=None,
                status="insufficient_data",
                observation_count=obs.observation_count,
                candidate_count=obs.candidate_count,
                missing_count=obs.missing_count,
                invalid_count=obs.invalid_count,
                notes=("mean nearest distance requires at least two reference points",),
                geometry_strategy=GEOMETRY_STRATEGY_HAVERSINE,
            )
        return MetricComputation(
            value=statistics.fmean(obs.access_distances_m),
            status="computed",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
            geometry_strategy=GEOMETRY_STRATEGY_HAVERSINE,
        )

    def median_nearest_distance(self, obs: DistanceObservations) -> MetricComputation:
        if len(obs.access_distances_m) < 2:
            return MetricComputation(
                value=None,
                status="insufficient_data",
                observation_count=obs.observation_count,
                candidate_count=obs.candidate_count,
                missing_count=obs.missing_count,
                invalid_count=obs.invalid_count,
                notes=("median nearest distance requires at least two reference points",),
                geometry_strategy=GEOMETRY_STRATEGY_HAVERSINE,
            )
        return MetricComputation(
            value=float(statistics.median(obs.access_distances_m)),
            status="computed",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
            geometry_strategy=GEOMETRY_STRATEGY_HAVERSINE,
        )

    def coverage_percentage(
        self,
        obs: AreaObservations,
        *,
        area_km2: float,
    ) -> MetricComputation:
        if not obs.areas_m2 or area_km2 <= 0:
            return MetricComputation(
                value=None,
                status="insufficient_data",
                observation_count=obs.observation_count,
                candidate_count=obs.candidate_count,
                missing_count=obs.missing_count,
                invalid_count=obs.invalid_count,
                geometry_strategy=GEOMETRY_STRATEGY_SPHERICAL_AREA,
            )
        area_m2 = area_km2 * 1_000_000.0
        raw = 100.0 * math.fsum(obs.areas_m2) / area_m2
        notes: list[str] = []
        value = raw
        if raw > 100.0:
            value = 100.0
            notes.append(f"overlapping_polygons: pre-clamp coverage was {raw:.3f}%")
        return MetricComputation(
            value=value,
            status="computed",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
            notes=tuple(notes),
            geometry_strategy=GEOMETRY_STRATEGY_SPHERICAL_AREA,
        )

    def ratio(
        self,
        numerator: MetricComputation,
        denominator: MetricComputation,
        *,
        candidate_count: int,
    ) -> MetricComputation:
        if (
            numerator.status != "computed"
            or denominator.status != "computed"
            or numerator.value is None
            or denominator.value is None
        ):
            return MetricComputation(
                value=None,
                status="insufficient_data",
                observation_count=0,
                candidate_count=candidate_count,
                missing_count=0,
                invalid_count=0,
            )
        if denominator.value == 0:
            return MetricComputation(
                value=None,
                status="insufficient_data",
                observation_count=numerator.observation_count,
                candidate_count=candidate_count,
                missing_count=0,
                invalid_count=0,
                notes=("zero denominator",),
            )
        return MetricComputation(
            value=numerator.value / denominator.value,
            status="computed",
            observation_count=numerator.observation_count,
            candidate_count=candidate_count,
            missing_count=0,
            invalid_count=0,
        )

    def compute_plan(
        self,
        plan: AnalysisPlan,
        datasets: DatasetRegistry,
        *,
        grounding_sources: list[KnowledgeSourceRef] | None = None,
    ) -> AnalysisResult:
        """Compute all metrics for a validated plan."""
        targets: list[TargetAnalysisResult] = []
        warnings: list[str] = []
        sources = list(grounding_sources or [])[:3]

        for target in plan.targets:
            record = datasets.get(target.dataset_ref)
            metric_results: list[MetricResult] = []
            primary_obs_counts = (0, 0, 0)

            for metric_req in plan.metrics:
                metric_result = self._compute_metric(metric_req, target, record)
                metric_results.append(metric_result)
                if metric_req.role == "primary":
                    primary_obs_counts = (
                        metric_result.observation_count,
                        metric_result.missing_count,
                        metric_result.invalid_count,
                    )
                if metric_result.notes:
                    warnings.extend(metric_result.notes)

            count_obs = extract_count(record)
            obs_count, miss_count, inv_count = primary_obs_counts

            data_prov = DataProvenance(
                feature_concept=plan.feature_concept,
                resolved_tags=list(record.resolved_tags),
                grounding_sources=sources,
                dataset_ref=record.dataset_ref,
                target_id=target.target_id,
                analysis_scope=record.scope.summary,
                scope_kind=record.scope.scope_kind,
                analysis_area_km2=record.scope.area_km2,
                retrieved_feature_count=record.feature_count,
                valid_observation_count=obs_count,
                missing_observation_count=miss_count,
                invalid_observation_count=inv_count,
                geometry_counts=dict(count_obs.geometry_counts),
                truncated=record.truncated,
                effective_limit=record.effective_limit,
                limit_reached=bool(record.truncated),
                retrieved_at=record.retrieved_at.isoformat(),
            )
            if record.truncated:
                warnings.append(
                    f"{target.label}: returned_count={record.feature_count} reached "
                    f"effective_limit={record.effective_limit}; "
                    "not a complete inventory — treat count/density as lower bounds."
                )
                metric_results = [
                    metric.model_copy(
                        update={
                            "notes": [
                                *metric.notes,
                                "results truncated at effective_limit; value is not an exact total",
                            ]
                        }
                    )
                    if metric.role == "primary"
                    else metric
                    for metric in metric_results
                ]
            targets.append(
                TargetAnalysisResult(
                    target_id=target.target_id,
                    label=target.label,
                    dataset_ref=target.dataset_ref,
                    data_provenance=data_prov,
                    metrics=metric_results,
                )
            )

        result = AnalysisResult(
            analysis_type=plan.analysis_type,
            feature_concept=plan.feature_concept,
            comparison_goal=plan.comparison_goal,
            targets=targets,
            metric_catalog_version=METRIC_CATALOG_VERSION,
            ruleset_version=RULESET_VERSION,
            warnings=list(dict.fromkeys(warnings)),
            limitations=[],
        )
        return result.model_copy(update={"limitations": select_limitations(result)})

    def _compute_metric(
        self,
        metric_req: MetricRequest,
        target: Any,
        record: DatasetRecord,
    ) -> MetricResult:
        definition = get_metric_definition(metric_req.metric)
        direction: MetricDirection = metric_req.direction or definition.default_direction
        unit = resolve_unit(definition, metric_req.property_key)
        ref_count = len(target.reference_points)

        if metric_req.metric == "ratio":
            assert metric_req.ratio is not None
            num_req = MetricRequest(
                metric=metric_req.ratio.numerator,
                role="supporting",
                inferred_goal=metric_req.inferred_goal,
                property_key=(
                    metric_req.property_key
                    if metric_req.ratio.numerator
                    in {"sum", "mean", "median", "standard_deviation", "min", "max"}
                    else None
                ),
                user_explicit=metric_req.user_explicit,
            )
            den_req = MetricRequest(
                metric=metric_req.ratio.denominator,
                role="supporting",
                inferred_goal=metric_req.inferred_goal,
                property_key=None,
                user_explicit=metric_req.user_explicit,
            )
            # For sum/count ratio, denominator is count (no property).
            num_comp = self._raw_compute(num_req, target, record)
            den_comp = self._raw_compute(den_req, target, record)
            computation = self.ratio(
                num_comp,
                den_comp,
                candidate_count=record.feature_count,
            )
        else:
            computation = self._raw_compute(metric_req, target, record)

        provenance = CalculationProvenance(
            metric=metric_req.metric,
            dataset_ref=record.dataset_ref,
            target_ref=target.target_id,
            implementation_id=definition.implementation_id,
            geometry_strategy=computation.geometry_strategy,
            observation_count=computation.observation_count,
            candidate_count=computation.candidate_count,
            missing_count=computation.missing_count,
            invalid_count=computation.invalid_count,
            unit=unit,
            property_key=metric_req.property_key,
            analysis_area_km2=record.scope.area_km2,
            reference_point_count=ref_count,
            metric_catalog_version=METRIC_CATALOG_VERSION,
        )
        return MetricResult(
            metric=metric_req.metric,
            role=metric_req.role,
            label=definition.display_label,
            status=computation.status,
            value=computation.value,
            unit=unit,
            direction=direction,
            observation_count=computation.observation_count,
            candidate_count=computation.candidate_count,
            missing_count=computation.missing_count,
            invalid_count=computation.invalid_count,
            property_key=metric_req.property_key,
            provenance=provenance,
            notes=list(computation.notes),
        )

    def _raw_compute(
        self,
        metric_req: MetricRequest,
        target: Any,
        record: DatasetRecord,
    ) -> MetricComputation:
        metric = metric_req.metric
        if metric == "count":
            return self.count(extract_count(record))
        if metric == "density":
            area = record.scope.area_km2 or 0.0
            return self.density(extract_count(record), area_km2=area)
        if metric in {"sum", "mean", "median", "standard_deviation", "min", "max"}:
            assert metric_req.property_key is not None
            numeric_obs = extract_numeric(record, metric_req.property_key)
            method = {
                "sum": self.sum,
                "mean": self.mean,
                "median": self.median,
                "standard_deviation": self.standard_deviation,
                "min": self.minimum,
                "max": self.maximum,
            }[metric]
            return method(numeric_obs)
        if metric in {"total_area", "mean_area", "median_area"}:
            area_obs = extract_areas(record)
            return {
                "total_area": self.total_area,
                "mean_area": self.mean_area,
                "median_area": self.median_area,
            }[metric](area_obs)
        if metric == "coverage_percentage":
            return self.coverage_percentage(
                extract_areas(record),
                area_km2=record.scope.area_km2 or 0.0,
            )
        if metric in {
            "nearest_distance",
            "mean_nearest_distance",
            "median_nearest_distance",
        }:
            distance_obs = extract_distances(record, list(target.reference_points))
            return {
                "nearest_distance": self.nearest_distance,
                "mean_nearest_distance": self.mean_nearest_distance,
                "median_nearest_distance": self.median_nearest_distance,
            }[metric](distance_obs)
        return MetricComputation(
            value=None,
            status="not_applicable",
            observation_count=0,
            candidate_count=record.feature_count,
            missing_count=0,
            invalid_count=0,
        )

    @staticmethod
    def _insufficient_numeric(obs: NumericObservations) -> MetricComputation:
        return MetricComputation(
            value=None,
            status="insufficient_data",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
        )

    @staticmethod
    def _insufficient_area(obs: AreaObservations) -> MetricComputation:
        return MetricComputation(
            value=None,
            status="insufficient_data",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
            geometry_strategy=GEOMETRY_STRATEGY_SPHERICAL_AREA,
        )

    @staticmethod
    def _insufficient_distance(obs: DistanceObservations) -> MetricComputation:
        return MetricComputation(
            value=None,
            status="insufficient_data",
            observation_count=obs.observation_count,
            candidate_count=obs.candidate_count,
            missing_count=obs.missing_count,
            invalid_count=obs.invalid_count,
            geometry_strategy=GEOMETRY_STRATEGY_HAVERSINE,
        )
