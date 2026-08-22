"""Bounded Metric Catalog — the only metrics the engine may compute."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict

from app.analytics.contracts import (
    AnalysisGoal,
    MetricDirection,
    MetricType,
)

METRIC_CATALOG_VERSION = "metric-catalog-1"

GeometryRequirement = Literal["any", "polygon", "positioned"]
MissingValueBehaviour = Literal["excluded_never_zero", "not_applicable"]
MetricFamily = Literal["count", "statistic", "area", "distance", "share"]


class MetricDefinition(BaseModel):
    """Catalog metadata for one supported metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: MetricType
    display_label: str
    family: MetricFamily
    requires_numeric_property: bool
    required_geometry: GeometryRequirement
    requires_reference_points: bool
    requires_analysis_area: bool
    min_observations: int
    zero_observations_allowed: bool
    missing_value_behaviour: MissingValueBehaviour
    unit: str
    display_precision: int
    valid_goals: tuple[AnalysisGoal, ...]
    default_direction: MetricDirection
    implementation_id: str
    tie_relative_epsilon: float


def _def(
    metric: MetricType,
    *,
    display_label: str,
    family: MetricFamily,
    requires_numeric_property: bool = False,
    required_geometry: GeometryRequirement = "any",
    requires_reference_points: bool = False,
    requires_analysis_area: bool = False,
    min_observations: int = 0,
    zero_observations_allowed: bool = False,
    unit: str,
    display_precision: int,
    valid_goals: tuple[AnalysisGoal, ...],
    default_direction: MetricDirection,
    method: str,
    tie_relative_epsilon: float = 0.005,
) -> MetricDefinition:
    return MetricDefinition(
        metric=metric,
        display_label=display_label,
        family=family,
        requires_numeric_property=requires_numeric_property,
        required_geometry=required_geometry,
        requires_reference_points=requires_reference_points,
        requires_analysis_area=requires_analysis_area,
        min_observations=min_observations,
        zero_observations_allowed=zero_observations_allowed,
        missing_value_behaviour=("not_applicable" if family == "count" else "excluded_never_zero"),
        unit=unit,
        display_precision=display_precision,
        valid_goals=valid_goals,
        default_direction=default_direction,
        implementation_id=f"SpatialAnalyticsEngine.{method}",
        tie_relative_epsilon=tie_relative_epsilon,
    )


METRIC_CATALOG: Mapping[MetricType, MetricDefinition] = {
    "count": _def(
        "count",
        display_label="Feature count",
        family="count",
        zero_observations_allowed=True,
        unit="features",
        display_precision=0,
        valid_goals=("abundance",),
        default_direction="higher_is_better",
        method="count",
        tie_relative_epsilon=0.0,
    ),
    "density": _def(
        "density",
        display_label="Feature density",
        family="count",
        requires_analysis_area=True,
        zero_observations_allowed=True,
        unit="features_per_km2",
        display_precision=1,
        valid_goals=("concentration",),
        default_direction="higher_is_better",
        method="density",
    ),
    "sum": _def(
        "sum",
        display_label="Sum",
        family="statistic",
        requires_numeric_property=True,
        min_observations=1,
        unit="property_unit",
        display_precision=2,
        valid_goals=("total_provision",),
        default_direction="higher_is_better",
        method="sum",
    ),
    "mean": _def(
        "mean",
        display_label="Mean",
        family="statistic",
        requires_numeric_property=True,
        min_observations=1,
        unit="property_unit",
        display_precision=2,
        valid_goals=("typical_value",),
        default_direction="neutral",
        method="mean",
    ),
    "median": _def(
        "median",
        display_label="Median",
        family="statistic",
        requires_numeric_property=True,
        min_observations=1,
        unit="property_unit",
        display_precision=2,
        valid_goals=("typical_value",),
        default_direction="neutral",
        method="median",
    ),
    "standard_deviation": _def(
        "standard_deviation",
        display_label="Standard deviation",
        family="statistic",
        requires_numeric_property=True,
        min_observations=2,
        unit="property_unit",
        display_precision=2,
        valid_goals=("variability",),
        default_direction="neutral",
        method="standard_deviation",
    ),
    "min": _def(
        "min",
        display_label="Minimum",
        family="statistic",
        requires_numeric_property=True,
        min_observations=1,
        unit="property_unit",
        display_precision=2,
        valid_goals=(
            "abundance",
            "concentration",
            "accessibility",
            "total_provision",
            "typical_value",
            "variability",
            "coverage",
            "relative_share",
        ),
        default_direction="neutral",
        method="minimum",
    ),
    "max": _def(
        "max",
        display_label="Maximum",
        family="statistic",
        requires_numeric_property=True,
        min_observations=1,
        unit="property_unit",
        display_precision=2,
        valid_goals=(
            "abundance",
            "concentration",
            "accessibility",
            "total_provision",
            "typical_value",
            "variability",
            "coverage",
            "relative_share",
        ),
        default_direction="neutral",
        method="maximum",
    ),
    "ratio": _def(
        "ratio",
        display_label="Ratio",
        family="share",
        min_observations=1,
        unit="ratio",
        display_precision=3,
        valid_goals=("relative_share",),
        default_direction="neutral",
        method="ratio",
    ),
    "total_area": _def(
        "total_area",
        display_label="Total area",
        family="area",
        required_geometry="polygon",
        min_observations=1,
        unit="m2",
        display_precision=0,
        valid_goals=("total_provision",),
        default_direction="higher_is_better",
        method="total_area",
    ),
    "mean_area": _def(
        "mean_area",
        display_label="Mean area",
        family="area",
        required_geometry="polygon",
        min_observations=1,
        unit="m2",
        display_precision=0,
        valid_goals=("typical_value",),
        default_direction="neutral",
        method="mean_area",
    ),
    "median_area": _def(
        "median_area",
        display_label="Median area",
        family="area",
        required_geometry="polygon",
        min_observations=1,
        unit="m2",
        display_precision=0,
        valid_goals=("typical_value",),
        default_direction="neutral",
        method="median_area",
    ),
    "nearest_distance": _def(
        "nearest_distance",
        display_label="Nearest distance",
        family="distance",
        required_geometry="positioned",
        requires_reference_points=True,
        min_observations=1,
        unit="m",
        display_precision=0,
        valid_goals=("accessibility",),
        default_direction="lower_is_better",
        method="nearest_distance",
    ),
    "mean_nearest_distance": _def(
        "mean_nearest_distance",
        display_label="Mean nearest distance",
        family="distance",
        required_geometry="positioned",
        requires_reference_points=True,
        min_observations=2,
        unit="m",
        display_precision=0,
        valid_goals=("accessibility", "typical_value"),
        default_direction="lower_is_better",
        method="mean_nearest_distance",
    ),
    "median_nearest_distance": _def(
        "median_nearest_distance",
        display_label="Median nearest distance",
        family="distance",
        required_geometry="positioned",
        requires_reference_points=True,
        min_observations=2,
        unit="m",
        display_precision=0,
        valid_goals=("accessibility", "typical_value"),
        default_direction="lower_is_better",
        method="median_nearest_distance",
    ),
    "coverage_percentage": _def(
        "coverage_percentage",
        display_label="Coverage percentage",
        family="share",
        required_geometry="polygon",
        requires_analysis_area=True,
        min_observations=1,
        unit="percent",
        display_precision=1,
        valid_goals=("coverage",),
        default_direction="higher_is_better",
        method="coverage_percentage",
    ),
}


def get_metric_definition(metric: MetricType) -> MetricDefinition:
    """Return the catalog entry for ``metric``."""
    return METRIC_CATALOG[metric]


def resolve_unit(definition: MetricDefinition, property_key: str | None) -> str:
    """Resolve display/storage unit, including property-tag placeholders."""
    if definition.unit != "property_unit":
        return definition.unit
    if property_key:
        return f"{property_key} (OSM tag units)"
    return "property_unit"


def assert_catalog_complete() -> None:
    """Raise if the catalog does not cover every MetricType."""
    expected = set(get_args(MetricType))
    actual = set(METRIC_CATALOG)
    if expected != actual:
        raise AssertionError(
            f"metric catalog mismatch: missing={expected - actual} extra={actual - expected}"
        )


assert_catalog_complete()
