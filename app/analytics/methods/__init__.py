"""Closed method registry — the only computations an indicator may bind to.

Indicators are data. Methods are code. A YAML definition may only name a
``method_id`` that exists here; formula text is never parsed or evaluated.

Each method declares which existing :class:`MetricType` primitives it uses.
That is the explicit relationship between the Indicator Catalog and the
Metric Catalog:

    Indicator  →  MethodSpec.method_id  →  MetricType primitive(s)

Phase 1 registers every method identifier the catalog needs. Methods with
``implemented=False`` are declared so definitions can load; the engine does
not yet compute them. Do not invent a parallel metric type enum here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from app.analytics.contracts import MetricType

MethodId = Literal[
    "count",
    "density",
    "area_share",
    "category_share",
    "shannon_entropy",
    "nearest_distance",
    "distance_decay_sum",
    "line_length",
    "length_density",
    "intersection_density",
    "statistic",
]


class MethodParameterSpec(BaseModel):
    """Allowed numeric parameter for one method (validated at catalog load)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=32, pattern=r"^[a-z][a-z0-9_]*$")
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: float | None = None


class MethodSpec(BaseModel):
    """One catalog-addressable computation, bound to Metric Catalog primitives."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_id: MethodId
    # Existing Metric Catalog entries this method is implemented with, or will
    # be implemented with. Empty when the method is not a MetricType wrapper.
    metric_primitives: tuple[MetricType, ...]
    implemented: bool
    implementation_id: str
    allowed_parameters: tuple[MethodParameterSpec, ...] = ()
    notes: str = ""


def _spec(
    method_id: MethodId,
    *,
    primitives: tuple[MetricType, ...] = (),
    implemented: bool,
    engine_method: str,
    parameters: tuple[MethodParameterSpec, ...] = (),
    notes: str = "",
) -> MethodSpec:
    return MethodSpec(
        method_id=method_id,
        metric_primitives=primitives,
        implemented=implemented,
        implementation_id=f"SpatialAnalyticsEngine.{engine_method}",
        allowed_parameters=parameters,
        notes=notes,
    )


METHOD_REGISTRY: Mapping[MethodId, MethodSpec] = {
    "count": _spec(
        "count",
        primitives=("count",),
        implemented=True,
        engine_method="count",
        notes="Direct Metric Catalog primitive: feature count.",
    ),
    "density": _spec(
        "density",
        primitives=("density",),
        implemented=True,
        engine_method="density",
        notes="Direct Metric Catalog primitive: features per km².",
    ),
    "area_share": _spec(
        "area_share",
        primitives=("coverage_percentage",),
        implemented=True,
        engine_method="coverage_percentage",
        notes=(
            "Indicator-level name for coverage_percentage: polygon area divided "
            "by analysis-area, expressed as a percent."
        ),
    ),
    "category_share": _spec(
        "category_share",
        implemented=False,
        engine_method="category_share",
        notes="Share of one tag category within a partitioned feature set. Not implemented.",
    ),
    "shannon_entropy": _spec(
        "shannon_entropy",
        implemented=True,
        engine_method="shannon_entropy",
        parameters=(MethodParameterSpec(name="log_base", exclusive_minimum=1.0, maximum=10.0),),
        notes=(
            "Shannon entropy of catalog tag-category count shares. Missing and "
            "unknown tags are excluded from N. Not Pielou-normalised."
        ),
    ),
    "nearest_distance": _spec(
        "nearest_distance",
        primitives=("nearest_distance",),
        implemented=True,
        engine_method="nearest_distance",
        notes="Direct Metric Catalog primitive: geodesic nearest-feature metres.",
    ),
    "distance_decay_sum": _spec(
        "distance_decay_sum",
        implemented=False,
        engine_method="distance_decay_sum",
        parameters=(MethodParameterSpec(name="decay_beta", exclusive_minimum=0.0, maximum=1.0),),
        notes=(
            "Hansen-style distance-weighted opportunity sum. Not implemented: "
            "the catalog formula does not pin whether distance is to a polygon "
            "boundary, centroid, or vertex, nor whether area is clipped."
        ),
    ),
    "line_length": _spec(
        "line_length",
        implemented=True,
        engine_method="line_length",
        notes="Sum of LineString geodesic lengths (haversine polyline).",
    ),
    "length_density": _spec(
        "length_density",
        implemented=True,
        engine_method="length_density",
        notes="Total geodesic line length (km) divided by analysis area (km²).",
    ),
    "intersection_density": _spec(
        "intersection_density",
        implemented=True,
        engine_method="intersection_density",
        notes=(
            "Degree ≥ 3 vertices per km² on a snapped, undirected road graph. "
            "Polyline bends and geometric crossings without a shared vertex "
            "are not intersections."
        ),
    ),
    "statistic": _spec(
        "statistic",
        primitives=("sum", "mean", "median", "standard_deviation", "min", "max"),
        implemented=True,
        engine_method="mean",
        notes="Delegates to the existing numeric-property statistic family.",
    ),
}


def get_method_spec(method_id: MethodId) -> MethodSpec:
    """Return the registry entry for ``method_id``."""
    return METHOD_REGISTRY[method_id]


def assert_method_registry_complete() -> None:
    """Raise if the registry does not cover every MethodId."""
    expected = set(get_args(MethodId))
    actual = set(METHOD_REGISTRY)
    if expected != actual:
        raise AssertionError(
            f"method registry mismatch: missing={expected - actual} extra={actual - expected}"
        )


assert_method_registry_complete()
