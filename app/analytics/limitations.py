"""Deterministic limitation selection for comparison reports."""

from __future__ import annotations

from app.analytics.contracts import AnalysisResult, LimitationCode, MetricType

LIMITATION_TEXTS: dict[LimitationCode, str] = {
    "osm_completeness": (
        "OpenStreetMap completeness varies by place and feature type; absence of "
        "a mapped feature does not prove it does not exist."
    ),
    "missing_property_values": (
        "Some features were missing the numeric property required for this metric; "
        "missing values were excluded and were never treated as zero."
    ),
    "nonstandard_property_values": (
        "Some property values could not be parsed as plain numbers and were excluded."
    ),
    "count_is_not_capacity": ("Feature count is not a measure of service capacity or quality."),
    "euclidean_not_route": (
        "Distances are geodesic (straight-line on the sphere), not route or walking "
        "network distances."
    ),
    "scope_definition_affects_results": (
        "Results depend on the chosen analysis scope (radius or bounding box)."
    ),
    "polygon_quality_varies": (
        "Polygon geometry quality in OpenStreetMap varies; areas are approximate."
    ),
    "polygon_geometry_partially_available": (
        "Some features lacked polygon rings (for example centre points only) and "
        "could not contribute to area metrics."
    ),
    "overlapping_polygons": (
        "Overlapping polygons may inflate aggregated area; coverage is clamped "
        "and no polygon union is computed in this version."
    ),
    "osm_not_official_inventory": (
        "OpenStreetMap is a collaborative map, not an official inventory."
    ),
    "results_truncated_at_limit": (
        "One or more datasets were truncated at the configured feature limit. "
        "Returned counts are incomplete inventories (lower bounds), not exact totals. "
        "Do not treat a truncated count as a definitive winner."
    ),
    "property_units_undeclared": (
        "OSM tag values do not declare units; property statistics use tag units as-is."
    ),
    "single_reference_point": (
        "Access distances were computed from a single reference point per target."
    ),
}


def select_limitations(result: AnalysisResult) -> list[LimitationCode]:
    """Return only limitations whose triggering condition occurred."""
    selected: list[LimitationCode] = ["osm_completeness", "osm_not_official_inventory"]
    seen = set(selected)

    def add(code: LimitationCode) -> None:
        if code not in seen:
            seen.add(code)
            selected.append(code)

    metrics_computed: set[MetricType] = set()
    for target in result.targets:
        if target.data_provenance.truncated:
            add("results_truncated_at_limit")
        for metric in target.metrics:
            if metric.status == "computed":
                metrics_computed.add(metric.metric)
            if metric.missing_count > 0:
                add("missing_property_values")
            if metric.invalid_count > 0:
                add("nonstandard_property_values")
            if metric.unit.endswith("(OSM tag units)"):
                add("property_units_undeclared")
            if (
                metric.metric in {"total_area", "mean_area", "median_area", "coverage_percentage"}
                and metric.candidate_count > metric.observation_count
            ):
                add("polygon_geometry_partially_available")
            if metric.provenance.reference_point_count == 1 and metric.metric in {
                "nearest_distance",
                "mean_nearest_distance",
                "median_nearest_distance",
            }:
                add("single_reference_point")
            if "overlapping" in " ".join(metric.notes).lower():
                add("overlapping_polygons")
            if (
                metric.metric in {"total_area", "mean_area", "median_area", "coverage_percentage"}
                and metric.observation_count >= 2
            ):
                add("overlapping_polygons")
                add("polygon_quality_varies")

    primary_metrics = {m.metric for t in result.targets for m in t.metrics if m.role == "primary"}
    if primary_metrics & {"count", "density"}:
        add("count_is_not_capacity")
    if metrics_computed & {
        "nearest_distance",
        "mean_nearest_distance",
        "median_nearest_distance",
    }:
        add("euclidean_not_route")
    if metrics_computed & {"density", "coverage_percentage"}:
        add("scope_definition_affects_results")

    return selected


def limitation_text(code: LimitationCode) -> str:
    return LIMITATION_TEXTS[code]
