"""Deterministic ranking of indicator values across analysis targets."""

from __future__ import annotations

from app.analytics.catalog import get_metric_definition
from app.analytics.contracts import (
    ComparisonResult,
    ComparisonVerdict,
    MetricComparison,
    MetricType,
    TargetMetricValue,
)
from app.indicators.contracts import IndicatorComputationResult, IndicatorDefinition


def plan_metric_type(method_id: str) -> MetricType:
    """Closest Metric Catalog primitive for an indicator method (plan/trace only)."""
    mapping: dict[str, MetricType] = {
        "count": "count",
        "density": "density",
        "area_share": "coverage_percentage",
        "nearest_distance": "nearest_distance",
        "length_density": "density",
        "intersection_density": "density",
        "shannon_entropy": "standard_deviation",
        "distance_decay_sum": "nearest_distance",
        "statistic": "mean",
        "category_share": "ratio",
    }
    return mapping.get(method_id, "count")


def compare_indicator_targets(
    definition: IndicatorDefinition,
    rows: tuple[tuple[str, str, IndicatorComputationResult], ...],
    *,
    analysis_type: str,
) -> ComparisonResult:
    """Rank computed indicator values. Does not invent numbers."""
    metric = plan_metric_type(definition.method_id)
    catalog = get_metric_definition(metric)
    ranking = [
        TargetMetricValue(
            target_id=target_id,
            label=label,
            value=result.value,
            status=(
                "computed"
                if result.status == "computed"
                else "insufficient_data"
                if result.status == "insufficient_data"
                else "not_applicable"
            ),
            observation_count=result.observation_count,
        )
        for target_id, label, result in rows
    ]
    computed = [item for item in ranking if item.status == "computed" and item.value is not None]
    rest = [item for item in ranking if item not in computed]
    reverse = definition.direction != "lower_is_better"
    computed_sorted = sorted(computed, key=lambda item: float(item.value or 0.0), reverse=reverse)
    ordered = computed_sorted + rest

    if analysis_type != "comparison" or len(computed_sorted) < 2:
        statement = _single_statement(definition, ordered)
        confidence = "insufficient_data" if len(computed_sorted) < 1 else "clear"
        verdict: ComparisonVerdict = "insufficient_data" if len(computed_sorted) < 2 else "higher"
        return ComparisonResult(
            analysis_type="comparison" if analysis_type == "comparison" else "single_target",
            primary=MetricComparison(
                metric=metric,
                role="primary",
                label=definition.display_label,
                unit=definition.unit,
                direction=definition.direction,
                ranking=ordered,
                verdict=verdict if analysis_type == "comparison" else "not_comparable",
                leading_target_id=computed_sorted[0].target_id if computed_sorted else None,
                difference_unit=definition.unit,
            ),
            overall_statement=statement,
            overall_confidence=confidence,
        )

    leader = computed_sorted[0]
    runner = computed_sorted[1]
    abs_diff = abs(float(leader.value or 0.0) - float(runner.value or 0.0))
    baseline = abs(float(runner.value or 0.0))
    rel = (abs_diff / baseline * 100.0) if baseline > 1e-12 else None
    epsilon = catalog.tie_relative_epsilon
    close = rel is not None and rel <= max(epsilon * 100.0, 5.0)
    if abs_diff <= 1e-12 or close:
        verdict = "tie"
        statement = (
            f"{leader.label} and {runner.label} are effectively tied on "
            f"{definition.display_label} ({_fmt(leader.value)} vs {_fmt(runner.value)} "
            f"{definition.unit})."
        )
        confidence = "close"
        preferred = None
    else:
        verb = "higher" if reverse else "lower"
        verdict = "higher" if reverse else "lower"
        statement = (
            f"{leader.label} has the {verb} {definition.display_label} "
            f"({_fmt(leader.value)} {definition.unit}) than {runner.label} "
            f"({_fmt(runner.value)} {definition.unit})."
        )
        if len(computed_sorted) > 2:
            others = ", ".join(f"{item.label}={_fmt(item.value)}" for item in computed_sorted[2:])
            statement = f"{statement} Remaining targets: {others}."
        confidence = "clear"
        preferred = leader.target_id

    return ComparisonResult(
        analysis_type="comparison",
        primary=MetricComparison(
            metric=metric,
            role="primary",
            label=definition.display_label,
            unit=definition.unit,
            direction=definition.direction,
            ranking=ordered,
            verdict=verdict,
            leading_target_id=leader.target_id,
            preferred_target_id=preferred,
            absolute_difference=abs_diff,
            relative_difference_percent=rel,
            difference_unit=definition.unit,
        ),
        overall_statement=statement,
        overall_confidence=confidence,
    )


def _single_statement(definition: IndicatorDefinition, ranking: list[TargetMetricValue]) -> str:
    if not ranking:
        return f"No {definition.display_label} value could be computed."
    first = ranking[0]
    if first.value is None:
        return f"{definition.display_label} could not be computed for {first.label}."
    return f"{first.label}: {_fmt(first.value)} {definition.unit} ({definition.display_label})."


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value - round(value)) < 1e-9:
        return f"{round(value):g}"
    return f"{value:.4g}"
