"""Mandatory metric revalidation for analytical follow-ups."""

from __future__ import annotations

from app.analytics.catalog import METRIC_CATALOG, METRIC_CATALOG_VERSION
from app.analytics.contracts import AnalysisGoal, MetricType
from app.analytics.rules import RULESET_VERSION, SEMANTIC_RULES
from app.execution_memory.contracts import MetricRevalidation, MetricSnapshot


def permitted_metrics_for_goal(goal: AnalysisGoal) -> tuple[MetricType, ...]:
    permitted: list[MetricType] = []
    for rule in SEMANTIC_RULES:
        if rule.supporting_only:
            continue
        if rule.goal == goal or rule.any_goal:
            for metric in rule.permitted_metrics:
                if metric not in permitted:
                    permitted.append(metric)
    return tuple(permitted)


def default_metric_for_goal(goal: AnalysisGoal) -> MetricType:
    mapping: dict[AnalysisGoal, MetricType] = {
        "abundance": "count",
        "concentration": "density",
        "accessibility": "nearest_distance",
        "total_provision": "total_area",
        "typical_value": "median_area",
        "variability": "standard_deviation",
        "coverage": "coverage_percentage",
        "relative_share": "ratio",
    }
    return mapping[goal]


def revalidate_metric(
    *,
    previous: MetricSnapshot,
    new_goal: AnalysisGoal,
) -> MetricRevalidation:
    """Never blindly reuse the previous metric."""
    previous_metric = previous.metric
    if previous_metric is None:
        replacement = default_metric_for_goal(new_goal)
        return MetricRevalidation(
            previous_metric=None,
            new_goal=new_goal,
            status="rejected",
            reason="no previous metric to reuse",
            replacement_metric=replacement,
        )
    permitted = permitted_metrics_for_goal(new_goal)
    catalog = METRIC_CATALOG.get(previous_metric)
    goal_ok = catalog is not None and new_goal in catalog.valid_goals
    if previous_metric in permitted and goal_ok:
        return MetricRevalidation(
            previous_metric=previous_metric,
            new_goal=new_goal,
            status="accepted",
            reason=(
                f"{previous_metric} remains valid for {new_goal} "
                f"({METRIC_CATALOG_VERSION}/{RULESET_VERSION})"
            ),
            replacement_metric=None,
        )
    replacement = default_metric_for_goal(new_goal)
    previous_goal = previous.inferred_goal or "abundance"
    return MetricRevalidation(
        previous_metric=previous_metric,
        new_goal=new_goal,
        status="rejected",
        reason=(f"{previous_goal} metric {previous_metric} does not represent {new_goal}"),
        replacement_metric=replacement,
    )
