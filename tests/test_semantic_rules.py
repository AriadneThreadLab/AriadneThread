"""Semantic metric rule validation."""

from __future__ import annotations

from app.analytics.rules import (
    RULESET_VERSION,
    SEMANTIC_RULES,
    STABLE_RULE_IDS,
    validate_metric_for_goal,
)


def test_stable_rule_ids() -> None:
    assert RULESET_VERSION == "metric-rules-1"
    assert STABLE_RULE_IDS == (
        "ABUNDANCE_COUNT_001",
        "CONCENTRATION_DENSITY_001",
        "ACCESSIBILITY_DISTANCE_001",
        "TOTAL_PROVISION_SUM_001",
        "GREENSPACE_TOTAL_AREA_001",
        "TYPICAL_VALUE_MEDIAN_001",
        "EXPLICIT_AVERAGE_MEAN_001",
        "VARIABILITY_STDDEV_001",
        "COVERAGE_PERCENTAGE_001",
        "RELATIVE_SHARE_RATIO_001",
        "RANGE_EXTREMES_001",
    )


def test_abundance_count() -> None:
    result = validate_metric_for_goal("abundance", "count", user_explicit=False, role="primary")
    assert result.status == "matched"
    assert result.matched_rule is not None
    assert result.matched_rule.rule_id == "ABUNDANCE_COUNT_001"


def test_typical_median_vs_explicit_mean() -> None:
    median = validate_metric_for_goal(
        "typical_value", "median_area", user_explicit=False, role="primary"
    )
    assert median.status == "matched"
    mean_blocked = validate_metric_for_goal(
        "typical_value", "mean_area", user_explicit=False, role="primary"
    )
    assert mean_blocked.status == "explicit_request_required"
    mean_ok = validate_metric_for_goal(
        "typical_value", "mean_area", user_explicit=True, role="primary"
    )
    assert mean_ok.status == "matched"


def test_false_claimed_rule_ignored() -> None:
    result = validate_metric_for_goal(
        "abundance",
        "count",
        user_explicit=False,
        role="primary",
        claimed_rule_ids=["COVERAGE_PERCENTAGE_001"],
    )
    assert result.status == "matched"
    assert result.ignored_claimed_rule_ids == ("COVERAGE_PERCENTAGE_001",)


def test_every_metric_reachable() -> None:
    permitted: set[str] = set()
    for rule in SEMANTIC_RULES:
        permitted.update(rule.permitted_metrics)
    from typing import get_args

    from app.analytics.contracts import MetricType

    assert set(get_args(MetricType)) <= permitted
