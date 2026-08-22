"""Versioned semantic metric rules and statistical guidance.

The LLM interprets user intent. These rules validate whether a proposed
(goal, metric) pair is analytically admissible. They are not phrase matchers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.analytics.contracts import (
    AnalysisGoal,
    MetricRole,
    MetricType,
    RuleId,
)

RULESET_VERSION = "metric-rules-1"


class SemanticMetricRule(BaseModel):
    """One stable validation rule for metric selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: RuleId
    goal: AnalysisGoal | None = None
    any_goal: bool = False
    permitted_metrics: tuple[MetricType, ...]
    statement: str
    requires_user_explicit: bool = False
    supporting_only: bool = False


SEMANTIC_RULES: tuple[SemanticMetricRule, ...] = (
    SemanticMetricRule(
        rule_id="ABUNDANCE_COUNT_001",
        goal="abundance",
        permitted_metrics=("count",),
        statement="How many features exist is measured by counting them.",
    ),
    SemanticMetricRule(
        rule_id="CONCENTRATION_DENSITY_001",
        goal="concentration",
        permitted_metrics=("density",),
        statement=("Concentration compares quantity against area, so features per km² is used."),
    ),
    SemanticMetricRule(
        rule_id="ACCESSIBILITY_DISTANCE_001",
        goal="accessibility",
        permitted_metrics=(
            "nearest_distance",
            "mean_nearest_distance",
            "median_nearest_distance",
        ),
        statement=(
            "Accessibility is a distance question, so distance to the nearest feature is used."
        ),
    ),
    SemanticMetricRule(
        rule_id="TOTAL_PROVISION_SUM_001",
        goal="total_provision",
        permitted_metrics=("sum",),
        statement="Total provision aggregates a numeric property across all features.",
    ),
    SemanticMetricRule(
        rule_id="GREENSPACE_TOTAL_AREA_001",
        goal="total_provision",
        permitted_metrics=("total_area",),
        statement="Provision of area-based features is measured by total mapped area.",
    ),
    SemanticMetricRule(
        rule_id="TYPICAL_VALUE_MEDIAN_001",
        goal="typical_value",
        permitted_metrics=("median", "median_area", "median_nearest_distance"),
        statement="A typical value is reported as the median, which is robust to outliers.",
    ),
    SemanticMetricRule(
        rule_id="EXPLICIT_AVERAGE_MEAN_001",
        goal="typical_value",
        permitted_metrics=("mean", "mean_area", "mean_nearest_distance"),
        statement="The arithmetic mean is used because the user asked for an average.",
        requires_user_explicit=True,
    ),
    SemanticMetricRule(
        rule_id="VARIABILITY_STDDEV_001",
        goal="variability",
        permitted_metrics=("standard_deviation",),
        statement="Consistency is measured as dispersion using the standard deviation.",
    ),
    SemanticMetricRule(
        rule_id="COVERAGE_PERCENTAGE_001",
        goal="coverage",
        permitted_metrics=("coverage_percentage",),
        statement="Coverage is the share of the analysis area occupied by the features.",
    ),
    SemanticMetricRule(
        rule_id="RELATIVE_SHARE_RATIO_001",
        goal="relative_share",
        permitted_metrics=("ratio",),
        statement=("A relative share is expressed as a ratio between two computed quantities."),
    ),
    SemanticMetricRule(
        rule_id="RANGE_EXTREMES_001",
        any_goal=True,
        permitted_metrics=("min", "max"),
        statement=(
            "Extremes describe the range of observed values and support the primary indicator."
        ),
        supporting_only=True,
    ),
)

STATISTICAL_GUIDANCE: dict[str, str] = {
    "median_preferred": (
        "Median is preferred for typical values because it is robust to skewed "
        "distributions and outliers."
    ),
    "mean_when_explicit": (
        "Mean is used when the user explicitly asks for an average or arithmetic mean."
    ),
    "stddev_is_variability": (
        "Standard deviation measures variability and consistency; it is not a "
        "generic better/worse score."
    ),
}


RuleMatchStatus = Literal[
    "matched", "semantic_rule_violation", "explicit_request_required", "role_not_permitted"
]


@dataclass(frozen=True, slots=True)
class RuleValidationResult:
    """Outcome of validating (goal, metric, role, user_explicit)."""

    status: RuleMatchStatus
    matched_rule: SemanticMetricRule | None
    supported_alternatives: tuple[MetricType, ...]
    ignored_claimed_rule_ids: tuple[RuleId, ...]
    notes: tuple[str, ...]


def _rules_for_goal(goal: AnalysisGoal) -> list[SemanticMetricRule]:
    return [rule for rule in SEMANTIC_RULES if rule.any_goal or rule.goal == goal]


def permitted_metrics_for_goal(goal: AnalysisGoal) -> tuple[MetricType, ...]:
    """Metrics the ruleset allows for ``goal`` (any role)."""
    metrics: list[MetricType] = []
    seen: set[MetricType] = set()
    for rule in _rules_for_goal(goal):
        for metric in rule.permitted_metrics:
            if metric not in seen:
                seen.add(metric)
                metrics.append(metric)
    return tuple(metrics)


def validate_metric_for_goal(
    goal: AnalysisGoal,
    metric: MetricType,
    *,
    user_explicit: bool,
    role: MetricRole,
    claimed_rule_ids: list[RuleId] | None = None,
) -> RuleValidationResult:
    """Validate whether ``metric`` is admissible for ``goal``."""
    claimed = tuple(claimed_rule_ids or ())
    candidates = [rule for rule in _rules_for_goal(goal) if metric in rule.permitted_metrics]
    alternatives = permitted_metrics_for_goal(goal)

    if not candidates:
        return RuleValidationResult(
            status="semantic_rule_violation",
            matched_rule=None,
            supported_alternatives=alternatives,
            ignored_claimed_rule_ids=(),
            notes=("No semantic rule permits this goal/metric pair.",),
        )

    # Prefer a non-supporting-only rule when the role is primary.
    matched = candidates[0]
    for rule in candidates:
        if role == "primary" and rule.supporting_only:
            continue
        matched = rule
        break

    if matched.supporting_only and role == "primary":
        return RuleValidationResult(
            status="role_not_permitted",
            matched_rule=matched,
            supported_alternatives=alternatives,
            ignored_claimed_rule_ids=(),
            notes=(f"Rule {matched.rule_id} allows supporting metrics only.",),
        )

    if matched.requires_user_explicit and not user_explicit:
        return RuleValidationResult(
            status="explicit_request_required",
            matched_rule=matched,
            supported_alternatives=tuple(
                m
                for m in alternatives
                if m in {"median", "median_area", "median_nearest_distance", "count", "density"}
            )
            or alternatives,
            ignored_claimed_rule_ids=(),
            notes=(f"Rule {matched.rule_id} requires an explicit user request for this metric.",),
        )

    ignored: list[RuleId] = []
    notes: list[str] = []
    for claim in claimed:
        if claim != matched.rule_id:
            ignored.append(claim)
    if ignored:
        notes.append("rule_claim_ignored")

    return RuleValidationResult(
        status="matched",
        matched_rule=matched,
        supported_alternatives=alternatives,
        ignored_claimed_rule_ids=tuple(ignored),
        notes=tuple(notes),
    )


STABLE_RULE_IDS: tuple[RuleId, ...] = tuple(rule.rule_id for rule in SEMANTIC_RULES)
