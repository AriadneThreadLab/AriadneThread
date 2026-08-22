"""Reusable Analysis Memory: methodology distilled from execution history.

Execution history (``ExecutionSnapshot``) is an immutable audit record: what
was asked, which landmarks were resolved, which features came back, what the
answer was. It exists for provenance, debugging and reproducibility, and its
*results* are never replayed as the answer to a new question.

This module projects that history into target-free knowledge — dataset
definition, scope, metric decision and validation rules — that a new request
may reuse after validation. Landmark names never enter a pattern: they are
generalised to a :data:`TargetCategory`.
"""

from __future__ import annotations

import json
import re

from app.agent.comparison_workflow import tags_for_feature_concept
from app.analytics.contracts import AnalysisGoal
from app.execution_memory.contracts import (
    PATTERN_VALIDATION_RULES,
    AnalysisPattern,
    ExecutionSnapshot,
    MetricSnapshot,
    PatternCheck,
    PatternComponent,
    PatternReuseAssessment,
    TargetCategory,
)
from app.execution_memory.metric_revalidation import revalidate_metric

_CATEGORY_PATTERNS: tuple[tuple[TargetCategory, re.Pattern[str]], ...] = (
    (
        "educational_facility",
        re.compile(r"\b(universit\w*|college|campus|school|institute|academy|faculty)\b", re.I),
    ),
    (
        "transport_hub",
        re.compile(r"\b(station|airport|terminal|metro|bus\s?stop|interchange|port)\b", re.I),
    ),
    (
        "healthcare_facility",
        re.compile(r"\b(hospital|clinic|medical\s?cent(?:er|re)|polyclinic)\b", re.I),
    ),
    (
        "civic_landmark",
        re.compile(r"\b(square|city\s?hall|museum|library|stadium|park\s?complex|bazaar)\b", re.I),
    ),
)

_CATEGORY_PHRASES: dict[TargetCategory, str] = {
    "educational_facility": "educational facilities",
    "transport_hub": "transport hubs",
    "healthcare_facility": "healthcare facilities",
    "civic_landmark": "civic landmarks",
    "place": "named places",
}

#: Words shared by many landmark names; they do not identify a specific place.
_GENERIC_LABEL_TOKENS = frozenset(
    {
        "the",
        "of",
        "and",
        "university",
        "universities",
        "technology",
        "college",
        "school",
        "institute",
        "academy",
        "faculty",
        "campus",
        "main",
        "hospital",
        "station",
        "square",
        "museum",
        "library",
    }
)


def classify_target_category(labels: tuple[str, ...] | list[str]) -> TargetCategory:
    """Generalise landmark names to the kind of place they describe."""
    blob = " ".join(labels)
    if not blob.strip():
        return "place"
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(blob):
            return category
    return "place"


def pattern_key_for(
    *,
    analysis_type: str,
    feature_concept: str,
    target_category: TargetCategory,
    scope_kind: str,
) -> str:
    concept = " ".join(feature_concept.lower().split()) or "features"
    return f"{analysis_type}|{concept}|{target_category}|{scope_kind}"


def identity_tokens(label: str) -> set[str]:
    """Tokens that identify a specific landmark (not its generic type)."""
    tokens = re.findall(r"[a-z0-9]+", label.lower())
    return {token for token in tokens if len(token) > 2 and token not in _GENERIC_LABEL_TOKENS}


def derive_analysis_pattern(snapshot: ExecutionSnapshot) -> AnalysisPattern | None:
    """Project one execution into reusable, target-free methodology."""
    if snapshot.outcome == "failed":
        return None
    tags = tuple(snapshot.grounding_tags)
    concept = snapshot.feature_concept.strip()
    if not tags or not concept:
        return None
    labels = [target.label for target in snapshot.targets]
    category = classify_target_category(labels)
    pattern = AnalysisPattern(
        pattern_key=pattern_key_for(
            analysis_type=snapshot.analysis_type,
            feature_concept=concept,
            target_category=category,
            scope_kind=snapshot.scope_kind,
        ),
        analysis_pattern=_describe(snapshot.analysis_type, concept, category),
        analysis_type=snapshot.analysis_type,
        feature_concept=concept[:120],
        target_category=category,
        observed_target_count=min(len(labels), 8),
        scope_kind=snapshot.scope_kind,
        dataset_tags=tags,
        radius_m=snapshot.radius_m,
        metric=snapshot.metric.metric,
        inferred_goal=snapshot.metric.inferred_goal or snapshot.inferred_goal,
        metric_rule_id=snapshot.metric.rule_id,
        metric_catalog_version=snapshot.metric.catalog_version,
        ruleset_version=snapshot.metric.ruleset_version,
        validation_rules=PATTERN_VALIDATION_RULES,
        source_execution_id=snapshot.execution_id,
        observed_at=snapshot.created_at,
        outcome=snapshot.outcome,
    )
    assert_pattern_is_target_free(pattern, labels)
    return pattern


def assert_pattern_is_target_free(
    pattern: AnalysisPattern,
    labels: tuple[str, ...] | list[str],
) -> None:
    """Guard the core invariant: methodology must not carry landmark identity."""
    blob = json.dumps(pattern.model_dump(mode="json"), ensure_ascii=False).lower()
    for label in labels:
        for token in identity_tokens(label):
            if token in blob:
                raise ValueError(
                    f"analysis pattern leaked target identity {token!r}; "
                    "reusable memory stores methodology only"
                )


def select_pattern(
    patterns: list[AnalysisPattern],
    *,
    feature_concept: str,
    target_category: TargetCategory,
    analysis_type: str = "comparison",
) -> AnalysisPattern | None:
    """Most recent pattern compatible with the current request."""
    if not patterns:
        return None
    ordered = sorted(patterns, key=lambda item: item.observed_at, reverse=True)
    wanted = pattern_key_for(
        analysis_type=analysis_type,
        feature_concept=feature_concept,
        target_category=target_category,
        scope_kind="point",
    )
    for item in ordered:
        if item.pattern_key == wanted:
            return item
    concept = " ".join(feature_concept.lower().split())
    for item in ordered:
        if item.feature_concept.lower() == concept:
            return item
    return None


def validate_pattern_reuse(
    pattern: AnalysisPattern | None,
    *,
    requested_labels: tuple[str, ...],
    feature_concept: str,
    inferred_goal: AnalysisGoal,
    radius_m: int | None,
) -> PatternReuseAssessment:
    """Decide whether stored methodology applies to the current request."""
    if pattern is None:
        return PatternReuseAssessment(
            reusable=False,
            checks=(
                PatternCheck(
                    name="pattern_available",
                    passed=False,
                    detail="no compatible analysis pattern in memory",
                ),
            ),
            recomputed_components=(
                "dataset_definition",
                "radius",
                "metric",
                "targets",
                "place_resolution",
                "osm_query",
                "metric_values",
            ),
        )

    checks: list[PatternCheck] = [
        PatternCheck(
            name="pattern_available",
            passed=True,
            detail=f"{pattern.analysis_pattern} ({pattern.pattern_key})",
        )
    ]

    expected_tags = tags_for_feature_concept(feature_concept)
    dataset_ok = bool(pattern.dataset_tags) and (
        expected_tags is None or set(expected_tags) == set(pattern.dataset_tags)
    )
    checks.append(
        PatternCheck(
            name="dataset_definition_valid",
            passed=dataset_ok,
            detail=(
                ",".join(pattern.dataset_tags)
                if dataset_ok
                else f"stored tags {','.join(pattern.dataset_tags) or 'none'} "
                f"do not describe {feature_concept}"
            ),
        )
    )

    metric_rev = revalidate_metric(
        previous=MetricSnapshot(
            metric=pattern.metric,
            inferred_goal=pattern.inferred_goal,
            rule_id=pattern.metric_rule_id,
            catalog_version=pattern.metric_catalog_version,
            ruleset_version=pattern.ruleset_version,
        ),
        new_goal=inferred_goal,
    )
    metric_ok = metric_rev.status == "accepted"
    checks.append(
        PatternCheck(name="metric_still_meaningful", passed=metric_ok, detail=metric_rev.reason)
    )

    unique = tuple(dict.fromkeys(label.strip().lower() for label in requested_labels if label))
    requested_category = classify_target_category(requested_labels)
    comparable = (
        len(unique) >= 2
        and len(unique) == len([label for label in requested_labels if label.strip()])
        and (
            requested_category == pattern.target_category
            or "place" in (requested_category, pattern.target_category)
        )
    )
    checks.append(
        PatternCheck(
            name="targets_comparable",
            passed=comparable,
            detail=f"{len(unique)} {requested_category.replace('_', ' ')} target(s)",
        )
    )

    effective_radius = radius_m or pattern.radius_m
    data_available = pattern.scope_kind == "point" and effective_radius is not None
    checks.append(
        PatternCheck(
            name="required_data_available",
            passed=data_available,
            detail=(
                f"{pattern.scope_kind} scope, {effective_radius} m"
                if data_available
                else "no comparable point-radius scope"
            ),
        )
    )

    radius_unchanged = radius_m is None or pattern.radius_m == radius_m

    reused: list[PatternComponent] = []
    recomputed: list[PatternComponent] = ["targets", "place_resolution", "metric_values"]
    if dataset_ok:
        reused.append("dataset_definition")
        reused.append("validation_rules")
    else:
        recomputed.append("dataset_definition")
    if data_available and radius_unchanged:
        reused.append("radius")
    else:
        recomputed.append("radius")
    if metric_ok:
        reused.append("metric")
    else:
        recomputed.append("metric")
    recomputed.append("osm_query")

    return PatternReuseAssessment(
        pattern_key=pattern.pattern_key,
        analysis_pattern=pattern.analysis_pattern,
        reusable=all(item.passed for item in checks),
        checks=tuple(checks),
        reused_components=tuple(dict.fromkeys(reused)),
        recomputed_components=tuple(dict.fromkeys(recomputed)),
    )


def _describe(analysis_type: str, feature_concept: str, category: TargetCategory) -> str:
    verb = "compare" if analysis_type == "comparison" else analysis_type
    return f"{verb} {feature_concept} around {_CATEGORY_PHRASES[category]}"[:200]
