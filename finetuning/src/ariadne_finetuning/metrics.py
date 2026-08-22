"""Plan-level evaluation metrics.

One aggregate accuracy number would hide the failures that matter: a plan can
be valid JSON, name the right analysis type and still drop a comparison target
or invent a coordinate. Every dimension is therefore reported separately, with
counts, so a metric that never applied is distinguishable from one that always
failed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ariadne_finetuning.schema import allowed_top_level_keys, is_valid

_NON_WORD = re.compile(r"[^a-z0-9]+")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

#: Comparison goals are free text; token overlap at or above this level counts
#: as the same stated goal.
GOAL_MATCH_THRESHOLD = 0.6

_COORDINATE_KEYS = frozenset({"lat", "lon", "latitude", "longitude"})


@dataclass(frozen=True)
class SampleOutcome:
    """Per-example evaluation result. ``None`` means "not applicable here"."""

    parsed: bool
    schema_valid: bool
    analysis_type: bool | None = None
    targets_preserved: bool | None = None
    radius: bool | None = None
    feature_concept: bool | None = None
    comparison_goal: bool | None = None
    metric_intent: bool | None = None
    hallucinated_fields: int = 0
    invented_coordinates: int = 0


@dataclass(frozen=True)
class PlanMetrics:
    """Aggregated, individually reported rates."""

    example_count: int
    valid_analysis_plan_rate: float
    analysis_type_accuracy: float | None
    target_preservation_accuracy: float | None
    radius_accuracy: float | None
    feature_concept_accuracy: float | None
    comparison_goal_accuracy: float | None
    metric_intent_accuracy: float | None
    hallucinated_field_rate: float
    invented_coordinate_rate: float
    parse_failure_count: int = 0
    applicability: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_count": self.example_count,
            "valid_analysis_plan_rate": self.valid_analysis_plan_rate,
            "analysis_type_accuracy": self.analysis_type_accuracy,
            "target_preservation_accuracy": self.target_preservation_accuracy,
            "radius_accuracy": self.radius_accuracy,
            "feature_concept_accuracy": self.feature_concept_accuracy,
            "comparison_goal_accuracy": self.comparison_goal_accuracy,
            "metric_intent_accuracy": self.metric_intent_accuracy,
            "hallucinated_field_rate": self.hallucinated_field_rate,
            "invented_coordinate_rate": self.invented_coordinate_rate,
            "parse_failure_count": self.parse_failure_count,
            "applicability": dict(self.applicability),
        }


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the model's reply, tolerating a chat template's surrounding prose."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        _, _, candidate = candidate.partition("\n")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(candidate)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _normalize(text: Any) -> str:
    return " ".join(_NON_WORD.sub(" ", str(text).lower()).split())


def _target_labels(plan: dict[str, Any]) -> list[str]:
    targets = plan.get("targets")
    if not isinstance(targets, list):
        return []
    return sorted(_normalize(item.get("label", "")) for item in targets if isinstance(item, dict))


def _primary_metric(plan: dict[str, Any]) -> str | None:
    metrics = plan.get("metrics")
    if not isinstance(metrics, list):
        return None
    for item in metrics:
        if isinstance(item, dict) and item.get("role") == "primary":
            value = item.get("metric")
            return str(value) if value is not None else None
    return None


def _coordinates(payload: Any, depth: int = 0) -> set[tuple[str, float]]:
    found: set[tuple[str, float]] = set()
    if depth > 8:
        return found
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in _COORDINATE_KEYS and isinstance(value, int | float):
                found.add((key, round(float(value), 6)))
            else:
                found |= _coordinates(value, depth + 1)
    elif isinstance(payload, list):
        for item in payload:
            found |= _coordinates(item, depth + 1)
    return found


def _goal_match(prediction: Any, reference: Any) -> bool:
    predicted = set(_normalize(prediction).split())
    expected = set(_normalize(reference).split())
    if not expected:
        return not predicted
    if not predicted:
        return False
    overlap = len(predicted & expected) / float(len(predicted | expected))
    return overlap >= GOAL_MATCH_THRESHOLD


def score_sample(
    prediction_text: str,
    reference: dict[str, Any],
    schema: dict[str, Any],
) -> SampleOutcome:
    """Compare one generated plan against its reference plan."""
    prediction = extract_json_object(prediction_text)
    if prediction is None:
        return SampleOutcome(parsed=False, schema_valid=False)

    valid = is_valid(prediction, schema)
    allowed = allowed_top_level_keys(schema)
    hallucinated = len([key for key in prediction if allowed and key not in allowed])

    reference_coordinates = _coordinates(reference)
    invented = len(_coordinates(prediction) - reference_coordinates)

    def compare(key: str) -> bool | None:
        if key not in reference:
            return None
        return _normalize(prediction.get(key)) == _normalize(reference.get(key))

    targets_preserved: bool | None = None
    if "targets" in reference:
        targets_preserved = _target_labels(prediction) == _target_labels(reference)

    goal: bool | None = None
    if "comparison_goal" in reference:
        goal = _goal_match(prediction.get("comparison_goal"), reference["comparison_goal"])

    metric_intent: bool | None = None
    if _primary_metric(reference) is not None:
        metric_intent = _primary_metric(prediction) == _primary_metric(reference)

    return SampleOutcome(
        parsed=True,
        schema_valid=valid,
        analysis_type=compare("analysis_type"),
        targets_preserved=targets_preserved,
        radius=compare("radius_m"),
        feature_concept=compare("feature_concept"),
        comparison_goal=goal,
        metric_intent=metric_intent,
        hallucinated_fields=hallucinated,
        invented_coordinates=invented,
    )


def _rate(values: list[bool | None]) -> tuple[float | None, int]:
    applicable = [value for value in values if value is not None]
    if not applicable:
        return None, 0
    return round(sum(applicable) / len(applicable), 4), len(applicable)


def aggregate(outcomes: list[SampleOutcome]) -> PlanMetrics:
    """Aggregate per-sample outcomes into the reported metric set."""
    count = len(outcomes)
    if count == 0:
        return PlanMetrics(
            example_count=0,
            valid_analysis_plan_rate=0.0,
            analysis_type_accuracy=None,
            target_preservation_accuracy=None,
            radius_accuracy=None,
            feature_concept_accuracy=None,
            comparison_goal_accuracy=None,
            metric_intent_accuracy=None,
            hallucinated_field_rate=0.0,
            invented_coordinate_rate=0.0,
        )

    applicability: dict[str, int] = {}
    rates: dict[str, float | None] = {}
    for name, values in (
        ("analysis_type_accuracy", [item.analysis_type for item in outcomes]),
        ("target_preservation_accuracy", [item.targets_preserved for item in outcomes]),
        ("radius_accuracy", [item.radius for item in outcomes]),
        ("feature_concept_accuracy", [item.feature_concept for item in outcomes]),
        ("comparison_goal_accuracy", [item.comparison_goal for item in outcomes]),
        ("metric_intent_accuracy", [item.metric_intent for item in outcomes]),
    ):
        rate, applicable = _rate(values)
        rates[name] = rate
        applicability[name] = applicable

    return PlanMetrics(
        example_count=count,
        valid_analysis_plan_rate=round(sum(1 for item in outcomes if item.schema_valid) / count, 4),
        analysis_type_accuracy=rates["analysis_type_accuracy"],
        target_preservation_accuracy=rates["target_preservation_accuracy"],
        radius_accuracy=rates["radius_accuracy"],
        feature_concept_accuracy=rates["feature_concept_accuracy"],
        comparison_goal_accuracy=rates["comparison_goal_accuracy"],
        metric_intent_accuracy=rates["metric_intent_accuracy"],
        hallucinated_field_rate=round(
            sum(1 for item in outcomes if item.hallucinated_fields) / count, 4
        ),
        invented_coordinate_rate=round(
            sum(1 for item in outcomes if item.invented_coordinates) / count, 4
        ),
        parse_failure_count=sum(1 for item in outcomes if not item.parsed),
        applicability=applicability,
    )
