"""Compact multi-target comparison planning contracts and helpers.

The LLM describes *what* to compare. The orchestrator controls dependency order
and executes registered tools. Coordinates are never model-authored.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_COMPARE_HINT = re.compile(
    r"\b("
    r"compare|comparison|which (?:area|one|university|campus)|"
    r"more parks|denser|density|better access|proportion|ratio|"
    r"diversity|coverage|closer to"
    r")\b",
    re.IGNORECASE,
)
_RADIUS_HINT = re.compile(
    r"\bwithin\b.{0,40}\b(\d+(?:\.\d+)?)\s*(km|kilometers?|kilometres?|m|meters?|metres?)\b",
    re.IGNORECASE,
)
_RADIUS_ANY = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(km|kilometers?|kilometres?|m|meters?|metres?)\b",
    re.IGNORECASE,
)
_TEHRAN_HINT = re.compile(r"\b(tehran|iran)\b", re.IGNORECASE)
_ISTANBUL_HINT = re.compile(
    r"\b(istanbul|t[uü]rkiye|turkey|boğaziçi|bogazici)\b",
    re.IGNORECASE,
)
_GREEN_SPACE_HINT = re.compile(r"green[\s-]?spaces?", re.IGNORECASE)
_PARK_HINT = re.compile(r"\bparks?\b", re.IGNORECASE)
_UNIVERSITY_NAME = re.compile(
    r"("
    r"[A-ZÀ-ÖØ-Þ][\w'.+-]*(?:\s+[A-ZÀ-ÖØ-Þ][\w'.+-]*)*\s+University"
    r"(?:\s+of\s+[A-ZÀ-ÖØ-Þ][\w'.+-]*)?"
    r"|"
    r"University\s+of\s+[A-ZÀ-ÖØ-Þ][\w'.+-]*(?:\s+[A-ZÀ-ÖØ-Þ][\w'.+-]*)*"
    r")",
)

DEFAULT_ANALYSIS_RADIUS_M = 2000

#: Explicit OSM tags for "green space" comparison queries. Queried as a union
#: (tag_match=any), not AND. Parks-only requests stay leisure=park.
GREEN_SPACE_TAGS: tuple[str, ...] = (
    "leisure=park",
    "landuse=grass",
    "landuse=recreation_ground",
    "natural=wood",
    "natural=grassland",
)
_OF_PATTERN = re.compile(
    r"(?:within\s+\d+(?:\.\d+)?\s*(?:km|m|kilometers?|metres?|meters?)\s+of\s+)"
    r"(.+?)(?:\.|,?\s+choose|,?\s+compare|,?\s+return|$)",
    re.IGNORECASE | re.DOTALL,
)


class ComparisonTargetDraft(BaseModel):
    """One named comparison landmark from the planner (no coordinates)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=2, max_length=120)
    place_query: str = Field(min_length=2, max_length=200)


class MultiTargetComparisonPlan(BaseModel):
    """Compact typed plan produced by the comparison planner stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_type: Literal["comparison"] = "comparison"
    feature_concept: str = Field(min_length=2, max_length=120)
    targets: list[ComparisonTargetDraft] = Field(min_length=1, max_length=4)
    radius_m: int = Field(gt=0, le=50_000)
    comparison_goal: str = Field(min_length=2, max_length=240)

    @field_validator("targets")
    @classmethod
    def _unique_labels(cls, value: list[ComparisonTargetDraft]) -> list[ComparisonTargetDraft]:
        labels = [item.label.strip().lower() for item in value]
        if len(labels) != len(set(labels)):
            raise ValueError("comparison target labels must be unique")
        return value


class MetricSelectionDraft(BaseModel):
    """LLM metric choice for an already-retrieved comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_metric: str = Field(min_length=2, max_length=64)
    inferred_goal: str = Field(min_length=2, max_length=120)
    rationale: str = Field(default="", max_length=400)
    supporting_metrics: list[str] = Field(default_factory=list, max_length=3)


COMPARISON_PLANNER_SYSTEM = (
    "You are Ariadne Thread's comparison planner. "
    "Return ONE JSON object only (no prose, no Markdown) with keys: "
    'analysis_type (always "comparison"), feature_concept, targets, radius_m, '
    "comparison_goal. "
    "targets is an array of {label, place_query}. "
    "Preserve explicit landmark names from the user message. "
    "place_query must be the landmark name, optionally with city/country "
    '(e.g. "University of Tehran, Tehran, Iran"). '
    "Do NOT invent coordinates, dataset refs, Overpass QL, tool calls, "
    "or final answers. Do NOT substitute unrelated landmarks "
    "(e.g. City Hall). radius_m must match the user distance "
    "(2 km → 2000)."
)

METRIC_PLANNER_SYSTEM = (
    "You select ONE primary metric from the catalog for a park/feature comparison. "
    "Return ONE JSON object only: "
    '{"primary_metric":"...","inferred_goal":"...","rationale":"...","supporting_metrics":[]}. '
    "Allowed primary_metric values: count, density, total_area, coverage_percentage, "
    "median_area, mean_area, standard_deviation. "
    "Prefer count for abundance/availability; density for concentration when equal "
    "point-radius scopes exist. Never invent numeric results."
)

FINAL_REPORT_SYSTEM = (
    "Write a concise comparison report using ONLY the provided computed values. "
    "Do not recalculate. If any target is truncated at a feature limit, say the "
    "count is a lower bound / incomplete inventory. Return ONE JSON object: "
    '{"final_answer":"..."}. No tool calls.'
)


def is_multi_target_landmark_comparison(message: str) -> bool:
    """Heuristic: analytical compare + multiple landmarks (radius optional)."""
    text = message.strip()
    if len(text) < 20:
        return False
    if not _COMPARE_HINT.search(text):
        return False
    labels = extract_landmark_labels(text)
    if len(labels) >= 2:
        return True
    joined = " or " in text.lower() or " and " in text.lower()
    if not joined:
        return False
    return bool(
        re.search(
            r"\b(university|college|campus|square|station|hospital|museum|airport)\b",
            text,
            re.IGNORECASE,
        )
        or re.search(r"\bof\s+[A-Z]", text)
    )


def is_indicator_analysis_request(message: str) -> bool:
    """Whether the comparison/indicator executor should handle this question."""
    if is_multi_target_landmark_comparison(message):
        return True
    if not _COMPARE_HINT.search(message):
        return False
    return len(extract_landmark_labels(message)) >= 1


def extract_radius_m_from_user(message: str) -> int | None:
    match = _RADIUS_HINT.search(message) or _RADIUS_ANY.search(message)
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("k"):
        return round(value * 1000)
    return round(value)


def extract_landmark_labels(message: str) -> tuple[str, ...]:
    """Best-effort landmark names from 'within … of', colon lists, or university names."""
    match = _OF_PATTERN.search(message)
    if match is not None:
        chunk = match.group(1).strip().rstrip(".")
        chunk = re.split(
            r"\.\s+|,\s*(?:choose|compare|return|generate)\b",
            chunk,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        parsed = _split_landmark_chunk(chunk)
        if len(parsed) >= 2:
            return parsed
    universities = _university_labels(message)
    if len(universities) >= 1:
        return universities
    if ":" in message:
        tail = message.split(":", 1)[1]
        parsed = _split_landmark_chunk(tail)
        if parsed:
            return parsed
    return ()


def canonicalize_place_query(
    label: str,
    *,
    user_message: str,
    proposed_query: str | None = None,
) -> str:
    """Preserve explicit target names; optionally append Tehran/Iran context."""
    base = re.sub(r"^(the|a|an)\s+", "", label.strip(), flags=re.IGNORECASE).strip()
    proposed = (proposed_query or "").strip()
    if proposed and _shares_label_identity(base, proposed):
        query = re.sub(r"^(the|a|an)\s+", "", proposed, flags=re.IGNORECASE).strip()
    else:
        query = base
    if _TEHRAN_HINT.search(user_message) and not re.search(r",\s*Iran\b", query, re.I):
        if re.search(r"\bTehran\b", query, re.I):
            query = f"{query}, Iran"
        else:
            query = f"{query}, Tehran, Iran"
    elif _ISTANBUL_HINT.search(user_message) and not re.search(
        r",\s*(Istanbul|Turkey|Türkiye)\b", query, re.I
    ):
        if re.search(r"\bIstanbul\b", query, re.I):
            query = f"{query}, Turkey"
        else:
            query = f"{query}, Istanbul, Turkey"
    return query[:200]


def _shares_label_identity(label: str, query: str) -> bool:
    label_tokens = _significant_tokens(label)
    query_tokens = _significant_tokens(query)
    if not label_tokens:
        return False
    overlap = label_tokens & query_tokens
    return len(overlap) >= max(1, (len(label_tokens) + 1) // 2)


def _significant_tokens(text: str) -> set[str]:
    stop = {
        "the",
        "of",
        "and",
        "in",
        "at",
        "near",
        "main",
        "campus",
        "area",
        "iran",
        "tehran",
    }
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {tok for tok in tokens if len(tok) > 2 and tok not in stop}


def parse_comparison_plan_payload(raw: str | dict[str, Any]) -> MultiTargetComparisonPlan:
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, dict):
        raise ValueError("comparison plan must be a JSON object")
    # Tolerate accidental nesting.
    if "comparison_plan" in payload and isinstance(payload["comparison_plan"], dict):
        payload = payload["comparison_plan"]
    return MultiTargetComparisonPlan.model_validate(payload)


def normalize_plan_with_user_context(
    plan: MultiTargetComparisonPlan,
    user_message: str,
) -> MultiTargetComparisonPlan:
    """Rewrite place queries from preserved labels + user radius."""
    radius = extract_radius_m_from_user(user_message) or plan.radius_m
    labels = extract_landmark_labels(user_message)
    targets: list[ComparisonTargetDraft] = []
    for index, target in enumerate(plan.targets):
        label = target.label
        if index < len(labels) and _shares_label_identity(labels[index], label):
            label = labels[index]
        elif (
            index < len(labels)
            and not _shares_label_identity(label, labels[index])
            and (
                _shares_label_identity(labels[index], target.place_query)
                or len(labels) == len(plan.targets)
            )
        ):
            # Prefer explicit user landmark wording when the model drifted.
            label = labels[index]
        place_query = canonicalize_place_query(
            label,
            user_message=user_message,
            proposed_query=target.place_query,
        )
        targets.append(ComparisonTargetDraft(label=label, place_query=place_query))
    updates: dict[str, Any] = {"targets": targets, "radius_m": radius}
    if _GREEN_SPACE_HINT.search(user_message):
        # Planner often substitutes "public parks"; keep the user's concept so
        # tags_for_feature_concept can apply the green-space union.
        updates["feature_concept"] = "green spaces"
        if not _GREEN_SPACE_HINT.search(plan.comparison_goal):
            updates["comparison_goal"] = "green-space provision"
    return plan.model_copy(update=updates)


def tags_for_feature_concept(concept: str) -> list[str] | None:
    """Return the explicit OSM tag list for a comparison feature concept.

    ``None`` means the concept is not a known built-in set and RAG grounding
    should decide. Green-space requests use a union of park/grass/wood tags.
    """
    text = concept.lower()
    if _GREEN_SPACE_HINT.search(text):
        return list(GREEN_SPACE_TAGS)
    if "leisure=park" in text or _PARK_HINT.search(text):
        return ["leisure=park"]
    return None


def seed_plan_from_user_message(message: str) -> MultiTargetComparisonPlan | None:
    """Deterministic seed when landmarks are explicit in the request."""
    labels = extract_landmark_labels(message)
    radius = extract_radius_m_from_user(message) or DEFAULT_ANALYSIS_RADIUS_M
    if len(labels) < 1:
        return None
    if _GREEN_SPACE_HINT.search(message):
        feature = "green spaces"
        goal = "green-space provision"
    elif _PARK_HINT.search(message):
        feature = "public parks"
        goal = "park availability"
    else:
        feature = "features"
        goal = "feature comparison"
    targets = [
        ComparisonTargetDraft(
            label=label,
            place_query=canonicalize_place_query(label, user_message=message),
        )
        for label in labels[:4]
    ]
    return MultiTargetComparisonPlan(
        feature_concept=feature,
        targets=targets,
        radius_m=radius,
        comparison_goal=goal,
    )


def _split_landmark_chunk(chunk: str) -> tuple[str, ...]:
    pieces = re.split(r",\s*and\s+|\s+and\s+|\s+or\s+|,\s+", chunk)
    cleaned: list[str] = []
    drop = {"tehran", "iran", "istanbul", "turkey", "türkiye", "the"}
    for part in pieces:
        part = re.sub(r"^(the|a|an)\s+", "", part, flags=re.IGNORECASE).strip(" .,?")
        if len(part) < 2 or part.lower() in drop:
            continue
        cleaned.append(part)
    return tuple(cleaned)


def _university_labels(message: str) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    for match in _UNIVERSITY_NAME.finditer(message):
        label = re.sub(r"\s+", " ", match.group(1)).strip(" .,")
        key = label.lower()
        if key in seen or len(label) < 5:
            continue
        seen.add(key)
        found.append(label)
    return tuple(found)
