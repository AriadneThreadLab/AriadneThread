"""Adaptive user questions from missing-requirement records.

This is a Solar Decision Support conversation, not a raw calculation API.
The first question is the user goal. Later questions are only that goal's
decision parameters. Provider-filled science and engine knobs are never asked
unless the user already stated them.
"""

from __future__ import annotations

from app.solar.contracts import (
    ENGINE_INTERNAL_FIELDS,
    ENGINE_OPTIONAL_FIELDS,
    PROVIDER_FILLED_FIELDS,
    SolarDecisionDraft,
    SolarFollowUpQuestion,
    SolarMissingRequirement,
)
from app.solar.slots import candidates_have_geometry

GOAL_PROMPT = (
    "Which goal should SolarDSS support: (1) existing building evaluation, "
    "(2) site selection, or (3) candidate ranking?"
)

QUESTION_PROMPTS: dict[str, str] = {
    "mode": GOAL_PROMPT,
    "location": "Where is the building or roof?",
    "site_point": "Where is the building or roof?",
    "site_selection_scope": "What area should be searched?",
    "region": "What area should be searched?",
    "aoi": "What area should be searched?",
    "candidates": "Which candidate sites or buildings should be ranked?",
    "candidate_geometries": (
        "Provide polygons or lon/lat for each candidate. Locations are not invented."
    ),
    "project_type": (
        "What is the building type: residential rooftop, commercial rooftop, or ground-mounted?"
    ),
    "target_capacity_kw": "What is the energy demand or target PV capacity?",
    "objective": "What is the primary energy objective?",
    "budget": "Do you have a budget constraint?",
    "battery_required": "Is battery storage required?",
    "grid_connection_required": "Are there grid connection or maximum-distance requirements?",
    "maximum_grid_distance": "Is there a maximum acceptable grid distance?",
    "bbox": "What bounding box (west,south,east,north) should be searched?",
    "generation_strategy": "Regular grid or hexagonal grid? Resolution is not chosen for you.",
    "cell_size": "What grid cell size and unit (m or deg) should be used?",
    "ranking_profile": (
        "What should ranking or land constraints optimize: energy yield, "
        "infrastructure access, environmental limits, or a balanced mix?"
    ),
    "constraint_profile": (
        "Which shipped constraint profile: utility_scale_default_example or "
        "rooftop_default_example?"
    ),
    "tilt_deg": "What array tilt in degrees should be used for yield?",
    "azimuth_deg": "What array azimuth in degrees (clockwise from north) should be used?",
    "system_loss_percent": "What system-loss percent should be used? It is not defaulted to 14%.",
    "top_n": "How many top-ranked sites should receive a yield estimate?",
    "mounting": "Ground or rooftop mounting for the yield engine?",
}

_FIXED_SITE_MUST: tuple[str, ...] = ("location",)
_FIXED_SITE_SHOULD: tuple[str, ...] = ("project_type", "target_capacity_kw")
_SITE_SELECTION_SHOULD: tuple[str, ...] = ("project_type", "ranking_profile")
_SITE_SELECTION_NICE: tuple[str, ...] = ("grid_connection_required",)
_CANDIDATE_RANKING_SHOULD: tuple[str, ...] = ("ranking_profile",)


def questions_for_missing(
    missing: tuple[SolarMissingRequirement, ...],
    *,
    draft: SolarDecisionDraft,
    mentioned: frozenset[str] = frozenset(),
) -> tuple[SolarFollowUpQuestion, ...]:
    """Ask the goal first, then only that goal's decision parameters."""
    if draft.mode is None:
        return (
            SolarFollowUpQuestion(
                field="mode",
                prompt=GOAL_PROMPT,
                required=True,
            ),
        )
    allow = set(_fields_to_ask(draft, mentioned=mentioned))
    filled = set(draft.filled_fields())
    seen: set[str] = set()
    questions: list[SolarFollowUpQuestion] = []
    for field in _fields_to_ask(draft, mentioned=mentioned):
        question = _question_for_field(
            field,
            draft=draft,
            filled=filled,
            mentioned=mentioned,
            seen=seen,
            required=True,
        )
        if question is not None:
            questions.append(question)
    for record in missing:
        field = record.field
        if field in allow:
            continue
        if field not in mentioned:
            continue
        if field in ENGINE_INTERNAL_FIELDS or field in ENGINE_OPTIONAL_FIELDS:
            question = _question_for_field(
                field,
                draft=draft,
                filled=filled,
                mentioned=mentioned,
                seen=seen,
                required=record.required,
            )
            if question is not None:
                questions.append(question)
    return tuple(questions)


def _fields_to_ask(draft: SolarDecisionDraft, *, mentioned: frozenset[str]) -> tuple[str, ...]:
    """Must/should decision slots always; nice-to-have only while still collecting."""
    must, should, nice = _goal_field_groups(draft)
    unfilled_must = tuple(field for field in must if not _already_satisfied(field, draft))
    unfilled_should = tuple(field for field in should if not _already_satisfied(field, draft))
    unfilled_nice = tuple(field for field in nice if not _already_satisfied(field, draft))
    ask = list(unfilled_must) + [field for field in unfilled_should if field not in unfilled_must]
    if unfilled_must or unfilled_should:
        ask.extend(field for field in unfilled_nice if field not in ask)
    else:
        ask.extend(field for field in unfilled_nice if field in mentioned and field not in ask)
    return tuple(ask)


def _goal_field_groups(
    draft: SolarDecisionDraft,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if draft.mode == "fixed_site":
        return _FIXED_SITE_MUST, _FIXED_SITE_SHOULD, ()
    if draft.site_selection_scope == "candidate_ranking":
        must: tuple[str, ...] = ("candidates",)
        if draft.candidate_ids:
            must = ("candidates", "candidate_geometries")
        return must, _CANDIDATE_RANKING_SHOULD, ()
    area = ("aoi",) if draft.site_selection_scope == "aoi_search" else ("region",)
    return area, _SITE_SELECTION_SHOULD, _SITE_SELECTION_NICE


def _question_for_field(
    field: str,
    *,
    draft: SolarDecisionDraft,
    filled: set[str],
    mentioned: frozenset[str],
    seen: set[str],
    required: bool,
) -> SolarFollowUpQuestion | None:
    if field in PROVIDER_FILLED_FIELDS:
        return None
    if field in ENGINE_INTERNAL_FIELDS and field not in mentioned:
        return None
    if field in filled or _already_satisfied(field, draft):
        return None
    if field not in QUESTION_PROMPTS:
        return None
    prompt_key = _prompt_field(field, draft)
    if prompt_key in seen:
        return None
    seen.add(prompt_key)
    return SolarFollowUpQuestion(
        field=field,
        prompt=QUESTION_PROMPTS[prompt_key],
        required=required,
    )


def _already_satisfied(field: str, draft: SolarDecisionDraft) -> bool:
    if field == "candidates":
        return bool(draft.candidate_ids)
    if field == "budget":
        return draft.budget_amount is not None
    if field == "maximum_grid_distance":
        return draft.maximum_grid_distance_m is not None
    if field in {"site_point", "location"}:
        return draft.location is not None
    if field == "bbox":
        return draft.bbox is not None
    if field == "cell_size":
        return draft.cell_size is not None and draft.cell_size_unit is not None
    if field == "candidate_geometries":
        return not draft.candidate_ids or candidates_have_geometry(draft)
    return getattr(draft, field, None) is not None


def _prompt_field(field: str, draft: SolarDecisionDraft) -> str:
    if field in {"region", "aoi", "site_selection_scope"}:
        return "region" if draft.site_selection_scope != "aoi_search" else "aoi"
    if field == "site_point":
        return "location"
    return field
