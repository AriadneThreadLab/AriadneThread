"""SolarDSS workflow snapshots for Active Learning.

Active Learning stores routing, question-selection and preference labels. It
never recomputes PV physics and never replaces PVGIS, pvlib, terrain, GIS
distance or MCDA formulas. Indicator numbers, when present, are provenance
copies from SolarDSS and are excluded from the export target.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.active_learning.sanitization import sanitize_text
from app.solar.contracts import (
    PROVIDER_FILLED_FIELDS,
    SolarDecisionMode,
    SolarDecisionReport,
    SolarProjectType,
    SolarSiteSelectionScope,
)

ANALYZE_SOLAR_SITE_TOOL = "analyze_solar_site"
SEARCH_SOLAR_KNOWLEDGE_TOOL = "search_solar_knowledge"

SCIENCE_NOT_COMPUTED: Literal["not_computed"] = "not_computed"
SCIENCE_SOLARDSS_COPY: Literal["solardss_copy"] = "solardss_copy"

DOES_NOT_REPLACE: tuple[str, ...] = (
    "PVGIS",
    "pvlib",
    "terrain calculations",
    "GIS distances",
    "MCDA formulas",
)

INDICATORS_NOTE = (
    "Active learning does not compute or replace SolarDSS indicators. "
    "Values here are provenance copies only and are never training targets."
)

#: Dict keys that would smuggle PV physics into a learning target.
FORBIDDEN_PHYSICS_KEYS: frozenset[str] = PROVIDER_FILLED_FIELDS | frozenset(
    {
        "pvgis",
        "pvlib",
        "mcda",
        "terrain",
        "terrain_slope",
        "annual_yield",
        "yield_kwh",
        "yield_kwh_per_kwp",
        "irradiance",
        "poa_irradiance",
        "capacity_factor",
        "performance_ratio",
    }
)


class SolarFeedbackKind(str, Enum):
    """Bounded expert-feedback kinds for SolarDSS workflows."""

    RANKING_CORRECTION = "ranking_correction"
    CANDIDATE_ACCEPT_REJECT = "candidate_accept_reject"
    PREFERENCE_CORRECTION = "preference_correction"
    INTENT_CLASSIFICATION = "intent_classification"
    CAPABILITY_SELECTION = "capability_selection"
    MISSING_DATA = "missing_data"
    RANKING_USEFULNESS = "ranking_usefulness"


class SolarRankingEntry(BaseModel):
    """One candidate's rank or accept/reject label. Never an indicator value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1, max_length=80)
    original_rank: int | None = Field(default=None, ge=1, le=10_000)
    decision: Literal["accept", "reject"] | None = None
    corrected_rank: int | None = Field(default=None, ge=1, le=10_000)
    reason: str | None = Field(default=None, max_length=400)

    @field_validator("reason")
    @classmethod
    def _clean_reason(cls, value: str | None) -> str | None:
        return None if value is None else sanitize_text(value, field="ranking_reason")


class SolarExpertFeedback(BaseModel):
    """Structured expert review of a SolarDSS routing or ranking turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: SolarFeedbackKind
    labels: list[SolarRankingEntry] = Field(default_factory=list)
    corrected_capability_id: str | None = Field(default=None, max_length=48)
    corrected_mode: SolarDecisionMode | None = None
    corrected_site_selection_scope: SolarSiteSelectionScope | None = None
    corrected_question_fields: list[str] = Field(default_factory=list)
    corrected_preference: dict[str, Any] | None = None
    preference_label: str | None = Field(default=None, max_length=120)
    ranking_useful: bool | None = None
    reason: str | None = Field(default=None, max_length=400)

    @field_validator("reason", "preference_label")
    @classmethod
    def _clean_text(cls, value: str | None) -> str | None:
        return None if value is None else sanitize_text(value, field="solar_feedback")

    @field_validator("corrected_question_fields")
    @classmethod
    def _clean_questions(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return cleaned

    @field_validator("corrected_preference")
    @classmethod
    def _no_physics_preference(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        found = physics_keys_in_payload(value)
        if found:
            raise ValueError(
                "corrected_preference must not contain PV physics keys: " + ", ".join(found)
            )
        return value


class SolarIndicatorProvenance(BaseModel):
    """Copy of SolarDSS indicators, or an explicit not-computed record.

    Active Learning must not treat ``values`` as labels or recompute them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    science: Literal["not_computed", "solardss_copy"] = SCIENCE_NOT_COMPUTED
    source: str = "analyze_solar_site"
    values: dict[str, Any] = Field(default_factory=dict)
    note: str = INDICATORS_NOTE


class SolarRankingProfile(BaseModel):
    """Candidates the model ranked or that the expert labelled."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_ids: list[str] = Field(default_factory=list)
    model_ranking: list[SolarRankingEntry] = Field(default_factory=list)
    capability_id: str | None = None


class SolarPlanningDecision(BaseModel):
    """What the model chose to do: intent, capability, tools, questions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_id: str | None = None
    mode: str | None = None
    site_selection_scope: str | None = None
    project_type: str | None = None
    tool_sequence: list[str] = Field(default_factory=list)
    questions_asked: list[str] = Field(default_factory=list)
    filled_fields: list[str] = Field(default_factory=list)
    missing_requirement_fields: list[str] = Field(default_factory=list)
    status: str | None = None


class SolarWorkflowProvenance(BaseModel):
    """Where the snapshot came from, and what Active Learning must not replace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    science_not_from_active_learning: Literal[True] = True
    does_not_replace: tuple[str, ...] = DOES_NOT_REPLACE
    requirements_source: str | None = None
    engine: str = "SolarDSS"


class SolarWorkflowSnapshot(BaseModel):
    """Review payload for one SolarDSS turn.

    Stores the original structured decision, indicator provenance, ranking
    profile, model planning decision, expert feedback, corrected preference
    and provenance. Indicator ``values`` are never exported as training targets.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    structured_decision: dict[str, Any] = Field(default_factory=dict)
    indicators: SolarIndicatorProvenance = Field(default_factory=SolarIndicatorProvenance)
    ranking_profile: SolarRankingProfile = Field(default_factory=SolarRankingProfile)
    model_planning_decision: SolarPlanningDecision = Field(default_factory=SolarPlanningDecision)
    expert_feedback: SolarExpertFeedback | None = None
    corrected_preference: dict[str, Any] | None = None
    provenance: SolarWorkflowProvenance = Field(default_factory=SolarWorkflowProvenance)

    @field_validator("structured_decision", "corrected_preference")
    @classmethod
    def _no_physics_slots(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        found = physics_keys_in_payload(value)
        if found:
            raise ValueError(
                "solar workflow slots must not hold PV physics keys: " + ", ".join(found)
            )
        return value


class SolarDecisionLearningTarget(BaseModel):
    """Exportable SFT target: routing, questions, preference and ranking labels.

    This is not an AnalysisPlan and not a SolarDSS indicator record. ``science``
    is always ``not_computed`` so a physics payload cannot masquerade as a label.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    science: Literal["not_computed"] = SCIENCE_NOT_COMPUTED
    mode: SolarDecisionMode | None = None
    site_selection_scope: SolarSiteSelectionScope | None = None
    project_type: SolarProjectType | None = None
    capability_id: str | None = Field(default=None, max_length=48)
    question_fields: list[str] = Field(default_factory=list)
    ranking_labels: list[SolarRankingEntry] = Field(default_factory=list)
    preference_label: str | None = Field(default=None, max_length=120)
    preference: dict[str, Any] | None = None
    does_not_replace: tuple[str, ...] = DOES_NOT_REPLACE

    @field_validator("preference")
    @classmethod
    def _no_physics_preference(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        found = physics_keys_in_payload(value)
        if found:
            raise ValueError("preference must not contain PV physics keys: " + ", ".join(found))
        return value

    @model_validator(mode="after")
    def _reject_physics_payload(self) -> SolarDecisionLearningTarget:
        found = physics_keys_in_payload(self.model_dump(mode="python"))
        if found:
            raise ValueError(
                "SolarDecisionLearningTarget must not contain PV physics keys: " + ", ".join(found)
            )
        return self


def is_physics_key(name: str) -> bool:
    """Whether ``name`` is a forbidden PV-physics or MCDA payload key."""
    return name.strip().lower() in FORBIDDEN_PHYSICS_KEYS


def physics_keys_in_payload(value: Any, *, path: str = "$") -> list[str]:
    """Return dotted paths whose dict keys are PV physics, excluding provenance copies."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if is_physics_key(str(key)) and not _is_indicator_copy_path(child):
                found.append(child)
            found.extend(physics_keys_in_payload(item, path=child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(physics_keys_in_payload(item, path=f"{path}[{index}]"))
    return found


def asked_provider_filled_fields(field_names: list[str] | tuple[str, ...]) -> list[str]:
    """Question or missing-requirement fields that SolarDSS providers already fill."""
    found: list[str] = []
    for name in field_names:
        token = name.strip().lower()
        if token in PROVIDER_FILLED_FIELDS:
            found.append(name)
    return found


def split_decision_and_indicators(
    problem: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy slots for planning; move leaked indicator keys into provenance."""
    slots: dict[str, Any] = {}
    values: dict[str, Any] = {}
    for key, item in problem.items():
        if is_physics_key(str(key)):
            values[key] = item
        else:
            slots[key] = item
    return slots, values


def snapshot_solar_workflow(
    report: SolarDecisionReport | None,
    *,
    tool_sequence: tuple[str, ...] | list[str] = (),
) -> SolarWorkflowSnapshot | None:
    """Build a review snapshot from a routing report. Does not compute indicators."""
    if report is None:
        return None
    slots, indicator_values = split_decision_and_indicators(dict(report.problem))
    science: Literal["not_computed", "solardss_copy"] = (
        SCIENCE_SOLARDSS_COPY if indicator_values else SCIENCE_NOT_COMPUTED
    )
    candidate_ids = _candidate_ids_from_problem(slots)
    questions = [item.field for item in report.questions]
    missing = [item.field for item in report.missing_requirements]
    return SolarWorkflowSnapshot(
        structured_decision=slots,
        indicators=SolarIndicatorProvenance(
            science=science,
            source=report.requirements_source,
            values=indicator_values,
            note=INDICATORS_NOTE,
        ),
        ranking_profile=SolarRankingProfile(
            candidate_ids=candidate_ids,
            capability_id=report.capability_id,
        ),
        model_planning_decision=SolarPlanningDecision(
            capability_id=report.capability_id,
            mode=report.mode,
            site_selection_scope=report.site_selection_scope,
            project_type=report.project_type,
            tool_sequence=list(tool_sequence),
            questions_asked=questions,
            filled_fields=list(report.filled_fields),
            missing_requirement_fields=missing,
            status=report.status,
        ),
        provenance=SolarWorkflowProvenance(
            requirements_source=report.requirements_source,
        ),
    )


def empty_solar_workflow(
    *, tool_sequence: tuple[str, ...] | list[str] = ()
) -> SolarWorkflowSnapshot:
    """Minimal snapshot when a solar decision was expected but no report exists."""
    return SolarWorkflowSnapshot(
        model_planning_decision=SolarPlanningDecision(tool_sequence=list(tool_sequence)),
    )


def merge_expert_feedback(
    snapshot: SolarWorkflowSnapshot | None,
    feedback: SolarExpertFeedback,
    *,
    tool_sequence: tuple[str, ...] | list[str] = (),
) -> SolarWorkflowSnapshot:
    """Attach expert labels without altering indicator provenance copies."""
    base = snapshot if snapshot is not None else empty_solar_workflow(tool_sequence=tool_sequence)
    ranking = base.ranking_profile
    ids = list(ranking.candidate_ids)
    for label in feedback.labels:
        if label.candidate_id not in ids:
            ids.append(label.candidate_id)
    ranking = ranking.model_copy(update={"candidate_ids": ids})
    planning = base.model_planning_decision
    planning_updates: dict[str, object] = {}
    if feedback.corrected_capability_id:
        planning_updates["capability_id"] = feedback.corrected_capability_id
        ranking = ranking.model_copy(update={"capability_id": feedback.corrected_capability_id})
    if feedback.corrected_mode is not None:
        planning_updates["mode"] = feedback.corrected_mode
    if feedback.corrected_site_selection_scope is not None:
        planning_updates["site_selection_scope"] = feedback.corrected_site_selection_scope
    if feedback.corrected_question_fields:
        planning_updates["questions_asked"] = list(feedback.corrected_question_fields)
    if planning_updates:
        planning = planning.model_copy(update=planning_updates)
    preference = (
        feedback.corrected_preference
        if feedback.corrected_preference is not None
        else base.corrected_preference
    )
    return base.model_copy(
        update={
            "ranking_profile": ranking,
            "model_planning_decision": planning,
            "expert_feedback": feedback,
            "corrected_preference": preference,
        }
    )


def solar_feedback_has_correction(feedback: SolarExpertFeedback) -> bool:
    """Whether the expert supplied a label that can become a training target."""
    if feedback.corrected_capability_id or feedback.corrected_mode is not None:
        return True
    if feedback.corrected_site_selection_scope is not None:
        return True
    if feedback.corrected_question_fields or feedback.corrected_preference:
        return True
    if feedback.preference_label:
        return True
    return any(
        item.decision is not None or item.corrected_rank is not None for item in feedback.labels
    )


def learning_target_from_workflow(snapshot: SolarWorkflowSnapshot) -> SolarDecisionLearningTarget:
    """Planning/preference target. Indicator ``values`` are dropped on purpose."""
    planning = snapshot.model_planning_decision
    feedback = snapshot.expert_feedback
    mode = _as_mode(planning.mode)
    scope = _as_scope(planning.site_selection_scope)
    project_type = _as_project_type(planning.project_type)
    capability = planning.capability_id
    questions = list(planning.questions_asked)
    labels: list[SolarRankingEntry] = []
    preference = snapshot.corrected_preference
    preference_label = None
    if feedback is not None:
        if feedback.corrected_mode is not None:
            mode = feedback.corrected_mode
        if feedback.corrected_site_selection_scope is not None:
            scope = feedback.corrected_site_selection_scope
        if feedback.corrected_capability_id:
            capability = feedback.corrected_capability_id
        if feedback.corrected_question_fields:
            questions = list(feedback.corrected_question_fields)
        labels = list(feedback.labels)
        if feedback.corrected_preference is not None:
            preference = feedback.corrected_preference
        preference_label = feedback.preference_label
    return SolarDecisionLearningTarget(
        science=SCIENCE_NOT_COMPUTED,
        mode=mode,
        site_selection_scope=scope,
        project_type=project_type,
        capability_id=capability,
        question_fields=questions,
        ranking_labels=labels,
        preference_label=preference_label,
        preference=preference,
        does_not_replace=DOES_NOT_REPLACE,
    )


def _is_indicator_copy_path(path: str) -> bool:
    return ".indicators.values." in path or path.startswith("$.indicators.values")


def _candidate_ids_from_problem(problem: dict[str, Any]) -> list[str]:
    candidates = problem.get("candidates")
    raw: Any = None
    if isinstance(candidates, dict):
        raw = candidates.get("candidate_ids")
    elif isinstance(candidates, list):
        raw = candidates
    if raw is None:
        raw = problem.get("candidate_ids")
    if not isinstance(raw, list):
        return []
    found: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in found:
            found.append(text)
    return found


def _as_mode(value: str | None) -> SolarDecisionMode | None:
    if value == "site_selection":
        return "site_selection"
    if value == "fixed_site":
        return "fixed_site"
    return None


def _as_scope(value: str | None) -> SolarSiteSelectionScope | None:
    if value == "region_search":
        return "region_search"
    if value == "aoi_search":
        return "aoi_search"
    if value == "candidate_ranking":
        return "candidate_ranking"
    return None


def _as_project_type(value: str | None) -> SolarProjectType | None:
    if value == "utility_scale":
        return "utility_scale"
    if value == "ground_mounted":
        return "ground_mounted"
    if value == "rooftop":
        return "rooftop"
    if value == "commercial_rooftop":
        return "commercial_rooftop"
    if value == "residential_rooftop":
        return "residential_rooftop"
    return None
