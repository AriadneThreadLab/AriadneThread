"""Indicator Catalog contracts.

An indicator is a named scientific measure of a geographic phenomenon. It sits
*above* the Metric Catalog: it selects a closed ``method_id`` which in turn
binds to existing :class:`~app.analytics.contracts.MetricType` primitive(s).

    IndicatorDefinition.method_id  →  MethodSpec  →  MetricType primitive(s)

Formula strings are human documentation. They are never parsed or evaluated.
The LLM may only choose an ``indicator_id`` that exists in the catalog.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.analytics.contracts import (
    MAX_TARGETS,
    AnalysisGoal,
    DecisionBasisType,
    GeoPoint,
    KnowledgeSourceRef,
    LimitationCode,
    MetricDirection,
    MetricRole,
    MetricSelectionEvidence,
    RuleId,
)
from app.analytics.datasets import DatasetRecord
from app.analytics.methods import MethodId
from app.analytics.rules import RULESET_VERSION
from app.osm.query_spec import (
    ALL_ELEMENT_TYPES,
    MAX_RADIUS_METERS,
    ElementType,
    OsmFeatureQuery,
    TagFilter,
)

INDICATOR_ID_PATTERN = r"^[a-z][a-z0-9_]{2,47}$"
REQUIREMENT_ID_PATTERN = r"^[a-z][a-z0-9_]{2,47}$"
CATEGORY_ID_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"

DomainId = Literal["core", "green_space", "mobility", "urban_services", "energy_grid"]
RequirementKind = Literal["osm_features", "reference_point", "analysis_area"]
RequirementSource = Literal["openstreetmap", "user_input", "derived"]
FeatureGeometry = Literal["any", "point", "line", "polygon", "positioned"]
IndicatorStatus = Literal["stable", "experimental", "deprecated"]


def parse_tag_literal(text: str) -> TagFilter:
    """Parse ``key`` or ``key=value`` into the existing OSM TagFilter model."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("tag literal must not be empty")
    if "=" in stripped:
        key, value = stripped.split("=", 1)
        return TagFilter(key=key, value=value)
    return TagFilter(key=stripped, value=None)


class TagCategory(BaseModel):
    """One named partition of an OSM feature set (e.g. park vs wood)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=CATEGORY_ID_PATTERN)
    label: str = Field(min_length=1, max_length=80)
    tags: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    tag_match: Literal["all", "any"] = "all"

    @field_validator("tags")
    @classmethod
    def _validate_tag_literals(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for literal in value:
            parse_tag_literal(literal)
        return value

    def tag_filters(self) -> tuple[TagFilter, ...]:
        """Resolved TagFilter objects for this category."""
        return tuple(parse_tag_literal(literal) for literal in self.tags)


class FeatureSpecification(BaseModel):
    """How to retrieve the subject features for an OSM data requirement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    geometry: FeatureGeometry
    element_types: tuple[ElementType, ...] = Field(min_length=1, max_length=3)
    categories: tuple[TagCategory, ...] = Field(min_length=1)
    documentation: tuple[KnowledgeSourceRef, ...] = Field(default_factory=tuple)

    @field_validator("element_types")
    @classmethod
    def _stable_element_order(cls, value: tuple[ElementType, ...]) -> tuple[ElementType, ...]:
        if len(set(value)) != len(value):
            raise ValueError("element_types must be unique")
        return tuple(kind for kind in ALL_ELEMENT_TYPES if kind in value)


class DataRequirement(BaseModel):
    """One named input an indicator needs before it can be computed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: str = Field(pattern=REQUIREMENT_ID_PATTERN)
    kind: RequirementKind
    source: RequirementSource
    optional: bool = False
    feature_spec: FeatureSpecification | None = None

    @model_validator(mode="after")
    def _require_feature_spec_for_osm(self) -> DataRequirement:
        if self.kind == "osm_features":
            if self.feature_spec is None:
                raise ValueError("osm_features requirements must declare feature_spec")
            if self.source != "openstreetmap":
                raise ValueError("osm_features requirements must use source=openstreetmap")
        elif self.feature_spec is not None:
            raise ValueError("feature_spec is only allowed for osm_features requirements")
        if self.kind == "reference_point" and self.source != "user_input":
            raise ValueError("reference_point requirements must use source=user_input")
        if self.kind == "analysis_area" and self.source != "derived":
            raise ValueError("analysis_area requirements must use source=derived")
        return self


class IndicatorReference(BaseModel):
    """A citable source for the indicator's methodology (not OSM documentation)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation: str = Field(min_length=8, max_length=400)
    url: str | None = Field(default=None, max_length=500)


class IndicatorDefinition(BaseModel):
    """One catalog entry. The LLM may select this id; it may not invent another."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    indicator_id: str = Field(pattern=INDICATOR_ID_PATTERN)
    domain: DomainId
    display_label: str = Field(min_length=1, max_length=80)
    purpose: str = Field(min_length=8, max_length=400)
    goals: tuple[AnalysisGoal, ...] = Field(min_length=1)
    method_id: MethodId
    formula: str = Field(min_length=1, max_length=400)
    parameters: Mapping[str, float] = Field(default_factory=dict)
    unit: str = Field(min_length=1, max_length=40)
    display_precision: int = Field(ge=0, le=6)
    direction: MetricDirection
    requirements: tuple[DataRequirement, ...] = Field(min_length=1)
    min_observations: int = Field(default=1, ge=0)
    limitations: tuple[LimitationCode, ...] = Field(default_factory=tuple)
    references: tuple[IndicatorReference, ...] = Field(default_factory=tuple)
    status: IndicatorStatus = "experimental"
    deprecated_by: str | None = Field(default=None, pattern=INDICATOR_ID_PATTERN)
    since_catalog_version: str = Field(min_length=1, max_length=40)

    @field_validator("goals")
    @classmethod
    def _unique_goals(cls, value: tuple[AnalysisGoal, ...]) -> tuple[AnalysisGoal, ...]:
        if len(set(value)) != len(value):
            raise ValueError("goals must be unique")
        return value

    @model_validator(mode="after")
    def _unique_requirement_ids(self) -> IndicatorDefinition:
        ids = [item.requirement_id for item in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("requirement_id values must be unique")
        if self.status == "deprecated" and self.deprecated_by is None:
            raise ValueError("deprecated indicators must set deprecated_by")
        if self.status != "deprecated" and self.deprecated_by is not None:
            raise ValueError("deprecated_by is only allowed when status is deprecated")
        return self


# --- Selection / planning (Phase 2). Sits above MetricSelectionTrace. ---

#: Evidence for domain/goal/indicator choice reuses the metric evidence shape
#: (basis, optional RuleId, statement, source_ref). It does not wrap a
#: MetricSelectionTrace: that object records compute-time metric feasibility.
IndicatorSelectionEvidence = MetricSelectionEvidence

IndicatorRejectionReason = Literal[
    "indicator_not_in_catalog",
    "goal_mismatch",
    "domain_mismatch",
    "semantic_rule_violation",
    "deprecated",
    "not_selected",
]

MAX_SUPPORTING_INDICATORS = 3


class AnalysisDomainSelection(BaseModel):
    """Detected analysis domain(s). Empty thematic set falls back to core."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    domains: tuple[DomainId, ...] = Field(min_length=1)
    basis: DecisionBasisType
    evidence: tuple[IndicatorSelectionEvidence, ...] = ()


class IndicatorCandidate(BaseModel):
    """One catalog indicator considered for a request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    indicator_id: str = Field(pattern=INDICATOR_ID_PATTERN)
    domain: DomainId
    display_label: str
    goals: tuple[AnalysisGoal, ...]
    method_id: MethodId
    eligible: bool
    rank: int = Field(ge=0)
    ineligibility_reason: IndicatorRejectionReason | None = None


class ProposedIndicatorChoice(BaseModel):
    """Optional LLM proposal. IDs are validated against the catalog, never trusted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_indicator_id: str = Field(pattern=INDICATOR_ID_PATTERN)
    supporting_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=MAX_SUPPORTING_INDICATORS
    )
    selection_reason: str = Field(default="", max_length=240)

    @model_validator(mode="after")
    def _unique_ids(self) -> ProposedIndicatorChoice:
        ids = (self.primary_indicator_id, *self.supporting_indicator_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("primary and supporting indicator ids must be unique")
        for item in self.supporting_indicator_ids:
            if not re.fullmatch(INDICATOR_ID_PATTERN, item):
                raise ValueError(f"invalid supporting indicator_id '{item}'")
        return self


class IndicatorSelection(BaseModel):
    """Validated primary plus optional supporting indicators."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_indicator_id: str | None = Field(default=None, pattern=INDICATOR_ID_PATTERN)
    supporting_indicator_ids: tuple[str, ...] = Field(
        default_factory=tuple, max_length=MAX_SUPPORTING_INDICATORS
    )
    primary_role: MetricRole = "primary"


class IndicatorRejection(BaseModel):
    """Why a candidate or proposal was not selected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    indicator_id: str
    reason: IndicatorRejectionReason
    detail: str = Field(max_length=240)


class IndicatorSelectionTrace(BaseModel):
    """Deterministic provenance for domain detection and indicator selection.

    This is not a MetricSelectionTrace: metric traces remain the compute-time
    record of feasibility checks and engine provenance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    inferred_goal: AnalysisGoal
    goal_explicit: bool
    domain_selection: AnalysisDomainSelection
    candidate_indicators: tuple[str, ...]
    candidates: tuple[IndicatorCandidate, ...]
    rejected_indicators: tuple[IndicatorRejection, ...]
    selection: IndicatorSelection
    selection_reason: str = Field(max_length=240)
    selection_evidence: tuple[IndicatorSelectionEvidence, ...]
    required_data: tuple[str, ...]
    methodology: MethodId | None = None
    parameters: Mapping[str, float] = Field(default_factory=dict)
    execution_status: Literal["completed", "rejected"]
    indicator_catalog_version: str
    ruleset_version: str = RULESET_VERSION
    cited_rule_ids: tuple[RuleId, ...] = ()


# --- OSM data requirement planning (Phase 3). Does not retrieve features. ---

GroundingSource = Literal["catalog_declared", "rag_grounding"]
PlannedRequirementKind = Literal["osm_features", "reference_point", "analysis_area"]


class PlanningTarget(BaseModel):
    """One analysis location. Coordinates are never accepted here (spatial trust)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    label: str = Field(min_length=1, max_length=80)
    place_ref: str | None = Field(
        default=None,
        pattern=r"^place_[1-9][0-9]*$",
        description="Trusted resolve_place handle. The model does not supply lat/lon.",
    )
    radius_m: int | None = Field(default=None, gt=0, le=MAX_RADIUS_METERS)
    place: str | None = Field(default=None, min_length=1, max_length=120)

    @model_validator(mode="after")
    def _exactly_one_trusted_scope(self) -> PlanningTarget:
        has_ref = self.place_ref is not None
        has_place = self.place is not None
        if has_ref == has_place:
            raise ValueError("exactly one of place_ref or place must be provided")
        if has_ref and self.radius_m is None:
            raise ValueError("place_ref scope requires radius_m from the user request")
        if has_place and self.radius_m is not None:
            raise ValueError("named place scope must not include radius_m")
        return self


class RagGroundingEvidence(BaseModel):
    """Offline OSM-RAG evidence the planner may attach. This is not a live search."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: str | None = Field(default=None, pattern=REQUIREMENT_ID_PATTERN)
    tags: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    documentation: tuple[KnowledgeSourceRef, ...] = Field(default_factory=tuple)

    @field_validator("tags")
    @classmethod
    def _validate_tag_literals(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for literal in value:
            parse_tag_literal(literal)
        return value


class DataPlanningRequest(BaseModel):
    """Inputs the planner accepts. Extra keys (Overpass QL, invented tags) are forbidden."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    indicator_ids: tuple[str, ...] = Field(min_length=1, max_length=MAX_SUPPORTING_INDICATORS + 1)
    targets: tuple[PlanningTarget, ...] = Field(min_length=1, max_length=MAX_TARGETS)
    rag_grounding: tuple[RagGroundingEvidence, ...] = ()

    @field_validator("indicator_ids")
    @classmethod
    def _unique_indicator_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("indicator_ids must be unique")
        for item in value:
            if not re.fullmatch(INDICATOR_ID_PATTERN, item):
                raise ValueError(f"invalid indicator_id '{item}'")
        return value


class GroundedOsmConcept(BaseModel):
    """How a required data concept is represented in OSM, with documentation provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: str
    concept: str
    tags: tuple[str, ...]
    documentation: tuple[KnowledgeSourceRef, ...]
    grounding_source: GroundingSource
    requested_by: tuple[str, ...]


class PlannedOsmDataset(BaseModel):
    """One unique OSM retrieval to be executed later by query_osm (not by this planner)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_key: str = Field(pattern=r"^planned_osm_[1-9][0-9]*$")
    requirement_id: str
    target_id: str
    query: OsmFeatureQuery
    requested_by: tuple[str, ...]
    documentation: tuple[KnowledgeSourceRef, ...]
    grounding_source: GroundingSource
    retrieval_tool: Literal["query_osm"] = "query_osm"


class PlannedReferenceLocation(BaseModel):
    """A trusted place_ref that resolve_place already (or will) populate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: str
    target_id: str
    place_ref: str
    requested_by: tuple[str, ...]
    retrieval_tool: Literal["resolve_place"] = "resolve_place"


class PlannedAnalysisBoundary(BaseModel):
    """Analysis area derived from the target scope. No extra Overpass call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: str
    target_id: str
    requested_by: tuple[str, ...]
    source: Literal["derived"] = "derived"


class IndicatorRequirementBinding(BaseModel):
    """Per-indicator view: catalog requirements and the planned datasets that satisfy them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    indicator_id: str
    requirement_ids: tuple[str, ...]
    dataset_keys: tuple[str, ...]
    reference_target_ids: tuple[str, ...]
    analysis_area_target_ids: tuple[str, ...]


class DataRequirementPlan(BaseModel):
    """Traceable OSM data plan for selected indicators. Retrieval is out of scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_indicators: tuple[str, ...]
    required_data: tuple[str, ...]
    indicator_bindings: tuple[IndicatorRequirementBinding, ...]
    grounded_concepts: tuple[GroundedOsmConcept, ...]
    planned_datasets: tuple[PlannedOsmDataset, ...]
    reference_locations: tuple[PlannedReferenceLocation, ...]
    analysis_boundaries: tuple[PlannedAnalysisBoundary, ...]
    documentation: tuple[KnowledgeSourceRef, ...]
    retrieved_dataset_refs: tuple[str, ...] = ()
    consolidated_requirement_ids: tuple[str, ...]
    indicator_catalog_version: str
    execution_status: Literal["completed"] = "completed"


# --- Deterministic indicator computation (executors). ---

IndicatorComputeStatus = Literal[
    "computed",
    "insufficient_data",
    "not_applicable",
    "unimplemented",
]


class IndicatorComputeRequest(BaseModel):
    """Inputs for one indicator executor. Features are already retrieved GeoJSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    indicator_id: str = Field(pattern=INDICATOR_ID_PATTERN)
    datasets: dict[str, DatasetRecord]
    reference_points: tuple[GeoPoint, ...] = ()


class IndicatorComputationResult(BaseModel):
    """Structured output of one deterministic indicator calculation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    indicator_id: str = Field(pattern=INDICATOR_ID_PATTERN)
    value: float | None = None
    unit: str
    observation_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    methodology: MethodId
    dataset_refs: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    status: IndicatorComputeStatus
