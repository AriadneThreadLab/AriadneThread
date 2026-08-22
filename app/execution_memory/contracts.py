"""Execution Memory domain contracts.

Snapshots are structured provenance, never hidden chain-of-thought.
Request-scoped ``osm_result_N`` / ``place_N`` identifiers are not durable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.analytics.contracts import AnalysisGoal, MetricType
from app.analytics.datasets import DatasetScope
from app.osm.contracts import GeoJsonFeatureCollection

SCHEMA_VERSION = "execution-memory-1"

FollowUpType = Literal[
    "NEW_ANALYSIS",
    "ADD_TARGET",
    "REMOVE_TARGET",
    "REPLACE_TARGETS",
    "CHANGE_GOAL",
    "CHANGE_FEATURE_CONCEPT",
    "CHANGE_SCOPE",
    "CHANGE_RADIUS",
    "CHANGE_METRIC_REQUEST",
    "REFRESH_DATA",
    "RECOMPARE",
    "AMBIGUOUS_FOLLOW_UP",
]

ReuseDecision = Literal[
    "FULL_REUSE",
    "PARTIAL_REUSE",
    "REVALIDATE_METRIC",
    "REFRESH_DATA",
    "FULL_REPLAN",
    "CANNOT_REUSE",
]

DatasetSource = Literal["reused", "refreshed", "new"]

MetricRevalidationStatus = Literal["accepted", "rejected", "not_applicable"]

ExecutionOutcome = Literal["completed", "partial", "failed"]

#: Category of the *kind* of place a comparison is built around. Reusable
#: Analysis Memory stores the category, never the landmark names themselves.
TargetCategory = Literal[
    "educational_facility",
    "transport_hub",
    "healthcare_facility",
    "civic_landmark",
    "place",
]

#: Validation rules a reusable pattern promises to enforce before reuse.
PATTERN_VALIDATION_RULES: tuple[str, ...] = (
    "compatible_targets",
    "comparable_scope",
    "metric_semantics",
    "dataset_completeness",
)

#: Methodology components a pattern can contribute to a new execution.
PatternComponent = Literal[
    "dataset_definition",
    "radius",
    "metric",
    "validation_rules",
    "targets",
    "place_resolution",
    "osm_query",
    "metric_values",
]

HIDDEN_REASONING_KEYS = frozenset(
    {
        "rationale",
        "reasoning",
        "reasoning_content",
        "chain_of_thought",
        "hidden_reasoning",
        "think",
        "<think",
    }
)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def strip_hidden_reasoning(payload: Any) -> Any:
    """Drop keys that look like model reasoning. Used before persistence."""
    if isinstance(payload, dict):
        return {
            str(key): strip_hidden_reasoning(value)
            for key, value in payload.items()
            if str(key).lower() not in HIDDEN_REASONING_KEYS and "<think" not in str(key).lower()
        }
    if isinstance(payload, list):
        return [strip_hidden_reasoning(item) for item in payload]
    return payload


class TrustedPlaceSnapshot(BaseModel):
    """Nominatim-backed place resolution stored for reuse (not LLM memory)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    label: str
    display_name: str
    latitude: float
    longitude: float
    source: str
    source_id: str


class ExecutionTargetSnapshot(BaseModel):
    """One comparison target in a persisted execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stable_id: str = Field(pattern=r"^t[1-9]$")
    label: str = Field(min_length=1, max_length=120)
    original_user_label: str = Field(min_length=1, max_length=120)
    place: TrustedPlaceSnapshot
    request_scoped_place_ref_at_creation: str | None = Field(
        default=None,
        description="Ephemeral place_N from the originating request; not a durable id.",
    )


class ExecutionDatasetSnapshot(BaseModel):
    """Persistent OSM dataset. Distinct from request-scoped dataset_ref."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_dataset_id: UUID = Field(default_factory=uuid4)
    target_stable_id: str
    retrieved_at: datetime
    source: str = "live_osm"
    endpoint: str = ""
    attribution: str = ""
    effective_limit: int = Field(ge=1)
    truncated: bool = False
    feature_count: int = Field(ge=0)
    resolved_tags: tuple[str, ...] = ()
    scope: DatasetScope
    query_spec: dict[str, Any] = Field(default_factory=dict)
    feature_collection: GeoJsonFeatureCollection
    request_scoped_dataset_ref_at_creation: str | None = Field(
        default=None,
        description="Ephemeral osm_result_N from the originating request; not reused as identity.",
    )

    @field_validator("query_spec")
    @classmethod
    def _no_durable_request_ref(cls, value: dict[str, Any]) -> dict[str, Any]:
        cleaned = strip_hidden_reasoning(value)
        if not isinstance(cleaned, dict):
            return {}
        # Never treat request-scoped refs as query identity.
        cleaned.pop("dataset_ref", None)
        return cleaned


class MetricSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: MetricType | None = None
    role: str = "primary"
    inferred_goal: AnalysisGoal | None = None
    rule_id: str | None = None
    catalog_version: str = ""
    ruleset_version: str = ""
    feasibility_ok: bool = True


class DocumentationSourceSnapshot(BaseModel):
    """OSM Wiki citation that grounded a reused tag definition.

    Provenance only: title, section, URL, and retrieval score. Passage bodies
    stay in the RAG corpus and are never persisted as execution memory.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1, max_length=160)
    section: str | None = Field(default=None, max_length=160)
    url: str = Field(min_length=8, max_length=1024)
    score: float = 0.0


class ComparisonSnapshot(BaseModel):
    """Deterministic comparison summary only (no model reasoning)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_statement: str | None = None
    overall_confidence: str | None = None
    ranking: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    truncated: bool = False


class ExecutionSnapshot(BaseModel):
    """One immutable analytical execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: UUID = Field(default_factory=uuid4)
    conversation_id: str = Field(min_length=8, max_length=64)
    request_id: str | None = None
    parent_execution_id: UUID | None = None
    created_at: datetime = Field(default_factory=_now)
    original_user_query: str = Field(min_length=2, max_length=2000)
    analysis_type: str = "comparison"
    analysis_goal: str = ""
    feature_concept: str = ""
    inferred_goal: AnalysisGoal | None = None
    outcome: ExecutionOutcome = "completed"
    radius_m: int | None = Field(default=None, gt=0, le=50_000)
    scope_kind: str = "point"
    grounding_tags: tuple[str, ...] = ()
    grounding_version: str = ""
    metric: MetricSnapshot = Field(default_factory=MetricSnapshot)
    targets: list[ExecutionTargetSnapshot] = Field(default_factory=list)
    datasets: list[ExecutionDatasetSnapshot] = Field(default_factory=list)
    documentation_sources: tuple[DocumentationSourceSnapshot, ...] = ()
    comparison: ComparisonSnapshot | None = None
    model: str = ""
    provider: str | None = None
    prompt_version: str | None = None
    tool_schema_version: str | None = None
    schema_version: str = SCHEMA_VERSION


class AnalysisPattern(BaseModel):
    """Reusable Analysis Memory: methodology only, never a stored answer.

    A pattern describes *how* a class of question was answered successfully
    (dataset definition, scope, metric, validation rules). It deliberately
    carries no landmark names, coordinates, feature collections or rankings —
    those live in the immutable execution history and are never replayed as
    the answer to a new question.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pattern_key: str = Field(min_length=3, max_length=200)
    analysis_pattern: str = Field(min_length=3, max_length=200)
    analysis_type: str = "comparison"
    feature_concept: str = Field(min_length=1, max_length=120)
    target_category: TargetCategory = "place"
    observed_target_count: int = Field(default=0, ge=0, le=8)
    scope_kind: str = "point"
    dataset_tags: tuple[str, ...] = ()
    radius_m: int | None = Field(default=None, gt=0, le=50_000)
    metric: MetricType | None = None
    inferred_goal: AnalysisGoal | None = None
    metric_rule_id: str | None = None
    metric_catalog_version: str = ""
    ruleset_version: str = ""
    validation_rules: tuple[str, ...] = PATTERN_VALIDATION_RULES
    source_execution_id: UUID | None = None
    observed_at: datetime = Field(default_factory=_now)
    outcome: ExecutionOutcome = "completed"


class PatternCheck(BaseModel):
    """One deterministic reuse precondition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    detail: str = ""


class PatternReuseAssessment(BaseModel):
    """Whether a stored methodology may be applied to the current request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pattern_key: str | None = None
    analysis_pattern: str | None = None
    reusable: bool = False
    checks: tuple[PatternCheck, ...] = ()
    reused_components: tuple[PatternComponent, ...] = ()
    recomputed_components: tuple[PatternComponent, ...] = ()

    def check(self, name: str) -> PatternCheck | None:
        for item in self.checks:
            if item.name == name:
                return item
        return None

    @property
    def blocks_reuse(self) -> bool:
        """True when the data-defining part of the methodology is invalid.

        A failed metric check does not block reuse: the metric is replanned
        while the dataset definition and scope stay valid.
        """
        for name in ("dataset_definition_valid", "targets_comparable", "required_data_available"):
            item = self.check(name)
            if item is not None and not item.passed:
                return True
        return False


class MetricRevalidation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    previous_metric: MetricType | None = None
    new_goal: AnalysisGoal | None = None
    status: MetricRevalidationStatus = "not_applicable"
    reason: str = ""
    replacement_metric: MetricType | None = None


class TargetDatasetAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stable_id: str
    label: str
    source: DatasetSource
    reasons: tuple[str, ...] = ()
    persistent_dataset_id: UUID | None = None


class FollowUpResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    follow_up_type: FollowUpType
    extra_intents: tuple[FollowUpType, ...] = ()
    added_labels: tuple[str, ...] = ()
    removed_labels: tuple[str, ...] = ()
    preserved_labels: tuple[str, ...] = ()
    requested_labels: tuple[str, ...] = ()
    new_radius_m: int | None = None
    new_goal: AnalysisGoal | None = None
    new_feature_concept: str | None = None
    refresh_requested: bool = False
    keep_previous_data: bool = False
    reasons: tuple[str, ...] = ()


class ReuseAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: ReuseDecision
    reasons: tuple[str, ...] = ()
    metric_revalidation: MetricRevalidation
    target_actions: tuple[TargetDatasetAction, ...] = ()
    feature_concept: str
    comparison_goal: str
    inferred_goal: AnalysisGoal
    radius_m: int
    grounding_tags: tuple[str, ...]
    re_ground: bool = False
    final_metric: MetricType | None = None


class IncrementalTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stable_id: str
    label: str
    place_query: str
    source: DatasetSource
    place: TrustedPlaceSnapshot | None = None
    persistent_dataset_id: UUID | None = None


class IncrementalComparisonPlan(BaseModel):
    """Backend plan for a follow-up comparison (Tool Registry still executes)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_execution_id: UUID
    follow_up: FollowUpResolution
    reuse: ReuseAssessment
    targets: tuple[IncrementalTarget, ...]
    pattern: AnalysisPattern | None = None
    pattern_reuse: PatternReuseAssessment | None = None
    user_facing_preamble: str = ""


class ExecutionMemoryTrace(BaseModel):
    """Safe provenance for Technical Panels. No chain-of-thought."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    memory_execution_id: str | None = None
    parent_execution_id: str | None = None
    memory_reuse_attempted: bool = False
    memory_reuse_decision: ReuseDecision | None = None
    memory_reuse_reasons: tuple[str, ...] = ()
    follow_up_type: FollowUpType | None = None
    reused_targets: tuple[str, ...] = ()
    refreshed_targets: tuple[str, ...] = ()
    new_targets: tuple[str, ...] = ()
    previous_metric: str | None = None
    metric_revalidation_status: MetricRevalidationStatus | None = None
    metric_revalidation_reason: str | None = None
    final_metric: str | None = None
    previous_scope: str | None = None
    final_scope: str | None = None
    dataset_freshness_checks: tuple[str, ...] = ()
    analysis_pattern: str | None = None
    pattern_key: str | None = None
    pattern_reusable: bool = False
    pattern_checks: tuple[str, ...] = ()
    reused_components: tuple[str, ...] = ()
    recomputed_components: tuple[str, ...] = ()
    requested_targets: tuple[str, ...] = ()
    documentation_sources_restored: bool = False
    documentation_source_titles: tuple[str, ...] = ()
