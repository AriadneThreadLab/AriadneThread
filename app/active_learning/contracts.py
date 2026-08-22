"""Active learning domain contracts.

This package selects informative interactions for **human review**. It never
trains a model, never mutates weights and never promotes a model: training runs
only in the separate ``finetuning/`` pipeline, which consumes exported dataset
files and never reaches back into this database.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.active_learning.sanitization import sanitize_payload, sanitize_text

#: Bumped when the meaning of stored candidate fields changes.
CANDIDATE_SCHEMA_VERSION = "active-learning-candidate-1"


class SelectionReason(str, Enum):
    """Why a completed run is worth a human look.

    Bounded codes only: scoring, review filters and dataset lineage all key off
    these values, so free-text reasons are not accepted anywhere.
    """

    TOOL_VALIDATION_FAILED = "TOOL_VALIDATION_FAILED"
    INVALID_STRUCTURED_OUTPUT = "INVALID_STRUCTURED_OUTPUT"
    WRONG_OR_INELIGIBLE_TOOL = "WRONG_OR_INELIGIBLE_TOOL"
    PLACE_RESOLUTION_AMBIGUOUS = "PLACE_RESOLUTION_AMBIGUOUS"
    PLACE_RESOLUTION_FAILED = "PLACE_RESOLUTION_FAILED"
    GROUNDING_CONFLICT = "GROUNDING_CONFLICT"
    METRIC_SELECTION_UNCERTAIN = "METRIC_SELECTION_UNCERTAIN"
    METRIC_FEASIBILITY_FAILED = "METRIC_FEASIBILITY_FAILED"
    GUARDRAIL_TRIGGERED = "GUARDRAIL_TRIGGERED"
    HALLUCINATED_COORDINATE_BLOCKED = "HALLUCINATED_COORDINATE_BLOCKED"
    HALLUCINATED_TAG_BLOCKED = "HALLUCINATED_TAG_BLOCKED"
    DEPENDENCY_ORDER_VIOLATION = "DEPENDENCY_ORDER_VIOLATION"
    LLM_PROTOCOL_ERROR = "LLM_PROTOCOL_ERROR"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    USER_NEGATIVE_FEEDBACK = "USER_NEGATIVE_FEEDBACK"
    HUMAN_CORRECTION_AVAILABLE = "HUMAN_CORRECTION_AVAILABLE"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    NOVEL_QUERY = "NOVEL_QUERY"
    SUCCESSFUL_HIGH_VALUE_TRACE = "SUCCESSFUL_HIGH_VALUE_TRACE"


class ReviewStatus(str, Enum):
    """Bounded human-review lifecycle."""

    PENDING = "pending"
    APPROVED = "approved"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    EXPORTED = "exported"


class DatasetSplit(str, Enum):
    """Dataset partition assigned at export time."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class TaskType(str, Enum):
    """Coarse shape of the user request, used for diversity accounting."""

    SPATIAL_SEARCH = "spatial_search"
    COMPARISON = "comparison"
    KNOWLEDGE_ONLY = "knowledge_only"
    UNKNOWN = "unknown"


class FeedbackSentiment(str, Enum):
    """Direction of end-user feedback on a completed run."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class ExportTask(str, Enum):
    """Supervised fine-tuning target built from approved candidates.

    Both tasks reuse an authoritative Ariadne schema; no parallel training
    schema is defined anywhere in this package.
    """

    #: ``AnalysisPlan`` — the validated argument model of ``analyze_features``.
    ANALYSIS_PLAN = "analysis_plan"
    #: ``MultiTargetComparisonPlan`` — the compact comparison planner schema.
    COMPARISON_PLAN = "comparison_plan"


class ErrorAttribution(str, Enum):
    """Whether a failure blames the model or an external dependency."""

    MODEL = "model"
    EXTERNAL = "external"
    NONE = "none"


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class ToolValidationEvent(BaseModel):
    """One registry-level accept/reject decision for a model tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(min_length=1, max_length=64)
    status: str = Field(pattern=r"^(passed|failed)$")
    error_code: str | None = Field(default=None, max_length=64)
    round_index: int = Field(default=0, ge=0)
    detail: str | None = Field(default=None, max_length=400)

    @field_validator("detail")
    @classmethod
    def _clean_detail(cls, value: str | None) -> str | None:
        return None if value is None else sanitize_text(value, field="detail")


class PlaceResolutionEvent(BaseModel):
    """One trusted place-resolution attempt, without raw geocoder payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1, max_length=160)
    status: str = Field(pattern=r"^(resolved|ambiguous|failed)$")
    source: str | None = Field(default=None, max_length=64)
    error_code: str | None = Field(default=None, max_length=64)

    @field_validator("label")
    @classmethod
    def _clean_label(cls, value: str) -> str:
        return sanitize_text(value, field="label")


class RagEvidenceSummary(BaseModel):
    """Compact record of documentation grounding (never passage bodies)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passage_count: int = Field(default=0, ge=0)
    documented_tags: list[str] = Field(default_factory=list, max_length=16)
    document_titles: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("documented_tags", "document_titles")
    @classmethod
    def _clean_items(cls, value: list[str]) -> list[str]:
        return [sanitize_text(item, field="rag_evidence_summary")[:200] for item in value]


class MetricSelectionSummary(BaseModel):
    """What the analytics layer decided, and why it was allowed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_metric: str | None = Field(default=None, max_length=64)
    metric_rule_id: str | None = Field(default=None, max_length=64)
    inferred_goal: str | None = Field(default=None, max_length=64)
    rejection_reasons: list[str] = Field(default_factory=list, max_length=8)
    analysis_status: str | None = Field(default=None, max_length=32)


class ActiveLearningCandidate(BaseModel):
    """A completed run retained as a *potential* training example.

    Retention is not approval. Only an explicit human decision may set
    ``approved_for_training``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=8, max_length=64)
    request_id: str = Field(min_length=1, max_length=64)
    created_at: datetime = Field(default_factory=_now)

    user_query: str = Field(min_length=1, max_length=2000)
    task_type: TaskType = TaskType.UNKNOWN
    task_signature: str = Field(default="", max_length=200)
    query_hash: str = Field(default="", max_length=64)

    analysis_plan: dict[str, Any] | None = None
    comparison_plan: dict[str, Any] | None = None
    model_output: str | None = Field(default=None, max_length=8000)
    final_answer: str | None = Field(default=None, max_length=8000)

    tool_sequence: list[str] = Field(default_factory=list, max_length=32)
    tool_validation_events: list[ToolValidationEvent] = Field(default_factory=list, max_length=32)
    place_resolution_events: list[PlaceResolutionEvent] = Field(default_factory=list, max_length=8)
    rag_evidence_summary: RagEvidenceSummary | None = None
    metric_selection: MetricSelectionSummary | None = None

    error_codes: list[str] = Field(default_factory=list, max_length=16)
    warnings: list[str] = Field(default_factory=list, max_length=16)
    external_error_codes: list[str] = Field(default_factory=list, max_length=16)

    selection_reasons: list[SelectionReason] = Field(default_factory=list, max_length=24)
    informativeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    duplicate_count: int = Field(default=0, ge=0)

    review_status: ReviewStatus = ReviewStatus.PENDING
    review_notes: str | None = Field(default=None, max_length=2000)
    reviewer: str | None = Field(default=None, max_length=64)
    reviewed_at: datetime | None = None
    corrected_output: dict[str, Any] | None = None
    approved_for_training: bool = False
    dataset_split: DatasetSplit | None = None

    model_id: str = Field(default="", max_length=128)
    prompt_version: str = Field(default="", max_length=64)
    tool_schema_version: str = Field(default="", max_length=64)
    schema_version: str = CANDIDATE_SCHEMA_VERSION

    #: True when the model produced a validated, usable result for the task.
    outcome_successful: bool = False

    @field_validator("user_query", "final_answer", "model_output", "review_notes")
    @classmethod
    def _clean_text(cls, value: str | None) -> str | None:
        return None if value is None else sanitize_text(value, field="candidate text")

    @field_validator("analysis_plan", "comparison_plan", "corrected_output")
    @classmethod
    def _clean_payload(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        cleaned = sanitize_payload(value, field="candidate payload")
        assert isinstance(cleaned, dict)
        return cleaned

    @field_validator("error_codes", "warnings", "external_error_codes")
    @classmethod
    def _clean_codes(cls, value: list[str]) -> list[str]:
        return [sanitize_text(item, field="code")[:300] for item in value]

    @model_validator(mode="after")
    def _validate_training_gate(self) -> ActiveLearningCandidate:
        if not self.approved_for_training:
            return self
        if self.review_status not in {
            ReviewStatus.APPROVED,
            ReviewStatus.CORRECTED,
            ReviewStatus.EXPORTED,
        }:
            raise ValueError("approved_for_training requires an approved or corrected review")
        if self.corrected_output is None and not self.outcome_successful:
            raise ValueError(
                "approved_for_training requires a validated successful output "
                "or a human-corrected target output"
            )
        return self

    @property
    def training_target(self) -> dict[str, Any] | None:
        """Human correction wins over the model's own output."""
        if self.corrected_output is not None:
            return self.corrected_output
        return self.analysis_plan


class ReviewDecision(BaseModel):
    """A human decision applied to one candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ReviewStatus
    reviewer: str | None = Field(default=None, max_length=64)
    notes: str | None = Field(default=None, max_length=2000)
    corrected_output: dict[str, Any] | None = None

    @field_validator("status")
    @classmethod
    def _reject_terminal_export(cls, value: ReviewStatus) -> ReviewStatus:
        if value is ReviewStatus.EXPORTED:
            raise ValueError("'exported' is set by the exporter, not by a reviewer")
        return value


class FeedbackSubmission(BaseModel):
    """End-user or reviewer feedback about a completed request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1, max_length=64)
    sentiment: FeedbackSentiment
    failure_category: str | None = Field(default=None, max_length=64)
    corrected_output: dict[str, Any] | None = None
    note: str | None = Field(default=None, max_length=2000)

    @field_validator("note")
    @classmethod
    def _clean_note(cls, value: str | None) -> str | None:
        return None if value is None else sanitize_text(value, field="note")


class CandidateStats(BaseModel):
    """Aggregate review-queue counters for operators."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total: int = Field(default=0, ge=0)
    by_review_status: dict[str, int] = Field(default_factory=dict)
    by_task_type: dict[str, int] = Field(default_factory=dict)
    by_selection_reason: dict[str, int] = Field(default_factory=dict)
    approved_for_training: int = Field(default=0, ge=0)
    mean_informativeness: float = Field(default=0.0, ge=0.0, le=1.0)


@runtime_checkable
class CandidateRepository(Protocol):
    """Persistence boundary for active-learning candidates."""

    async def add(self, candidate: ActiveLearningCandidate) -> ActiveLearningCandidate: ...

    async def get(self, candidate_id: str) -> ActiveLearningCandidate | None: ...

    async def get_by_request(self, request_id: str) -> ActiveLearningCandidate | None: ...

    async def replace(self, candidate: ActiveLearningCandidate) -> ActiveLearningCandidate: ...

    async def list_candidates(
        self,
        *,
        review_status: ReviewStatus | None = None,
        min_score: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ActiveLearningCandidate]: ...

    async def count_by_signature(self, task_signature: str) -> int: ...

    async def recent_by_signature(
        self, task_signature: str, *, limit: int = 20
    ) -> list[ActiveLearningCandidate]: ...

    async def find_by_query_hash(self, query_hash: str) -> list[ActiveLearningCandidate]: ...

    async def stats(self) -> CandidateStats: ...
