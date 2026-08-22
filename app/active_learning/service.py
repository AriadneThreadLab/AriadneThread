"""Active-learning selection service.

Boundary rules enforced here:

* The orchestrator is not modified to make selection decisions. It records
  operational trace events; this service reads a finished run and decides.
* Selection never blocks or alters a user response. :meth:`observe_run` catches
  everything and returns ``None`` on failure.
* Nothing in this module trains, fine-tunes, loads or promotes a model.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.active_learning.contracts import (
    ActiveLearningCandidate,
    CandidateRepository,
    CandidateStats,
    DatasetSplit,
    FeedbackSentiment,
    FeedbackSubmission,
    ReviewDecision,
    ReviewStatus,
    SelectionReason,
)
from app.active_learning.novelty import (
    NoveltyAssessment,
    build_task_signature,
    is_near_duplicate,
    query_fingerprint,
)
from app.active_learning.scoring import REASON_WEIGHTS, ActiveLearningScorer, ScoreBreakdown
from app.active_learning.signals import RunSignals, extract_run_signals
from app.agent.contracts import GeoAgentResponse
from app.core.errors import ReviewTransitionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Configurable retention thresholds."""

    min_score_to_store: float = 0.20
    min_score_for_review: float = 0.35
    near_duplicate_threshold: float = 0.85
    max_duplicates_per_signature: int = 5
    recent_run_cache: int = 256


@dataclass(frozen=True, slots=True)
class _RecentRun:
    """Just enough of a finished run to build a candidate later."""

    request_id: str
    user_query: str
    signals: RunSignals
    analysis_plan: dict[str, Any] | None
    comparison_plan: dict[str, Any] | None
    final_answer: str
    model_id: str
    prompt_version: str
    tool_schema_version: str


def _canonical_json(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _comparison_plan_from_trace(response: GeoAgentResponse) -> dict[str, Any] | None:
    for event in response.trace:
        details = event.details or {}
        plan = details.get("comparison_plan")
        if isinstance(plan, dict):
            return dict(plan)
    return None


class ActiveLearningService:
    """Evaluates finished runs and owns the human review lifecycle."""

    def __init__(
        self,
        repository: CandidateRepository,
        *,
        policy: SelectionPolicy | None = None,
        scorer: ActiveLearningScorer | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._repository = repository
        self._policy = policy or SelectionPolicy()
        self._scorer = scorer or ActiveLearningScorer()
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._recent: OrderedDict[str, _RecentRun] = OrderedDict()

    @property
    def policy(self) -> SelectionPolicy:
        return self._policy

    async def observe_run(
        self,
        *,
        request_id: str,
        user_query: str,
        response: GeoAgentResponse,
        prompt_version: str = "",
        tool_schema_version: str = "",
    ) -> ActiveLearningCandidate | None:
        """Evaluate one finished run; return a stored candidate or ``None``.

        Never raises: a selection bug must not turn a good answer into an HTTP
        error.
        """
        try:
            return await self._observe_run(
                request_id=request_id,
                user_query=user_query,
                response=response,
                prompt_version=prompt_version,
                tool_schema_version=tool_schema_version,
            )
        except Exception:
            # Strictly best-effort: a selection bug is a logged warning, never
            # a degraded answer for the user.
            logger.warning(
                "active_learning_observe_failed request_id=%s", request_id, exc_info=True
            )
            return None

    async def _observe_run(
        self,
        *,
        request_id: str,
        user_query: str,
        response: GeoAgentResponse,
        prompt_version: str,
        tool_schema_version: str,
    ) -> ActiveLearningCandidate | None:
        signals = extract_run_signals(user_query, response)
        analysis_plan = (
            response.analysis.plan.model_dump(mode="json") if response.analysis else None
        )
        recent = _RecentRun(
            request_id=request_id,
            user_query=user_query,
            signals=signals,
            analysis_plan=analysis_plan,
            comparison_plan=_comparison_plan_from_trace(response),
            final_answer=response.answer,
            model_id=response.model,
            prompt_version=prompt_version,
            tool_schema_version=tool_schema_version,
        )
        self._remember(recent)

        novelty = await self._assess_novelty(user_query, signals)
        breakdown = self._scorer.score(signals, novelty)

        if breakdown.score < self._policy.min_score_to_store:
            logger.debug(
                "active_learning_filtered request_id=%s score=%.3f external_only=%s",
                request_id,
                breakdown.score,
                breakdown.external_only,
            )
            return None
        if novelty.duplicate_count >= self._policy.max_duplicates_per_signature:
            logger.debug(
                "active_learning_signature_saturated request_id=%s signature=%s",
                request_id,
                novelty.task_signature,
            )
            return None

        candidate = self._build_candidate(recent, novelty, breakdown)
        stored = await self._repository.add(candidate)
        logger.info(
            "active_learning_candidate_stored request_id=%s candidate_id=%s score=%.3f reasons=%s",
            request_id,
            stored.candidate_id,
            stored.informativeness_score,
            [reason.value for reason in stored.selection_reasons],
        )
        return stored

    async def record_feedback(self, feedback: FeedbackSubmission) -> ActiveLearningCandidate | None:
        """Fold human feedback into an existing or newly materialised candidate."""
        extra: set[SelectionReason] = set()
        if feedback.sentiment is FeedbackSentiment.NEGATIVE:
            extra.add(SelectionReason.USER_NEGATIVE_FEEDBACK)
        if feedback.corrected_output is not None:
            extra.add(SelectionReason.HUMAN_CORRECTION_AVAILABLE)

        existing = await self._repository.get_by_request(feedback.request_id)
        if existing is not None:
            return await self._repository.replace(self._apply_feedback(existing, feedback, extra))

        recent = self._recent.get(feedback.request_id)
        if recent is None:
            logger.info(
                "active_learning_feedback_unknown_request request_id=%s", feedback.request_id
            )
            return None

        novelty = await self._assess_novelty(recent.user_query, recent.signals)
        breakdown = self._scorer.score(recent.signals, novelty, extra_reasons=frozenset(extra))
        candidate = self._build_candidate(recent, novelty, breakdown)
        candidate = self._apply_feedback(candidate, feedback, extra)
        return await self._repository.add(candidate)

    async def review(self, candidate_id: str, decision: ReviewDecision) -> ActiveLearningCandidate:
        """Apply a human review decision and its training-eligibility gate."""
        candidate = await self._repository.get(candidate_id)
        if candidate is None:
            raise ReviewTransitionError(f"unknown active-learning candidate '{candidate_id}'")
        if candidate.review_status is ReviewStatus.EXPORTED:
            raise ReviewTransitionError(
                "candidate has already been exported into a frozen dataset version"
            )

        corrected = decision.corrected_output or candidate.corrected_output
        if decision.status is ReviewStatus.CORRECTED and corrected is None:
            raise ReviewTransitionError("a corrected review requires a corrected_output payload")
        approving_failure = (
            decision.status is ReviewStatus.APPROVED
            and corrected is None
            and not candidate.outcome_successful
        )
        if approving_failure:
            raise ReviewTransitionError(
                "an unreviewed failed output cannot be approved as-is; supply a "
                "corrected_output or reject the candidate"
            )

        approved = decision.status in {ReviewStatus.APPROVED, ReviewStatus.CORRECTED}
        updated = candidate.model_copy(
            update={
                "review_status": decision.status,
                "reviewer": decision.reviewer,
                "review_notes": decision.notes,
                "reviewed_at": self._clock(),
                "corrected_output": corrected,
                "approved_for_training": approved,
            }
        )
        # Re-validate the training gate through the model, not around it.
        validated = ActiveLearningCandidate.model_validate(updated.model_dump(mode="python"))
        return await self._repository.replace(validated)

    async def mark_exported(
        self,
        candidate: ActiveLearningCandidate,
        *,
        split: DatasetSplit,
    ) -> ActiveLearningCandidate:
        """Freeze a candidate against the dataset version that consumed it."""
        if not candidate.approved_for_training:
            raise ReviewTransitionError(
                f"candidate '{candidate.candidate_id}' is not approved for training"
            )
        updated = candidate.model_copy(
            update={"review_status": ReviewStatus.EXPORTED, "dataset_split": split}
        )
        return await self._repository.replace(updated)

    async def list_candidates(
        self,
        *,
        review_status: ReviewStatus | None = None,
        review_queue_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ActiveLearningCandidate]:
        min_score = self._policy.min_score_for_review if review_queue_only else 0.0
        return await self._repository.list_candidates(
            review_status=review_status,
            min_score=min_score,
            limit=limit,
            offset=offset,
        )

    async def get(self, candidate_id: str) -> ActiveLearningCandidate | None:
        return await self._repository.get(candidate_id)

    async def stats(self) -> CandidateStats:
        return await self._repository.stats()

    def _remember(self, recent: _RecentRun) -> None:
        if self._policy.recent_run_cache <= 0:
            return
        self._recent[recent.request_id] = recent
        self._recent.move_to_end(recent.request_id)
        while len(self._recent) > self._policy.recent_run_cache:
            self._recent.popitem(last=False)

    async def _assess_novelty(self, user_query: str, signals: RunSignals) -> NoveltyAssessment:
        metric = signals.metric_selection.selected_metric if signals.metric_selection else None
        signature = build_task_signature(
            task_type=signals.task_type,
            feature_concept=signals.feature_concept,
            scope_kind=signals.scope_kind,
            metric=metric,
        )
        fingerprint = query_fingerprint(user_query)

        exact = await self._repository.find_by_query_hash(fingerprint)
        signature_count = await self._repository.count_by_signature(signature)
        neighbours = await self._repository.recent_by_signature(signature, limit=20)

        nearest = 0.0
        near_duplicates = 0
        for neighbour in neighbours:
            if is_near_duplicate(
                user_query,
                neighbour.user_query,
                threshold=self._policy.near_duplicate_threshold,
            ):
                near_duplicates += 1
                nearest = 1.0
        duplicate_count = max(len(exact), near_duplicates, min(signature_count, len(neighbours)))
        return NoveltyAssessment(
            query_hash=fingerprint,
            task_signature=signature,
            duplicate_count=duplicate_count,
            nearest_similarity=nearest,
            is_novel=duplicate_count == 0,
        )

    def _build_candidate(
        self,
        recent: _RecentRun,
        novelty: NoveltyAssessment,
        breakdown: ScoreBreakdown,
    ) -> ActiveLearningCandidate:
        signals = recent.signals
        return ActiveLearningCandidate(
            candidate_id=self._id_factory(),
            request_id=recent.request_id,
            created_at=self._clock(),
            user_query=recent.user_query,
            task_type=signals.task_type,
            task_signature=novelty.task_signature,
            query_hash=novelty.query_hash,
            analysis_plan=recent.analysis_plan,
            comparison_plan=recent.comparison_plan,
            model_output=_canonical_json(recent.analysis_plan or recent.comparison_plan),
            final_answer=recent.final_answer or None,
            tool_sequence=list(signals.tool_sequence),
            tool_validation_events=list(signals.tool_validation_events),
            place_resolution_events=list(signals.place_resolution_events),
            rag_evidence_summary=signals.rag_evidence,
            metric_selection=signals.metric_selection,
            error_codes=list(signals.model_error_codes),
            warnings=list(signals.warnings),
            external_error_codes=list(signals.external_error_codes),
            selection_reasons=list(breakdown.reasons),
            informativeness_score=breakdown.score,
            duplicate_count=novelty.duplicate_count,
            outcome_successful=signals.outcome_successful,
            model_id=recent.model_id,
            prompt_version=recent.prompt_version,
            tool_schema_version=recent.tool_schema_version,
        )

    def _apply_feedback(
        self,
        candidate: ActiveLearningCandidate,
        feedback: FeedbackSubmission,
        extra: set[SelectionReason],
    ) -> ActiveLearningCandidate:
        reasons = list(dict.fromkeys([*candidate.selection_reasons, *sorted(extra, key=str)]))
        notes = feedback.note or candidate.review_notes
        if feedback.failure_category:
            prefix = f"[{feedback.failure_category}]"
            notes = f"{prefix} {notes}" if notes else prefix
        rescored = min(
            1.0,
            candidate.informativeness_score
            + sum(REASON_WEIGHTS.get(reason, 0.0) for reason in extra),
        )
        update: dict[str, Any] = {
            "selection_reasons": reasons,
            "review_notes": notes,
            "corrected_output": feedback.corrected_output or candidate.corrected_output,
            "informativeness_score": round(rescored, 4),
        }
        if candidate.review_status is not ReviewStatus.EXPORTED:
            # Feedback reopens review; it never approves anything by itself. An
            # already-exported candidate keeps its status so the frozen dataset
            # version stays reproducible.
            update["review_status"] = ReviewStatus.PENDING
            update["approved_for_training"] = False
        return candidate.model_copy(update=update)
