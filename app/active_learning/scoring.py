"""Deterministic informativeness scoring.

The scorer never asks a language model whether a sample is interesting, and it
never reads a model's self-reported confidence: local 7B confidence is not
calibrated, so it would mostly add noise. Every input is an observed, bounded
fact about what the backend accepted or rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.active_learning.contracts import SelectionReason
from app.active_learning.novelty import NoveltyAssessment
from app.active_learning.signals import RunSignals

#: Contribution of each reason before the novelty adjustment. Values are tuned
#: so that a single strong guardrail block clears the review threshold on its
#: own, while an ordinary success needs novelty to qualify.
REASON_WEIGHTS: dict[SelectionReason, float] = {
    SelectionReason.HUMAN_CORRECTION_AVAILABLE: 0.60,
    SelectionReason.USER_NEGATIVE_FEEDBACK: 0.50,
    SelectionReason.HALLUCINATED_COORDINATE_BLOCKED: 0.45,
    SelectionReason.HALLUCINATED_TAG_BLOCKED: 0.40,
    SelectionReason.INVALID_STRUCTURED_OUTPUT: 0.40,
    SelectionReason.DEPENDENCY_ORDER_VIOLATION: 0.35,
    SelectionReason.GROUNDING_CONFLICT: 0.35,
    SelectionReason.LLM_PROTOCOL_ERROR: 0.35,
    SelectionReason.PLACE_RESOLUTION_AMBIGUOUS: 0.35,
    SelectionReason.METRIC_FEASIBILITY_FAILED: 0.30,
    SelectionReason.TOOL_VALIDATION_FAILED: 0.30,
    SelectionReason.WRONG_OR_INELIGIBLE_TOOL: 0.30,
    SelectionReason.GUARDRAIL_TRIGGERED: 0.25,
    SelectionReason.MODEL_DISAGREEMENT: 0.25,
    SelectionReason.PLACE_RESOLUTION_FAILED: 0.25,
    SelectionReason.SUCCESSFUL_HIGH_VALUE_TRACE: 0.25,
    SelectionReason.METRIC_SELECTION_UNCERTAIN: 0.20,
    SelectionReason.NOVEL_QUERY: 0.20,
    # A local model timeout is mostly a hardware/latency fact. It is worth
    # noticing, but it is weak evidence that the weights need to change.
    SelectionReason.MODEL_TIMEOUT: 0.15,
}

#: Score given to a run whose only failure came from an external dependency.
EXTERNAL_ONLY_SCORE = 0.05


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Explainable scoring result (persisted as score + reasons)."""

    score: float
    base_score: float
    novelty_factor: float
    reasons: tuple[SelectionReason, ...]
    external_only: bool


class ActiveLearningScorer:
    """Assigns a bounded ``0.0-1.0`` informativeness score."""

    def __init__(
        self,
        *,
        duplicate_decay: float = 0.25,
        min_novelty_factor: float = 0.35,
        external_only_score: float = EXTERNAL_ONLY_SCORE,
    ) -> None:
        self._duplicate_decay = duplicate_decay
        self._min_novelty_factor = min_novelty_factor
        self._external_only_score = external_only_score

    def score(
        self,
        signals: RunSignals,
        novelty: NoveltyAssessment,
        *,
        extra_reasons: frozenset[SelectionReason] = frozenset(),
    ) -> ScoreBreakdown:
        """Score one run from its signals and how novel it looks."""
        reasons = set(signals.reasons) | set(extra_reasons)
        has_model_signal = bool(reasons - {SelectionReason.NOVEL_QUERY})

        if signals.external_error_codes and not has_model_signal:
            # An Overpass 504 is an infrastructure event. Retaining it as a
            # training example would teach the model nothing.
            return ScoreBreakdown(
                score=self._external_only_score,
                base_score=self._external_only_score,
                novelty_factor=1.0,
                reasons=tuple(sorted(reasons, key=lambda item: item.value)),
                external_only=True,
            )

        if novelty.is_novel:
            reasons.add(SelectionReason.NOVEL_QUERY)

        base = min(1.0, sum(REASON_WEIGHTS.get(reason, 0.0) for reason in reasons))
        factor = self._novelty_factor(novelty.duplicate_count)
        return ScoreBreakdown(
            score=round(min(1.0, max(0.0, base * factor)), 4),
            base_score=round(base, 4),
            novelty_factor=round(factor, 4),
            reasons=tuple(sorted(reasons, key=lambda item: item.value)),
            external_only=False,
        )

    def _novelty_factor(self, duplicate_count: int) -> float:
        if duplicate_count <= 0:
            return 1.0
        decayed = 1.0 - self._duplicate_decay * duplicate_count
        return max(self._min_novelty_factor, decayed)
