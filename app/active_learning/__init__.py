"""Active learning: choosing which interactions deserve human review.

This package *selects and curates*. It does not train. The only thing it hands
to the fine-tuning pipeline is a versioned dataset file produced by
:mod:`app.active_learning.export`.
"""

from __future__ import annotations

from app.active_learning.contracts import (
    ActiveLearningCandidate,
    CandidateRepository,
    CandidateStats,
    DatasetSplit,
    ExportTask,
    FeedbackSentiment,
    FeedbackSubmission,
    ReviewDecision,
    ReviewStatus,
    SelectionReason,
    TaskType,
)
from app.active_learning.scoring import ActiveLearningScorer
from app.active_learning.service import ActiveLearningService, SelectionPolicy

__all__ = [
    "ActiveLearningCandidate",
    "ActiveLearningScorer",
    "ActiveLearningService",
    "CandidateRepository",
    "CandidateStats",
    "DatasetSplit",
    "ExportTask",
    "FeedbackSentiment",
    "FeedbackSubmission",
    "ReviewDecision",
    "ReviewStatus",
    "SelectionPolicy",
    "SelectionReason",
    "TaskType",
]
