"""Composition helpers for the active-learning subsystem."""

from __future__ import annotations

from app.active_learning.contracts import CandidateRepository
from app.active_learning.repository import SqlAlchemyCandidateRepository
from app.active_learning.service import ActiveLearningService, SelectionPolicy
from app.core.config import Settings
from app.db.session import Database


def build_selection_policy(settings: Settings) -> SelectionPolicy:
    return SelectionPolicy(
        min_score_to_store=settings.active_learning_min_score_to_store,
        min_score_for_review=settings.active_learning_min_score_for_review,
        near_duplicate_threshold=settings.active_learning_novelty_threshold,
        max_duplicates_per_signature=settings.active_learning_max_duplicates_per_signature,
        recent_run_cache=settings.active_learning_recent_run_cache,
    )


def build_active_learning_service(
    settings: Settings,
    database: Database,
    *,
    repository: CandidateRepository | None = None,
) -> ActiveLearningService | None:
    """Build the service, or ``None`` when selection is disabled.

    Opens no connection: the repository borrows the application's existing
    session factory and only touches PostgreSQL when a run is evaluated.
    """
    if not settings.active_learning_enabled:
        return None
    return ActiveLearningService(
        repository or SqlAlchemyCandidateRepository(database),
        policy=build_selection_policy(settings),
    )
