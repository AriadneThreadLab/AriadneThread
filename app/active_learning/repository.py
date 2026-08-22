"""Candidate persistence.

Two implementations of the same contract: an in-memory store used by the
offline test suite and by dry-run CLI work, and a PostgreSQL store using the
project's existing SQLAlchemy conventions.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from app.active_learning.contracts import (
    ActiveLearningCandidate,
    CandidateStats,
    ReviewStatus,
)
from app.active_learning.export import DatasetManifest
from app.db.models import (
    ActiveLearningCandidateRow,
    ActiveLearningReviewRow,
    TrainingDatasetExportRow,
)
from app.db.session import Database

#: Bounded scan for operator statistics; the review queue is not a data
#: warehouse and an admin command should never table-scan without a limit.
STATS_SCAN_LIMIT = 10_000


def _to_row_values(candidate: ActiveLearningCandidate) -> dict[str, Any]:
    payload = candidate.model_dump(mode="json")
    payload["observed_at"] = payload.pop("created_at")
    return payload


def _from_row(row: ActiveLearningCandidateRow) -> ActiveLearningCandidate:
    return ActiveLearningCandidate.model_validate(
        {
            "candidate_id": row.candidate_id,
            "request_id": row.request_id,
            "created_at": row.observed_at,
            "user_query": row.user_query,
            "task_type": row.task_type,
            "task_signature": row.task_signature,
            "query_hash": row.query_hash,
            "analysis_plan": row.analysis_plan,
            "comparison_plan": row.comparison_plan,
            "model_output": row.model_output,
            "final_answer": row.final_answer,
            "tool_sequence": row.tool_sequence,
            "tool_validation_events": row.tool_validation_events,
            "place_resolution_events": row.place_resolution_events,
            "rag_evidence_summary": row.rag_evidence_summary,
            "metric_selection": row.metric_selection,
            "error_codes": row.error_codes,
            "warnings": row.warnings,
            "external_error_codes": row.external_error_codes,
            "selection_reasons": row.selection_reasons,
            "informativeness_score": row.informativeness_score,
            "duplicate_count": row.duplicate_count,
            "review_status": row.review_status,
            "review_notes": row.review_notes,
            "reviewer": row.reviewer,
            "reviewed_at": row.reviewed_at,
            "corrected_output": row.corrected_output,
            "approved_for_training": row.approved_for_training,
            "dataset_split": row.dataset_split,
            "model_id": row.model_id,
            "prompt_version": row.prompt_version,
            "tool_schema_version": row.tool_schema_version,
            "schema_version": row.schema_version,
            "outcome_successful": row.outcome_successful,
        }
    )


def _sort_key(candidate: ActiveLearningCandidate) -> tuple[float, str]:
    # Highest score first, then a stable tiebreak so listings and exports are
    # reproducible regardless of insertion order.
    return (-candidate.informativeness_score, candidate.candidate_id)


def compute_stats(candidates: list[ActiveLearningCandidate]) -> CandidateStats:
    """Aggregate counters shared by both repository implementations."""
    by_status: dict[str, int] = {}
    by_task: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    approved = 0
    total_score = 0.0
    for candidate in candidates:
        by_status[candidate.review_status.value] = (
            by_status.get(candidate.review_status.value, 0) + 1
        )
        by_task[candidate.task_type.value] = by_task.get(candidate.task_type.value, 0) + 1
        for reason in candidate.selection_reasons:
            by_reason[reason.value] = by_reason.get(reason.value, 0) + 1
        approved += int(candidate.approved_for_training)
        total_score += candidate.informativeness_score
    count = len(candidates)
    return CandidateStats(
        total=count,
        by_review_status=dict(sorted(by_status.items())),
        by_task_type=dict(sorted(by_task.items())),
        by_selection_reason=dict(sorted(by_reason.items())),
        approved_for_training=approved,
        mean_informativeness=round(total_score / count, 4) if count else 0.0,
    )


class InMemoryCandidateRepository:
    """Process-local store for tests and offline experimentation."""

    def __init__(self) -> None:
        self._items: dict[str, ActiveLearningCandidate] = {}

    async def add(self, candidate: ActiveLearningCandidate) -> ActiveLearningCandidate:
        self._items[candidate.candidate_id] = candidate
        return candidate

    async def get(self, candidate_id: str) -> ActiveLearningCandidate | None:
        return self._items.get(candidate_id)

    async def get_by_request(self, request_id: str) -> ActiveLearningCandidate | None:
        matches = [item for item in self._items.values() if item.request_id == request_id]
        return sorted(matches, key=_sort_key)[0] if matches else None

    async def replace(self, candidate: ActiveLearningCandidate) -> ActiveLearningCandidate:
        self._items[candidate.candidate_id] = candidate
        return candidate

    async def list_candidates(
        self,
        *,
        review_status: ReviewStatus | None = None,
        min_score: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ActiveLearningCandidate]:
        selected = [
            item
            for item in self._items.values()
            if (review_status is None or item.review_status is review_status)
            and item.informativeness_score >= min_score
        ]
        return sorted(selected, key=_sort_key)[offset : offset + limit]

    async def count_by_signature(self, task_signature: str) -> int:
        return sum(1 for item in self._items.values() if item.task_signature == task_signature)

    async def recent_by_signature(
        self, task_signature: str, *, limit: int = 20
    ) -> list[ActiveLearningCandidate]:
        matches = [item for item in self._items.values() if item.task_signature == task_signature]
        return sorted(matches, key=lambda item: item.created_at, reverse=True)[:limit]

    async def find_by_query_hash(self, query_hash: str) -> list[ActiveLearningCandidate]:
        return [item for item in self._items.values() if item.query_hash == query_hash]

    async def stats(self) -> CandidateStats:
        return compute_stats(list(self._items.values()))


class SqlAlchemyCandidateRepository:
    """PostgreSQL-backed store using the application's session factory."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def add(self, candidate: ActiveLearningCandidate) -> ActiveLearningCandidate:
        async with self._database.session() as session:
            session.add(ActiveLearningCandidateRow(**_to_row_values(candidate)))
        return candidate

    async def get(self, candidate_id: str) -> ActiveLearningCandidate | None:
        async with self._database.session() as session:
            row = await session.scalar(
                select(ActiveLearningCandidateRow).where(
                    ActiveLearningCandidateRow.candidate_id == candidate_id
                )
            )
            return None if row is None else _from_row(row)

    async def get_by_request(self, request_id: str) -> ActiveLearningCandidate | None:
        async with self._database.session() as session:
            row = await session.scalar(
                select(ActiveLearningCandidateRow)
                .where(ActiveLearningCandidateRow.request_id == request_id)
                .order_by(ActiveLearningCandidateRow.informativeness_score.desc())
                .limit(1)
            )
            return None if row is None else _from_row(row)

    async def replace(self, candidate: ActiveLearningCandidate) -> ActiveLearningCandidate:
        values = _to_row_values(candidate)
        async with self._database.session() as session:
            row = await session.scalar(
                select(ActiveLearningCandidateRow).where(
                    ActiveLearningCandidateRow.candidate_id == candidate.candidate_id
                )
            )
            if row is None:
                session.add(ActiveLearningCandidateRow(**values))
            else:
                for key, value in values.items():
                    setattr(row, key, value)
                session.add(
                    ActiveLearningReviewRow(
                        candidate_id=candidate.candidate_id,
                        status=candidate.review_status.value,
                        reviewer=candidate.reviewer,
                        notes=candidate.review_notes,
                        corrected_output=candidate.corrected_output,
                    )
                )
        return candidate

    async def list_candidates(
        self,
        *,
        review_status: ReviewStatus | None = None,
        min_score: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ActiveLearningCandidate]:
        statement = select(ActiveLearningCandidateRow).where(
            ActiveLearningCandidateRow.informativeness_score >= min_score
        )
        if review_status is not None:
            statement = statement.where(
                ActiveLearningCandidateRow.review_status == review_status.value
            )
        statement = (
            statement.order_by(
                ActiveLearningCandidateRow.informativeness_score.desc(),
                ActiveLearningCandidateRow.candidate_id.asc(),
            )
            .offset(offset)
            .limit(limit)
        )
        async with self._database.session() as session:
            rows = (await session.scalars(statement)).all()
            return [_from_row(row) for row in rows]

    async def count_by_signature(self, task_signature: str) -> int:
        async with self._database.session() as session:
            total = await session.scalar(
                select(func.count())
                .select_from(ActiveLearningCandidateRow)
                .where(ActiveLearningCandidateRow.task_signature == task_signature)
            )
            return int(total or 0)

    async def recent_by_signature(
        self, task_signature: str, *, limit: int = 20
    ) -> list[ActiveLearningCandidate]:
        async with self._database.session() as session:
            rows = (
                await session.scalars(
                    select(ActiveLearningCandidateRow)
                    .where(ActiveLearningCandidateRow.task_signature == task_signature)
                    .order_by(ActiveLearningCandidateRow.observed_at.desc())
                    .limit(limit)
                )
            ).all()
            return [_from_row(row) for row in rows]

    async def find_by_query_hash(self, query_hash: str) -> list[ActiveLearningCandidate]:
        async with self._database.session() as session:
            rows = (
                await session.scalars(
                    select(ActiveLearningCandidateRow).where(
                        ActiveLearningCandidateRow.query_hash == query_hash
                    )
                )
            ).all()
            return [_from_row(row) for row in rows]

    async def stats(self) -> CandidateStats:
        async with self._database.session() as session:
            rows = (
                await session.scalars(select(ActiveLearningCandidateRow).limit(STATS_SCAN_LIMIT))
            ).all()
            return compute_stats([_from_row(row) for row in rows])


async def record_dataset_export(database: Database, manifest: DatasetManifest) -> None:
    """Persist dataset lineage so an experiment can be traced back to its rows."""
    async with database.session() as session:
        session.add(
            TrainingDatasetExportRow(
                dataset_version=manifest.dataset_version,
                task=manifest.task.value,
                schema_version=manifest.candidate_schema_version,
                exported_at=manifest.exported_at,
                candidate_count=manifest.example_count,
                split_counts=dict(manifest.split_counts),
                candidate_ids=[entry.candidate_id for entry in manifest.lineage],
                request_ids=sorted({entry.request_id for entry in manifest.lineage}),
                source_models=list(manifest.source_models),
                content_digest=manifest.content_digest,
            )
        )
