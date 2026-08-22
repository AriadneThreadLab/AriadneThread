"""Execution Memory persistence.

In-memory store for offline tests; PostgreSQL store for the running service.
Insert-only: historical executions are never updated in place.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol
from uuid import UUID

from sqlalchemy import select

from app.db.models import ExecutionMemoryDatasetRow, ExecutionMemoryRow
from app.db.session import Database
from app.execution_memory.contracts import ExecutionSnapshot, strip_hidden_reasoning


class ExecutionMemoryRepository(Protocol):
    async def save(self, snapshot: ExecutionSnapshot) -> ExecutionSnapshot: ...

    async def get(self, execution_id: UUID) -> ExecutionSnapshot | None: ...

    async def latest_for_conversation(self, conversation_id: str) -> ExecutionSnapshot | None: ...

    async def history_for_conversation(
        self,
        conversation_id: str,
        *,
        limit: int,
    ) -> list[ExecutionSnapshot]: ...

    async def enforce_retention(
        self,
        conversation_id: str,
        *,
        max_per_conversation: int,
        max_age_seconds: int,
    ) -> None: ...


class InMemoryExecutionMemoryRepository:
    """Process-local store used by the offline test suite."""

    def __init__(self) -> None:
        self._items: dict[UUID, ExecutionSnapshot] = {}

    async def save(self, snapshot: ExecutionSnapshot) -> ExecutionSnapshot:
        if snapshot.execution_id in self._items:
            raise ValueError("execution snapshots are immutable; create a new execution_id")
        self._items[snapshot.execution_id] = snapshot
        return snapshot

    async def get(self, execution_id: UUID) -> ExecutionSnapshot | None:
        return self._items.get(execution_id)

    async def latest_for_conversation(self, conversation_id: str) -> ExecutionSnapshot | None:
        matches = [item for item in self._items.values() if item.conversation_id == conversation_id]
        if not matches:
            return None
        return max(matches, key=lambda item: item.created_at)

    async def history_for_conversation(
        self,
        conversation_id: str,
        *,
        limit: int,
    ) -> list[ExecutionSnapshot]:
        matches = [item for item in self._items.values() if item.conversation_id == conversation_id]
        matches.sort(key=lambda item: item.created_at, reverse=True)
        return matches[: max(limit, 0)]

    async def enforce_retention(
        self,
        conversation_id: str,
        *,
        max_per_conversation: int,
        max_age_seconds: int,
    ) -> None:
        matches = sorted(
            [item for item in self._items.values() if item.conversation_id == conversation_id],
            key=lambda item: item.created_at,
        )
        if not matches:
            return
        latest_id = matches[-1].execution_id
        cutoff = datetime.now(tz=timezone.utc) - timedelta(seconds=max_age_seconds)
        kept = [
            item for item in matches if item.execution_id == latest_id or item.created_at >= cutoff
        ]
        if len(kept) > max_per_conversation:
            kept = kept[-max_per_conversation:]
            if matches[-1] not in kept:
                kept.append(matches[-1])
        keep_ids = {item.execution_id for item in kept}
        for item in matches:
            if item.execution_id not in keep_ids:
                self._items.pop(item.execution_id, None)


class PostgresExecutionMemoryRepository:
    """Insert-only PostgreSQL persistence."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def save(self, snapshot: ExecutionSnapshot) -> ExecutionSnapshot:
        payload = strip_hidden_reasoning(snapshot.model_dump(mode="json"))
        if not isinstance(payload, dict):
            raise ValueError("execution snapshot payload is invalid")
        payload.pop("datasets", None)
        async with self._database.session() as session:
            row = ExecutionMemoryRow(
                id=snapshot.execution_id,
                conversation_id=snapshot.conversation_id,
                request_id=snapshot.request_id,
                parent_execution_id=snapshot.parent_execution_id,
                original_user_query=snapshot.original_user_query,
                analysis_type=snapshot.analysis_type,
                feature_concept=snapshot.feature_concept,
                outcome=snapshot.outcome,
                snapshot=payload,
            )
            session.add(row)
            for dataset in snapshot.datasets:
                session.add(
                    ExecutionMemoryDatasetRow(
                        execution_id=snapshot.execution_id,
                        execution_dataset_id=dataset.execution_dataset_id,
                        target_stable_id=dataset.target_stable_id,
                        retrieved_at=dataset.retrieved_at,
                        truncated=dataset.truncated,
                        feature_count=dataset.feature_count,
                        provenance=strip_hidden_reasoning(
                            {
                                "source": dataset.source,
                                "endpoint": dataset.endpoint,
                                "attribution": dataset.attribution,
                                "effective_limit": dataset.effective_limit,
                                "resolved_tags": list(dataset.resolved_tags),
                                "scope": dataset.scope.model_dump(mode="json"),
                                "query_spec": dataset.query_spec,
                                "request_scoped_dataset_ref_at_creation": (
                                    dataset.request_scoped_dataset_ref_at_creation
                                ),
                            }
                        ),
                        feature_collection=dataset.feature_collection,
                    )
                )
        return snapshot

    async def get(self, execution_id: UUID) -> ExecutionSnapshot | None:
        async with self._database.session() as session:
            row = await session.get(ExecutionMemoryRow, execution_id)
            if row is None:
                return None
            return _from_row(row)

    async def latest_for_conversation(self, conversation_id: str) -> ExecutionSnapshot | None:
        async with self._database.session() as session:
            result = await session.execute(
                select(ExecutionMemoryRow)
                .where(ExecutionMemoryRow.conversation_id == conversation_id)
                .order_by(ExecutionMemoryRow.created_at.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _from_row(row)

    async def history_for_conversation(
        self,
        conversation_id: str,
        *,
        limit: int,
    ) -> list[ExecutionSnapshot]:
        async with self._database.session() as session:
            result = await session.execute(
                select(ExecutionMemoryRow)
                .where(ExecutionMemoryRow.conversation_id == conversation_id)
                .order_by(ExecutionMemoryRow.created_at.desc())
                .limit(max(limit, 0))
            )
            return [_from_row(row) for row in result.scalars().all()]

    async def enforce_retention(
        self,
        conversation_id: str,
        *,
        max_per_conversation: int,
        max_age_seconds: int,
    ) -> None:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(seconds=max_age_seconds)
        async with self._database.session() as session:
            result = await session.execute(
                select(ExecutionMemoryRow)
                .where(ExecutionMemoryRow.conversation_id == conversation_id)
                .order_by(ExecutionMemoryRow.created_at.asc())
            )
            rows = list(result.scalars().all())
            if not rows:
                return
            latest_id = rows[-1].id
            keep: list[ExecutionMemoryRow] = [
                row for row in rows if row.id == latest_id or row.created_at >= cutoff
            ]
            if len(keep) > max_per_conversation:
                keep = keep[-max_per_conversation:]
                if rows[-1] not in keep:
                    keep.append(rows[-1])
            keep_ids = {row.id for row in keep}
            for row in rows:
                if row.id not in keep_ids:
                    await session.delete(row)


def _from_row(row: ExecutionMemoryRow) -> ExecutionSnapshot:
    payload = dict(row.snapshot or {})
    payload["execution_id"] = str(row.id)
    payload["conversation_id"] = row.conversation_id
    payload["request_id"] = row.request_id
    payload["parent_execution_id"] = (
        str(row.parent_execution_id) if row.parent_execution_id else None
    )
    payload["original_user_query"] = row.original_user_query
    payload["analysis_type"] = row.analysis_type
    payload["feature_concept"] = row.feature_concept
    payload["outcome"] = row.outcome
    payload["created_at"] = row.created_at
    datasets = []
    for item in row.datasets:
        prov = item.provenance or {}
        datasets.append(
            {
                "execution_dataset_id": str(item.execution_dataset_id),
                "target_stable_id": item.target_stable_id,
                "retrieved_at": item.retrieved_at,
                "source": prov.get("source", "live_osm"),
                "endpoint": prov.get("endpoint", ""),
                "attribution": prov.get("attribution", ""),
                "effective_limit": prov.get("effective_limit", 50),
                "truncated": item.truncated,
                "feature_count": item.feature_count,
                "resolved_tags": tuple(prov.get("resolved_tags") or ()),
                "scope": prov.get("scope"),
                "query_spec": prov.get("query_spec") or {},
                "feature_collection": item.feature_collection,
                "request_scoped_dataset_ref_at_creation": prov.get(
                    "request_scoped_dataset_ref_at_creation"
                ),
            }
        )
    payload["datasets"] = datasets
    return ExecutionSnapshot.model_validate(payload)
