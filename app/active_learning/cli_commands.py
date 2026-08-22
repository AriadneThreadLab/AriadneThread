"""Developer/admin commands for the active-learning review queue.

A terminal is the whole labelling platform for the first implementation: list,
inspect, approve, correct, reject, count, export. Nothing here trains a model.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.active_learning.contracts import (
    ActiveLearningCandidate,
    ExportTask,
    ReviewDecision,
    ReviewStatus,
)
from app.active_learning.export import (
    DEFAULT_SPLIT_RATIOS,
    DatasetManifest,
    export_dataset,
    validate_target_payload,
    write_export,
)
from app.active_learning.factory import build_selection_policy
from app.active_learning.repository import (
    SqlAlchemyCandidateRepository,
    record_dataset_export,
)
from app.active_learning.service import ActiveLearningService
from app.core.config import Settings
from app.core.errors import DatasetExportError, GeoAgentError
from app.db.session import Database, DatabaseConfig

#: Upper bound on candidates pulled into one export run.
EXPORT_FETCH_LIMIT = 5000


@asynccontextmanager
async def _service(settings: Settings) -> AsyncIterator[tuple[Database, ActiveLearningService]]:
    database = Database(DatabaseConfig(url=settings.database_url))
    try:
        yield (
            database,
            ActiveLearningService(
                SqlAlchemyCandidateRepository(database),
                policy=build_selection_policy(settings),
            ),
        )
    finally:
        await database.dispose()


def _summary_line(candidate: ActiveLearningCandidate) -> str:
    reasons = ",".join(reason.value for reason in candidate.selection_reasons) or "-"
    return (
        f"{candidate.candidate_id}  score={candidate.informativeness_score:.2f}  "
        f"status={candidate.review_status.value}  task={candidate.task_type.value}  "
        f"reasons={reasons}\n    query: {candidate.user_query[:110]}"
    )


async def run_list(
    settings: Settings,
    *,
    status: str | None,
    review_queue: bool,
    limit: int,
    offset: int,
) -> int:
    review_status = ReviewStatus(status) if status else None
    async with _service(settings) as (_, service):
        candidates = await service.list_candidates(
            review_status=review_status,
            review_queue_only=review_queue,
            limit=limit,
            offset=offset,
        )
    if not candidates:
        print("no active-learning candidates matched")
        return 0
    for candidate in candidates:
        print(_summary_line(candidate))
    print(f"\n{len(candidates)} candidate(s)")
    return 0


async def run_show(settings: Settings, *, candidate_id: str) -> int:
    async with _service(settings) as (_, service):
        candidate = await service.get(candidate_id)
    if candidate is None:
        print(f"unknown candidate: {candidate_id}")
        return 1
    print(json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


async def run_review(
    settings: Settings,
    *,
    candidate_id: str,
    status: ReviewStatus,
    reviewer: str | None,
    note: str | None,
    corrected_output: dict[str, object] | None = None,
) -> int:
    async with _service(settings) as (_, service):
        try:
            candidate = await service.review(
                candidate_id,
                ReviewDecision(
                    status=status,
                    reviewer=reviewer,
                    notes=note,
                    corrected_output=corrected_output,
                ),
            )
        except GeoAgentError as exc:
            print(f"error: {exc.message}")
            return 1
    print(
        f"{candidate.candidate_id} -> {candidate.review_status.value} "
        f"(approved_for_training={candidate.approved_for_training})"
    )
    return 0


async def run_correct(
    settings: Settings,
    *,
    candidate_id: str,
    corrected_output_file: Path,
    task: ExportTask,
    reviewer: str | None,
    note: str | None,
) -> int:
    try:
        payload = json.loads(corrected_output_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: corrected output could not be read: {exc}")
        return 1
    if not isinstance(payload, dict):
        print("error: corrected output must be a JSON object")
        return 1
    try:
        # Fail here rather than at export time: a hand-written correction is the
        # most likely place for a schema mistake.
        validated = validate_target_payload(payload, task)
    except DatasetExportError as exc:
        print(f"error: {exc.message}")
        return 1
    return await run_review(
        settings,
        candidate_id=candidate_id,
        status=ReviewStatus.CORRECTED,
        reviewer=reviewer,
        note=note,
        corrected_output=validated,
    )


async def run_stats(settings: Settings) -> int:
    async with _service(settings) as (_, service):
        stats = await service.stats()
    print(json.dumps(stats.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


async def run_export(
    settings: Settings,
    *,
    dataset_version: str,
    task: ExportTask,
    output_dir: Path,
    dry_run: bool,
) -> int:
    async with _service(settings) as (database, service):
        approved: list[ActiveLearningCandidate] = []
        for status in (ReviewStatus.APPROVED, ReviewStatus.CORRECTED):
            approved.extend(
                await service.list_candidates(review_status=status, limit=EXPORT_FETCH_LIMIT)
            )
        if not approved:
            print("no approved candidates to export")
            return 1

        try:
            export = export_dataset(
                approved,
                dataset_version=dataset_version,
                task=task,
                exported_at=datetime.now(tz=timezone.utc),
                ratios=DEFAULT_SPLIT_RATIOS,
                near_duplicate_threshold=settings.active_learning_novelty_threshold,
            )
        except DatasetExportError as exc:
            print(f"error: {exc.message}")
            return 1

        manifest = export.manifest
        print(
            f"dataset_version={manifest.dataset_version} task={manifest.task.value} "
            f"examples={manifest.example_count} splits={manifest.split_counts} "
            f"digest={manifest.content_digest[:12]}"
        )
        for skipped in manifest.skipped:
            print(f"  skipped {skipped.candidate_id}: {skipped.reason}")
        if dry_run:
            print("dry run: nothing written, no candidate marked as exported")
            return 0
        if manifest.example_count == 0:
            print("error: nothing valid to export")
            return 1

        written = write_export(export, output_dir)
        for name, path in sorted(written.items()):
            print(f"  wrote {name}: {path}")

        await _mark_exported(service, export.manifest, approved)
        await record_dataset_export(database, manifest)
    return 0


async def _mark_exported(
    service: ActiveLearningService,
    manifest: DatasetManifest,
    approved: Sequence[ActiveLearningCandidate],
) -> None:
    """Freeze exported candidates against their dataset split."""
    by_id = {candidate.candidate_id: candidate for candidate in approved}
    for entry in manifest.lineage:
        candidate = by_id.get(entry.candidate_id)
        if candidate is None:
            continue
        await service.mark_exported(candidate, split=entry.split)
