"""Curated dataset export.

The only channel between Ariadne and the fine-tuning pipeline is a versioned
file produced here. Training code reads those files; it never reaches into the
active-learning tables.

Two properties matter most:

* **Authoritative schema.** Assistant targets are validated against the real
  ``AnalysisPlan`` / ``MultiTargetComparisonPlan`` models before they are
  written, and the generated JSON Schema travels with the dataset.
* **Determinism.** For a fixed dataset version and candidate set, the bytes of
  every split file are identical on every run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.active_learning.contracts import (
    ActiveLearningCandidate,
    DatasetSplit,
    ExportTask,
    ReviewStatus,
)
from app.active_learning.novelty import is_near_duplicate
from app.agent.comparison_workflow import MultiTargetComparisonPlan
from app.analytics.contracts import AnalysisPlan
from app.core.errors import DatasetExportError

#: Conversational JSONL layout understood by TRL / Hugging Face chat templates.
SFT_FORMAT_VERSION = "sft-conversational-1"

#: Default train / validation / test proportions, applied to *groups*.
DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)

_ANALYSIS_PLAN_INSTRUCTION = (
    "Return one JSON object matching the Ariadne AnalysisPlan schema. "
    "Use only the dataset references listed above."
)
_COMPARISON_PLAN_INSTRUCTION = (
    "Return one JSON object matching the Ariadne MultiTargetComparisonPlan schema. "
    "Preserve the landmark names from the request and do not invent coordinates."
)


class ChatMessage(BaseModel):
    """One conversational turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(pattern=r"^(user|assistant)$")
    content: str = Field(min_length=1)


class TrainingExample(BaseModel):
    """One SFT sample. Metadata lives in the manifest, never in the target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: list[ChatMessage] = Field(min_length=2, max_length=2)


class LineageEntry(BaseModel):
    """Provenance of one exported sample."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    request_id: str
    review_status: ReviewStatus
    split: DatasetSplit
    group_key: str
    model_id: str = ""
    prompt_version: str = ""
    tool_schema_version: str = ""
    corrected: bool = False


class SkippedCandidate(BaseModel):
    """A candidate that could not be turned into a valid sample."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    reason: str


class DatasetManifest(BaseModel):
    """Everything needed to reproduce and audit one dataset version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: str
    task: ExportTask
    sft_format_version: str = SFT_FORMAT_VERSION
    candidate_schema_version: str
    target_schema_name: str
    exported_at: datetime
    example_count: int = Field(ge=0)
    split_counts: dict[str, int] = Field(default_factory=dict)
    split_candidate_ids: dict[str, list[str]] = Field(default_factory=dict)
    lineage: list[LineageEntry] = Field(default_factory=list)
    source_models: list[str] = Field(default_factory=list)
    prompt_versions: list[str] = Field(default_factory=list)
    reviewer_statuses: dict[str, int] = Field(default_factory=dict)
    skipped: list[SkippedCandidate] = Field(default_factory=list)
    content_digest: str = ""


class DatasetExport(BaseModel):
    """In-memory export result; writing to disk is a separate step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: DatasetManifest
    splits: dict[str, list[TrainingExample]]
    target_json_schema: dict[str, Any]

    def jsonl(self, split: DatasetSplit) -> str:
        """Deterministic JSONL text for one split."""
        return "".join(
            json.dumps(example.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
            for example in self.splits.get(split.value, [])
        )


def canonical_json(payload: dict[str, Any]) -> str:
    """Stable JSON rendering used for every assistant target."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def is_exportable(candidate: ActiveLearningCandidate) -> bool:
    """Only reviewed, approved candidates with a target may be exported."""
    if not candidate.approved_for_training:
        return False
    if candidate.review_status not in {ReviewStatus.APPROVED, ReviewStatus.CORRECTED}:
        return False
    return candidate.corrected_output is not None or candidate.outcome_successful


def _target_payload(candidate: ActiveLearningCandidate, task: ExportTask) -> dict[str, Any] | None:
    # A human correction always wins over the model's original output.
    if candidate.corrected_output is not None:
        return candidate.corrected_output
    if task is ExportTask.ANALYSIS_PLAN:
        return candidate.analysis_plan
    return candidate.comparison_plan


def _validated_target(payload: dict[str, Any], task: ExportTask) -> dict[str, Any]:
    model: type[BaseModel] = (
        AnalysisPlan if task is ExportTask.ANALYSIS_PLAN else MultiTargetComparisonPlan
    )
    return model.model_validate(payload).model_dump(mode="json")


def validate_target_payload(payload: dict[str, Any], task: ExportTask) -> dict[str, Any]:
    """Validate a (possibly hand-written) target against the authoritative model."""
    try:
        return _validated_target(payload, task)
    except ValidationError as exc:
        raise DatasetExportError(
            f"corrected output is not a valid {task.value}: {exc.error_count()} validation error(s)"
        ) from exc


def _user_content(
    candidate: ActiveLearningCandidate,
    task: ExportTask,
    target: dict[str, Any],
) -> str:
    if task is ExportTask.COMPARISON_PLAN:
        return f"{candidate.user_query}\n\n{_COMPARISON_PLAN_INSTRUCTION}"
    # AnalysisPlan binds targets to backend-generated dataset references, so the
    # prompt must state which references exist. They are rendered from the
    # validated plan itself, never invented at export time.
    targets = target.get("targets", [])
    lines = sorted(
        f"- {item['dataset_ref']}: {item['label']}"
        for item in targets
        if isinstance(item, dict) and "dataset_ref" in item and "label" in item
    )
    catalogue = "\n".join(lines)
    return (
        f"{candidate.user_query}\n\nAvailable datasets:\n{catalogue}\n\n"
        f"{_ANALYSIS_PLAN_INSTRUCTION}"
    )


def group_candidates(
    candidates: Sequence[ActiveLearningCandidate],
    *,
    near_duplicate_threshold: float = 0.85,
) -> dict[str, list[ActiveLearningCandidate]]:
    """Cluster candidates so that related examples never span two splits.

    Two candidates join the same group when they share a task signature or when
    their queries are near-duplicates. Iteration order is fixed by
    ``candidate_id`` so grouping is reproducible.
    """
    groups: dict[str, list[ActiveLearningCandidate]] = {}
    signature_index: dict[str, str] = {}
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        key = signature_index.get(candidate.task_signature)
        if key is None:
            for existing_key, members in groups.items():
                if any(
                    is_near_duplicate(
                        candidate.user_query,
                        member.user_query,
                        threshold=near_duplicate_threshold,
                    )
                    for member in members
                ):
                    key = existing_key
                    break
        if key is None:
            key = candidate.query_hash or candidate.candidate_id
            groups[key] = []
        groups.setdefault(key, []).append(candidate)
        if candidate.task_signature:
            signature_index.setdefault(candidate.task_signature, key)
    return groups


def assign_split(
    group_key: str,
    *,
    dataset_version: str,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> DatasetSplit:
    """Deterministically place a whole group into one split.

    Hashing the group key (not the row) is what keeps near-duplicates out of
    both train and test. The same dataset version always yields the same
    assignment, which is what freezes the test set.
    """
    digest = hashlib.sha256(f"{dataset_version}:{group_key}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / float(1 << 64)
    train, validation, _ = ratios
    if position < train:
        return DatasetSplit.TRAIN
    if position < train + validation:
        return DatasetSplit.VALIDATION
    return DatasetSplit.TEST


def export_dataset(
    candidates: Sequence[ActiveLearningCandidate],
    *,
    dataset_version: str,
    task: ExportTask = ExportTask.ANALYSIS_PLAN,
    exported_at: datetime,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    near_duplicate_threshold: float = 0.85,
) -> DatasetExport:
    """Build a versioned SFT dataset from approved candidates."""
    if not dataset_version.strip():
        raise DatasetExportError("dataset_version must not be empty")

    eligible = [candidate for candidate in candidates if is_exportable(candidate)]
    groups = group_candidates(eligible, near_duplicate_threshold=near_duplicate_threshold)
    group_of: dict[str, str] = {
        candidate.candidate_id: key for key, members in groups.items() for candidate in members
    }

    splits: dict[str, list[TrainingExample]] = {split.value: [] for split in DatasetSplit}
    split_ids: dict[str, list[str]] = {split.value: [] for split in DatasetSplit}
    lineage: list[LineageEntry] = []
    skipped: list[SkippedCandidate] = []

    for candidate in sorted(eligible, key=lambda item: item.candidate_id):
        payload = _target_payload(candidate, task)
        if payload is None:
            skipped.append(
                SkippedCandidate(
                    candidate_id=candidate.candidate_id,
                    reason=f"no {task.value} target available",
                )
            )
            continue
        try:
            target = _validated_target(payload, task)
        except ValidationError as exc:
            skipped.append(
                SkippedCandidate(
                    candidate_id=candidate.candidate_id,
                    reason=f"target rejected by {task.value} schema ({exc.error_count()} errors)",
                )
            )
            continue

        group_key = group_of[candidate.candidate_id]
        split = assign_split(group_key, dataset_version=dataset_version, ratios=ratios)
        splits[split.value].append(
            TrainingExample(
                messages=[
                    ChatMessage(role="user", content=_user_content(candidate, task, target)),
                    ChatMessage(role="assistant", content=canonical_json(target)),
                ]
            )
        )
        split_ids[split.value].append(candidate.candidate_id)
        lineage.append(
            LineageEntry(
                candidate_id=candidate.candidate_id,
                request_id=candidate.request_id,
                review_status=candidate.review_status,
                split=split,
                group_key=group_key,
                model_id=candidate.model_id,
                prompt_version=candidate.prompt_version,
                tool_schema_version=candidate.tool_schema_version,
                corrected=candidate.corrected_output is not None,
            )
        )

    schema_model: type[BaseModel] = (
        AnalysisPlan if task is ExportTask.ANALYSIS_PLAN else MultiTargetComparisonPlan
    )
    statuses: dict[str, int] = {}
    for entry in lineage:
        statuses[entry.review_status.value] = statuses.get(entry.review_status.value, 0) + 1

    manifest = DatasetManifest(
        dataset_version=dataset_version,
        task=task,
        candidate_schema_version=(
            eligible[0].schema_version
            if eligible
            else ActiveLearningCandidate.model_fields["schema_version"].default
        ),
        target_schema_name=schema_model.__name__,
        exported_at=exported_at,
        example_count=len(lineage),
        split_counts={key: len(value) for key, value in sorted(split_ids.items())},
        split_candidate_ids={key: list(value) for key, value in sorted(split_ids.items())},
        lineage=lineage,
        source_models=sorted({entry.model_id for entry in lineage if entry.model_id}),
        prompt_versions=sorted({entry.prompt_version for entry in lineage if entry.prompt_version}),
        reviewer_statuses=dict(sorted(statuses.items())),
        skipped=skipped,
    )

    export = DatasetExport(
        manifest=manifest,
        splits=splits,
        target_json_schema=schema_model.model_json_schema(),
    )
    digest = hashlib.sha256()
    for split in DatasetSplit:
        digest.update(export.jsonl(split).encode("utf-8"))
    return export.model_copy(
        update={"manifest": manifest.model_copy(update={"content_digest": digest.hexdigest()})}
    )


def write_export(export: DatasetExport, directory: Path) -> dict[str, Path]:
    """Write split files, the manifest and the authoritative target schema."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{export.manifest.task.value}_{export.manifest.dataset_version}"
    written: dict[str, Path] = {}
    for split in DatasetSplit:
        path = directory / f"{stem}.{split.value}.jsonl"
        path.write_text(export.jsonl(split), encoding="utf-8")
        written[split.value] = path
    manifest_path = directory / f"{stem}.manifest.json"
    manifest_path.write_text(
        json.dumps(export.manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    written["manifest"] = manifest_path
    schema_path = directory / f"{stem}.schema.json"
    schema_path.write_text(
        json.dumps(export.target_json_schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written["schema"] = schema_path
    return written
