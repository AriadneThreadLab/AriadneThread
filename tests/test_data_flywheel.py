"""The handoff between Ariadne and the fine-tuning pipeline.

The two systems share exactly one thing: a versioned dataset directory. This
test writes that directory with the application exporter and reads it back with
the training package, which is the only way to catch a drift between them.

``finetuning/src`` is added to ``sys.path`` here because the training package is
deliberately *not* installed with the web service.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from app.active_learning.contracts import (
    ActiveLearningCandidate,
    DatasetSplit,
    ExportTask,
    ReviewStatus,
    TaskType,
)
from app.active_learning.export import export_dataset, write_export

from tests.active_learning_fixtures import ANALYSIS_PLAN, COMPARISON_QUERY

REPO_ROOT = Path(__file__).resolve().parents[1]
FINETUNING_SRC = REPO_ROOT / "finetuning" / "src"

if str(FINETUNING_SRC) not in sys.path:
    sys.path.insert(0, str(FINETUNING_SRC))

dataset_module = pytest.importorskip("ariadne_finetuning.dataset")
schema_module = pytest.importorskip("ariadne_finetuning.schema")

EXPORTED_AT = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)

QUESTIONS = [
    "Compare public parks within 2 km of Campus North and Campus South.",
    "Compare libraries within 1 km of Station East and Station West.",
    "Compare clinics within 3 km of District One and District Two.",
    "Compare museums within 4 km of Quarter Alpha and Quarter Beta.",
    "Compare playgrounds within 900 m of Park Gate and Market Gate.",
    COMPARISON_QUERY,
]


def approved_candidates() -> list[ActiveLearningCandidate]:
    return [
        ActiveLearningCandidate(
            candidate_id=f"candidate-{index:04d}",
            request_id=f"req-{index}",
            user_query=question,
            task_type=TaskType.COMPARISON,
            task_signature=f"comparison|feature{index}|radius|count",
            query_hash=f"hash-{index}",
            analysis_plan=ANALYSIS_PLAN,
            outcome_successful=True,
            review_status=ReviewStatus.APPROVED,
            approved_for_training=True,
            model_id="deepseek-r1:7b",
            prompt_version="ariadne-system-prompt-1",
        )
        for index, question in enumerate(QUESTIONS)
    ]


def test_exported_dataset_is_readable_by_the_finetuning_pipeline(tmp_path):
    export = export_dataset(
        approved_candidates(),
        dataset_version="v1",
        task=ExportTask.ANALYSIS_PLAN,
        exported_at=EXPORTED_AT,
    )
    write_export(export, tmp_path)

    bundle = dataset_module.load_bundle(tmp_path, task="analysis_plan", dataset_version="v1")

    assert bundle.dataset_version == "v1"
    total = len(bundle.train) + len(bundle.validation) + len(bundle.test)
    assert total == export.manifest.example_count
    assert bundle.manifest["target_schema_name"] == "AnalysisPlan"
    # The plan schema Ariadne generated is the schema training validates against.
    assert bundle.target_schema["title"] == "AnalysisPlan"


def test_every_exported_target_satisfies_the_exported_schema(tmp_path):
    export = export_dataset(approved_candidates(), dataset_version="v1", exported_at=EXPORTED_AT)
    write_export(export, tmp_path)
    bundle = dataset_module.load_bundle(tmp_path, task="analysis_plan", dataset_version="v1")

    for split in ("train", "validation", "test"):
        for sample in bundle.split(split):
            assert schema_module.is_valid(sample.reference(), bundle.target_schema)


def test_the_exporter_produces_a_leakage_free_bundle(tmp_path):
    export = export_dataset(approved_candidates(), dataset_version="v1", exported_at=EXPORTED_AT)
    write_export(export, tmp_path)
    bundle = dataset_module.load_bundle(tmp_path, task="analysis_plan", dataset_version="v1")

    assert dataset_module.leakage_report(bundle) == []


def test_re_exporting_the_same_version_reproduces_the_frozen_test_set(tmp_path):
    candidates = approved_candidates()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    write_export(
        export_dataset(candidates, dataset_version="v1", exported_at=EXPORTED_AT), first_dir
    )
    write_export(
        export_dataset(
            list(reversed(candidates)),
            dataset_version="v1",
            exported_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        ),
        second_dir,
    )

    first = dataset_module.load_bundle(first_dir, task="analysis_plan", dataset_version="v1")
    second = dataset_module.load_bundle(second_dir, task="analysis_plan", dataset_version="v1")

    assert first.test_digest == second.test_digest


def test_a_new_dataset_version_may_reshuffle_but_stays_self_consistent(tmp_path):
    candidates = approved_candidates()
    write_export(
        export_dataset(candidates, dataset_version="v1", exported_at=EXPORTED_AT),
        tmp_path / "v1",
    )
    write_export(
        export_dataset(candidates, dataset_version="v2", exported_at=EXPORTED_AT),
        tmp_path / "v2",
    )

    v1 = dataset_module.load_bundle(tmp_path / "v1", task="analysis_plan", dataset_version="v1")
    v2 = dataset_module.load_bundle(tmp_path / "v2", task="analysis_plan", dataset_version="v2")

    assert dataset_module.validate_bundle(v1) == []
    assert dataset_module.validate_bundle(v2) == []


def test_split_membership_is_recorded_for_lineage(tmp_path):
    export = export_dataset(approved_candidates(), dataset_version="v1", exported_at=EXPORTED_AT)

    for entry in export.manifest.lineage:
        assert entry.candidate_id in export.manifest.split_candidate_ids[entry.split.value]
    counted = sum(export.manifest.split_counts[split.value] for split in DatasetSplit)
    assert counted == export.manifest.example_count
