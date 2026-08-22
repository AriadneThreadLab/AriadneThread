"""End-to-end fine-tuning pipeline.

    exported dataset -> validation -> baseline evaluation -> QLoRA SFT
    -> adapter -> evaluation on the same frozen test set -> registry entry

The trainer and the model runners are injected, so the whole sequence can be
exercised offline with fakes. Nothing in here deploys anything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ariadne_finetuning.config import FineTuningConfig
from ariadne_finetuning.dataset import DatasetBundle, load_bundle, validate_bundle
from ariadne_finetuning.evaluation import (
    EvaluationReport,
    ModelRunner,
    evaluate_on_test_split,
)
from ariadne_finetuning.evaluation import compare as compare_reports
from ariadne_finetuning.registry import (
    ExperimentRecord,
    ExperimentRegistry,
    ExperimentStatus,
    current_git_commit,
)

#: ``(config, adapter_path_or_None) -> runner``
RunnerFactory = Callable[[FineTuningConfig, Path | None], ModelRunner]
#: ``(config, bundle) -> adapter directory``
TrainerFn = Callable[[FineTuningConfig, DatasetBundle], Path]


class PipelineError(RuntimeError):
    """The pipeline refused to continue."""


@dataclass(frozen=True)
class PipelineResult:
    """Everything the completion report needs."""

    config: FineTuningConfig
    bundle: DatasetBundle
    baseline: EvaluationReport
    finetuned: EvaluationReport | None
    comparison: dict[str, Any] | None
    adapter_path: Path | None
    experiment: ExperimentRecord | None

    def summary(self) -> dict[str, Any]:
        return {
            "experiment_id": self.config.experiment_id,
            "base_model_id": self.config.base_model_id,
            "dataset_version": self.bundle.dataset_version,
            "test_examples": len(self.bundle.test),
            "baseline": self.baseline.as_dict(),
            "finetuned": self.finetuned.as_dict() if self.finetuned else None,
            "comparison": self.comparison,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "status": self.experiment.status.value if self.experiment else None,
        }


def run_pipeline(
    config: FineTuningConfig,
    *,
    runner_factory: RunnerFactory,
    trainer: TrainerFn,
    registry: ExperimentRegistry,
    bundle: DatasetBundle | None = None,
    baseline_only: bool = False,
    now: datetime | None = None,
    repo_root: Path | None = None,
) -> PipelineResult:
    """Validate, measure, train, measure again, record. Never deploy."""
    resolved = bundle or load_bundle(
        config.dataset_dir, task=config.task, dataset_version=config.dataset_version
    )
    problems = validate_bundle(resolved)
    if problems:
        raise PipelineError("dataset validation failed: " + "; ".join(problems))

    timestamp = now or datetime.now(tz=timezone.utc)

    # Baseline first: without it, a post-training number means nothing.
    baseline = evaluate_on_test_split(runner_factory(config, None), resolved, now=timestamp)
    if baseline_only:
        return PipelineResult(
            config=config,
            bundle=resolved,
            baseline=baseline,
            finetuned=None,
            comparison=None,
            adapter_path=None,
            experiment=None,
        )

    adapter_path = trainer(config, resolved)
    finetuned = evaluate_on_test_split(
        runner_factory(config, adapter_path), resolved, now=timestamp
    )
    # Raises if the two runs did not see the same frozen examples.
    comparison = compare_reports(baseline, finetuned)

    experiment = registry.record(
        ExperimentRecord(
            experiment_id=config.experiment_id,
            base_model_id=config.base_model_id,
            dataset_version=resolved.dataset_version,
            dataset_digest=resolved.test_digest,
            task=config.task,
            adapter_path=str(adapter_path),
            training_config=config.to_dict(),
            baseline_metrics=baseline.metrics.as_dict(),
            finetuned_metrics=finetuned.metrics.as_dict(),
            # Always experimental. Promotion is a separate, human step.
            status=ExperimentStatus.EXPERIMENTAL,
            created_at=timestamp,
            git_commit=current_git_commit(repo_root or Path.cwd()),
        )
    )

    return PipelineResult(
        config=config,
        bundle=resolved,
        baseline=baseline,
        finetuned=finetuned,
        comparison=comparison,
        adapter_path=adapter_path,
        experiment=experiment,
    )
