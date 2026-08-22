"""Evaluation against a frozen test set.

The base model is measured before training and the adapter after it, on exactly
the same examples. :class:`EvaluationReport` carries the dataset digest so a
before/after comparison can prove that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from ariadne_finetuning.dataset import ChatSample, DatasetBundle
from ariadne_finetuning.metrics import PlanMetrics, SampleOutcome, aggregate, score_sample


class EvaluationError(ValueError):
    """Evaluation was asked to do something meaningless."""


@runtime_checkable
class ModelRunner(Protocol):
    """Anything that turns a prompt into a plan string.

    Keeping this a protocol is what lets the offline test suite evaluate the
    pipeline without transformers, a GPU or a download.
    """

    @property
    def label(self) -> str: ...

    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class EvaluationReport:
    """Metrics plus everything needed to trust the comparison."""

    model_label: str
    dataset_version: str
    split: str
    test_digest: str
    metrics: PlanMetrics
    evaluated_at: datetime
    prompt_keys: tuple[str, ...] = ()
    outcomes: tuple[SampleOutcome, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_label": self.model_label,
            "dataset_version": self.dataset_version,
            "split": self.split,
            "test_digest": self.test_digest,
            "evaluated_at": self.evaluated_at.isoformat(),
            "metrics": self.metrics.as_dict(),
        }


def evaluate_samples(
    runner: ModelRunner,
    samples: Sequence[ChatSample],
    *,
    schema: dict[str, Any],
    dataset_version: str,
    test_digest: str,
    split: str = "test",
    now: datetime | None = None,
) -> EvaluationReport:
    """Run one model over a fixed sample sequence."""
    if not samples:
        raise EvaluationError("refusing to report metrics for an empty evaluation set")

    outcomes: list[SampleOutcome] = []
    for sample in samples:
        generated = runner.generate(sample.prompt)
        outcomes.append(score_sample(generated, sample.reference(), schema))

    return EvaluationReport(
        model_label=runner.label,
        dataset_version=dataset_version,
        split=split,
        test_digest=test_digest,
        metrics=aggregate(outcomes),
        evaluated_at=now or datetime.now(tz=timezone.utc),
        prompt_keys=tuple(sample.prompt_key for sample in samples),
        outcomes=tuple(outcomes),
    )


def evaluate_on_test_split(
    runner: ModelRunner,
    bundle: DatasetBundle,
    *,
    now: datetime | None = None,
) -> EvaluationReport:
    """Evaluate on the frozen test split of an exported dataset."""
    return evaluate_samples(
        runner,
        bundle.test,
        schema=bundle.target_schema,
        dataset_version=bundle.dataset_version,
        test_digest=bundle.test_digest,
        now=now,
    )


def assert_comparable(baseline: EvaluationReport, candidate: EvaluationReport) -> None:
    """Refuse to compare two reports that did not see the same examples."""
    if baseline.test_digest != candidate.test_digest:
        raise EvaluationError(
            "reports were produced against different test sets; the comparison is invalid"
        )
    if baseline.prompt_keys != candidate.prompt_keys:
        raise EvaluationError("reports did not evaluate the same prompts in the same order")


def compare(baseline: EvaluationReport, candidate: EvaluationReport) -> dict[str, Any]:
    """Before/after table for the completion report."""
    assert_comparable(baseline, candidate)
    before = baseline.metrics.as_dict()
    after = candidate.metrics.as_dict()
    rows: dict[str, Any] = {}
    for key, base_value in before.items():
        tuned_value = after.get(key)
        if isinstance(base_value, int | float) and isinstance(tuned_value, int | float):
            rows[key] = {
                "baseline": base_value,
                "finetuned": tuned_value,
                "delta": round(tuned_value - base_value, 4),
            }
        else:
            rows[key] = {"baseline": base_value, "finetuned": tuned_value, "delta": None}
    return {
        "baseline_model": baseline.model_label,
        "finetuned_model": candidate.model_label,
        "dataset_version": baseline.dataset_version,
        "test_digest": baseline.test_digest,
        "example_count": baseline.metrics.example_count,
        "metrics": rows,
    }
