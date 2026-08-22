"""Plan metrics, frozen-test-set evaluation and before/after comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ariadne_finetuning.dataset import load_bundle, load_jsonl
from ariadne_finetuning.evaluation import (
    EvaluationError,
    compare,
    evaluate_on_test_split,
    evaluate_samples,
)
from ariadne_finetuning.metrics import aggregate, extract_json_object, score_sample
from ariadne_finetuning.schema import is_valid, schema_errors
from conftest import ANALYSIS_PLAN_SCHEMA, plan

GOLD_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "gold"


class ScriptedRunner:
    """Returns a canned generation per prompt; records what it was asked."""

    def __init__(self, replies: dict[str, str] | str, label: str = "fake") -> None:
        self._replies = replies
        self._label = label
        self.seen: list[str] = []

    @property
    def label(self) -> str:
        return self._label

    def generate(self, prompt: str) -> str:
        self.seen.append(prompt)
        if isinstance(self._replies, str):
            return self._replies
        return self._replies.get(prompt, "")


def as_text(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True)


# --- schema ---------------------------------------------------------------


def test_valid_plan_passes_the_exported_schema():
    assert is_valid(plan(), ANALYSIS_PLAN_SCHEMA)


def test_missing_required_field_fails_the_schema():
    broken = plan()
    del broken["targets"]
    assert "targets" in " ".join(schema_errors(broken, ANALYSIS_PLAN_SCHEMA))


def test_unknown_field_fails_the_schema():
    assert not is_valid({**plan(), "radius_m": 2000}, ANALYSIS_PLAN_SCHEMA)


def test_invalid_enum_value_fails_the_schema():
    assert not is_valid({**plan(), "analysis_type": "clustering"}, ANALYSIS_PLAN_SCHEMA)


# --- metrics --------------------------------------------------------------


def test_perfect_generation_scores_everything_correct():
    reference = plan()
    outcome = score_sample(as_text(reference), reference, ANALYSIS_PLAN_SCHEMA)

    assert outcome.parsed and outcome.schema_valid
    assert outcome.analysis_type is True
    assert outcome.targets_preserved is True
    assert outcome.feature_concept is True
    assert outcome.metric_intent is True
    assert outcome.hallucinated_fields == 0
    assert outcome.invented_coordinates == 0


def test_dropped_target_is_caught_even_when_the_plan_parses():
    reference = plan()
    prediction = plan(targets=(("ut", "University of Tehran", "osm_result_1"),))
    prediction["analysis_type"] = "single_target"

    outcome = score_sample(as_text(prediction), reference, ANALYSIS_PLAN_SCHEMA)

    assert outcome.parsed is True
    assert outcome.targets_preserved is False
    assert outcome.analysis_type is False


def test_wrong_primary_metric_is_caught():
    outcome = score_sample(as_text(plan(metric="density")), plan(), ANALYSIS_PLAN_SCHEMA)
    assert outcome.metric_intent is False


def test_invented_coordinates_are_counted():
    reference = plan()
    prediction = plan()
    prediction["targets"][0]["reference_points"] = [{"lat": 35.7, "lon": 51.4}]  # type: ignore[index]

    outcome = score_sample(as_text(prediction), reference, ANALYSIS_PLAN_SCHEMA)

    assert outcome.invented_coordinates == 2


def test_hallucinated_top_level_field_is_counted():
    outcome = score_sample(
        as_text({**plan(), "overpass_query": "[out:json];"}), plan(), ANALYSIS_PLAN_SCHEMA
    )
    assert outcome.hallucinated_fields == 1
    assert outcome.schema_valid is False


def test_unparsable_generation_is_a_parse_failure_not_a_crash():
    outcome = score_sample("I think you want parks near campus.", plan(), ANALYSIS_PLAN_SCHEMA)

    assert outcome.parsed is False
    assert outcome.schema_valid is False
    metrics = aggregate([outcome])
    assert metrics.parse_failure_count == 1
    assert metrics.valid_analysis_plan_rate == 0.0


def test_json_is_recovered_from_a_fenced_reply():
    text = '```json\n{"analysis_type": "comparison"}\n```'
    assert extract_json_object(text) == {"analysis_type": "comparison"}


def test_inapplicable_metrics_report_null_rather_than_zero():
    # AnalysisPlan has no radius_m, so radius accuracy must not read as 0%.
    metrics = aggregate([score_sample(as_text(plan()), plan(), ANALYSIS_PLAN_SCHEMA)])

    assert metrics.radius_accuracy is None
    assert metrics.applicability["radius_accuracy"] == 0
    assert metrics.analysis_type_accuracy == 1.0
    assert metrics.applicability["analysis_type_accuracy"] == 1


def test_metrics_are_reported_per_dimension_not_as_one_number():
    payload = aggregate([score_sample(as_text(plan()), plan(), ANALYSIS_PLAN_SCHEMA)]).as_dict()

    for key in (
        "valid_analysis_plan_rate",
        "analysis_type_accuracy",
        "target_preservation_accuracy",
        "radius_accuracy",
        "feature_concept_accuracy",
        "comparison_goal_accuracy",
        "metric_intent_accuracy",
        "hallucinated_field_rate",
        "invented_coordinate_rate",
    ):
        assert key in payload


# --- evaluation -----------------------------------------------------------


def test_evaluation_refuses_an_empty_test_set():
    with pytest.raises(EvaluationError):
        evaluate_samples(
            ScriptedRunner(""),
            [],
            schema=ANALYSIS_PLAN_SCHEMA,
            dataset_version="v1",
            test_digest="abc",
        )


def test_baseline_and_finetuned_must_see_the_same_examples(dataset_dir):
    bundle = load_bundle(dataset_dir, task="analysis_plan", dataset_version="v1")
    reference = bundle.test[0].completion

    base = ScriptedRunner("not a plan", label="base")
    tuned = ScriptedRunner(reference, label="finetuned")

    baseline_report = evaluate_on_test_split(base, bundle)
    tuned_report = evaluate_on_test_split(tuned, bundle)

    assert base.seen == tuned.seen, "both models must be asked exactly the same prompts"
    assert baseline_report.test_digest == tuned_report.test_digest

    table = compare(baseline_report, tuned_report)
    assert table["metrics"]["valid_analysis_plan_rate"]["baseline"] == 0.0
    assert table["metrics"]["valid_analysis_plan_rate"]["finetuned"] == 1.0
    assert table["metrics"]["valid_analysis_plan_rate"]["delta"] == 1.0


def test_comparison_across_different_test_sets_is_refused(dataset_dir):
    bundle = load_bundle(dataset_dir, task="analysis_plan", dataset_version="v1")
    baseline_report = evaluate_on_test_split(ScriptedRunner("", label="base"), bundle)
    other = evaluate_samples(
        ScriptedRunner("", label="finetuned"),
        bundle.test,
        schema=bundle.target_schema,
        dataset_version="v1",
        test_digest="a-different-digest",
    )

    with pytest.raises(EvaluationError):
        compare(baseline_report, other)


# --- gold set -------------------------------------------------------------


def test_gold_set_is_frozen_curated_and_schema_valid():
    samples = load_jsonl(GOLD_DIR / "analysis_plan_gold_v1.test.jsonl")
    schema = json.loads(
        (GOLD_DIR / "analysis_plan_gold_v1.schema.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (GOLD_DIR / "analysis_plan_gold_v1.manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["frozen"] is True
    assert manifest["split_counts"]["train"] == 0
    assert len(samples) == manifest["example_count"]
    for sample in samples:
        assert is_valid(sample.reference(), schema)


def test_gold_set_covers_the_metric_selection_distinctions():
    samples = load_jsonl(GOLD_DIR / "analysis_plan_gold_v1.test.jsonl")
    metrics = {
        item["metric"]
        for sample in samples
        for item in sample.reference()["metrics"]
        if item["role"] == "primary"
    }
    assert {"count", "density", "total_area", "median_area", "mean_area"} <= metrics


def test_a_perfect_model_scores_perfectly_on_the_gold_set():
    samples = load_jsonl(GOLD_DIR / "analysis_plan_gold_v1.test.jsonl")
    schema = json.loads(
        (GOLD_DIR / "analysis_plan_gold_v1.schema.json").read_text(encoding="utf-8")
    )
    replies = {sample.prompt: sample.completion for sample in samples}

    report = evaluate_samples(
        ScriptedRunner(replies, label="oracle"),
        samples,
        schema=schema,
        dataset_version="gold_v1",
        test_digest="gold",
    )

    assert report.metrics.valid_analysis_plan_rate == 1.0
    assert report.metrics.metric_intent_accuracy == 1.0
    assert report.metrics.invented_coordinate_rate == 0.0
