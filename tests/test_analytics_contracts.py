"""AnalysisPlan validation contracts."""

from __future__ import annotations

import pytest
from app.analytics.contracts import AnalysisPlan, AnalysisTarget, MetricRequest
from pydantic import ValidationError


def _target(tid: str = "a", ref: str = "osm_result_1") -> AnalysisTarget:
    return AnalysisTarget(target_id=tid, label=f"Target {tid}", dataset_ref=ref)


def test_comparison_requires_two_targets() -> None:
    with pytest.raises(ValidationError, match="at least two targets"):
        AnalysisPlan(
            analysis_type="comparison",
            feature_concept="parks",
            comparison_goal="Compare park counts",
            targets=[_target()],
            metrics=[
                MetricRequest(metric="count", role="primary", inferred_goal="abundance"),
            ],
        )


def test_exactly_one_primary_metric() -> None:
    with pytest.raises(ValidationError, match="exactly one metric"):
        AnalysisPlan(
            analysis_type="comparison",
            feature_concept="parks",
            comparison_goal="Compare park counts",
            targets=[_target("a"), _target("b", "osm_result_2")],
            metrics=[
                MetricRequest(metric="count", role="supporting", inferred_goal="abundance"),
                MetricRequest(metric="density", role="supporting", inferred_goal="concentration"),
            ],
        )


def test_property_key_required_for_mean() -> None:
    with pytest.raises(ValidationError, match="property_key"):
        AnalysisPlan(
            analysis_type="single_target",
            feature_concept="parks",
            comparison_goal="Mean capacity",
            targets=[_target()],
            metrics=[
                MetricRequest(
                    metric="mean",
                    role="primary",
                    inferred_goal="typical_value",
                    user_explicit=True,
                ),
            ],
        )


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate(
            {
                "analysis_type": "single_target",
                "feature_concept": "parks",
                "comparison_goal": "Count parks",
                "targets": [
                    {
                        "target_id": "a",
                        "label": "A",
                        "dataset_ref": "osm_result_1",
                    }
                ],
                "metrics": [
                    {
                        "metric": "count",
                        "role": "primary",
                        "inferred_goal": "abundance",
                    }
                ],
                "formula": "len(x)",
            }
        )


def test_protocol_markers_rejected_in_labels() -> None:
    with pytest.raises(ValidationError):
        AnalysisPlan(
            analysis_type="single_target",
            feature_concept="tool_calls",
            comparison_goal="Count parks here",
            targets=[_target()],
            metrics=[
                MetricRequest(metric="count", role="primary", inferred_goal="abundance"),
            ],
        )


def test_unknown_metric_rejected() -> None:
    with pytest.raises(ValidationError):
        MetricRequest(metric="custom_score", role="primary", inferred_goal="abundance")  # type: ignore[arg-type]
