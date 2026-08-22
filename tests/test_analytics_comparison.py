"""ComparisonEngine behaviour."""

from __future__ import annotations

from datetime import datetime, timezone

from app.analytics.comparison import ComparisonEngine
from app.analytics.contracts import (
    AnalysisPlan,
    AnalysisResult,
    AnalysisTarget,
    CalculationProvenance,
    DataProvenance,
    MetricRequest,
    MetricResult,
    TargetAnalysisResult,
)


def _metric(
    metric: str,
    value: float,
    *,
    role: str = "primary",
    direction: str = "higher_is_better",
    target: str = "a",
) -> MetricResult:
    return MetricResult(
        metric=metric,  # type: ignore[arg-type]
        role=role,  # type: ignore[arg-type]
        label=metric,
        status="computed",
        value=value,
        unit="features",
        direction=direction,  # type: ignore[arg-type]
        observation_count=3,
        candidate_count=3,
        missing_count=0,
        invalid_count=0,
        provenance=CalculationProvenance(
            metric=metric,  # type: ignore[arg-type]
            dataset_ref="osm_result_1",
            target_ref=target,
            implementation_id="SpatialAnalyticsEngine.count",
            observation_count=3,
            candidate_count=3,
            missing_count=0,
            invalid_count=0,
            unit="features",
            metric_catalog_version="metric-catalog-1",
        ),
    )


def _result(values: dict[str, float], *, direction: str = "higher_is_better") -> AnalysisResult:
    targets = []
    for index, (tid, value) in enumerate(values.items(), start=1):
        targets.append(
            TargetAnalysisResult(
                target_id=tid,
                label=tid.upper(),
                dataset_ref=f"osm_result_{index}",
                data_provenance=DataProvenance(
                    feature_concept="parks",
                    resolved_tags=["leisure=park"],
                    dataset_ref=f"osm_result_{index}",
                    target_id=tid,
                    analysis_scope="scope",
                    scope_kind="point",
                    retrieved_feature_count=3,
                    valid_observation_count=3,
                    missing_observation_count=0,
                    invalid_observation_count=0,
                    retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
                ),
                metrics=[_metric("count", value, direction=direction, target=tid)],
            )
        )
    return AnalysisResult(
        analysis_type="comparison",
        feature_concept="parks",
        comparison_goal="Compare counts",
        targets=targets,
        metric_catalog_version="metric-catalog-1",
        ruleset_version="metric-rules-1",
    )


def test_higher_is_better() -> None:
    plan = AnalysisPlan(
        analysis_type="comparison",
        feature_concept="parks",
        comparison_goal="Which has more parks",
        targets=[
            AnalysisTarget(target_id="a", label="A", dataset_ref="osm_result_1"),
            AnalysisTarget(target_id="b", label="B", dataset_ref="osm_result_2"),
        ],
        metrics=[MetricRequest(metric="count", role="primary", inferred_goal="abundance")],
    )
    comparison = ComparisonEngine().compare(plan, _result({"a": 10, "b": 4}))
    assert comparison.primary is not None
    assert comparison.primary.preferred_target_id == "a"
    assert (
        "better" in comparison.overall_statement.lower() or "higher" in comparison.overall_statement
    )


def test_neutral_stddev_has_no_normative_language() -> None:
    plan = AnalysisPlan(
        analysis_type="comparison",
        feature_concept="parks",
        comparison_goal="Consistency",
        targets=[
            AnalysisTarget(target_id="a", label="A", dataset_ref="osm_result_1"),
            AnalysisTarget(target_id="b", label="B", dataset_ref="osm_result_2"),
        ],
        metrics=[
            MetricRequest(
                metric="standard_deviation",
                role="primary",
                inferred_goal="variability",
                property_key="capacity",
                user_explicit=True,
            )
        ],
    )
    # Build result with stddev metrics
    result = _result({"a": 2.0, "b": 8.0}, direction="neutral")
    # Replace metric type on results
    updated = []
    for target in result.targets:
        m = target.metrics[0].model_copy(
            update={
                "metric": "standard_deviation",
                "label": "Standard deviation",
                "unit": "capacity (OSM tag units)",
                "direction": "neutral",
            }
        )
        updated.append(target.model_copy(update={"metrics": [m]}))
    result = result.model_copy(update={"targets": updated})
    comparison = ComparisonEngine().compare(plan, result)
    assert comparison.primary is not None
    assert comparison.primary.preferred_target_id is None
    text = comparison.overall_statement.lower()
    for token in ("better", "worse", "superior", "best", "worst"):
        assert token not in text


def test_tie_on_counts() -> None:
    plan = AnalysisPlan(
        analysis_type="comparison",
        feature_concept="parks",
        comparison_goal="Compare counts",
        targets=[
            AnalysisTarget(target_id="a", label="A", dataset_ref="osm_result_1"),
            AnalysisTarget(target_id="b", label="B", dataset_ref="osm_result_2"),
        ],
        metrics=[MetricRequest(metric="count", role="primary", inferred_goal="abundance")],
    )
    comparison = ComparisonEngine().compare(plan, _result({"a": 5, "b": 5}))
    assert comparison.primary is not None
    assert comparison.primary.verdict == "tie"


def _result_with_counts(
    values: dict[str, float],
    *,
    truncated_ids: set[str] | None = None,
    limit: int = 50,
) -> AnalysisResult:
    truncated_ids = truncated_ids or set()
    result = _result(values)
    updated = []
    for target in result.targets:
        count = int(values[target.target_id])
        truncated = target.target_id in truncated_ids
        prov = target.data_provenance.model_copy(
            update={
                "retrieved_feature_count": count,
                "truncated": truncated,
                "limit_reached": truncated,
                "effective_limit": limit,
            }
        )
        metric = target.metrics[0].model_copy(
            update={
                "value": float(count),
                "observation_count": count,
                "candidate_count": count,
            }
        )
        updated.append(target.model_copy(update={"data_provenance": prov, "metrics": [metric]}))
    return result.model_copy(update={"targets": updated})


def test_truncated_count_does_not_claim_exact_relative_difference() -> None:
    plan = AnalysisPlan(
        analysis_type="comparison",
        feature_concept="parks",
        comparison_goal="Which has more parks",
        targets=[
            AnalysisTarget(target_id="a", label="A", dataset_ref="osm_result_1"),
            AnalysisTarget(target_id="b", label="B", dataset_ref="osm_result_2"),
        ],
        metrics=[MetricRequest(metric="count", role="primary", inferred_goal="abundance")],
    )
    comparison = ComparisonEngine().compare(
        plan,
        _result_with_counts({"a": 25, "b": 50}, truncated_ids={"b"}),
    )
    text = comparison.overall_statement.lower()
    assert "% difference" not in comparison.overall_statement
    assert "100" not in comparison.overall_statement
    assert "retrieval limit" in text
    assert "lower bound" in text
    assert comparison.overall_confidence == "close"
