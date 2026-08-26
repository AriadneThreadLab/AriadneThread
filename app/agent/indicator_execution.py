"""Build an AnalysisBlock from deterministic indicator results.

This adapter sits in the agent layer so indicator modules stay free of
orchestration and the Metric Catalog plan schema stays closed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.analytics.catalog import METRIC_CATALOG_VERSION
from app.analytics.contracts import (
    AnalysisBlock,
    AnalysisDecisionTrace,
    AnalysisPlan,
    AnalysisResult,
    AnalysisTarget,
    CalculationProvenance,
    DataProvenance,
    GeoPoint,
    MetricRequest,
    MetricResult,
    MetricSelectionEvidence,
    MetricSelectionTrace,
    TargetAnalysisResult,
)
from app.analytics.datasets import DatasetRecord
from app.analytics.methods import METHOD_REGISTRY
from app.analytics.rules import RULESET_VERSION
from app.indicators.catalog import INDICATOR_CATALOG_VERSION, get_indicator
from app.indicators.compare import compare_indicator_targets, plan_metric_type
from app.indicators.compute import compute_indicator
from app.indicators.contracts import (
    IndicatorComputationResult,
    IndicatorComputeRequest,
    IndicatorSelectionTrace,
)
from app.indicators.report import assemble_indicator_report
from app.places.contracts import ResolvedPlaceRecord


def execute_selected_indicator(
    *,
    selection: IndicatorSelectionTrace,
    target_rows: tuple[tuple[str, str, dict[str, DatasetRecord]], ...],
    places: tuple[ResolvedPlaceRecord | None, ...],
    tags: tuple[str, ...],
    extra_warnings: tuple[str, ...] = (),
) -> AnalysisBlock:
    """Compute the selected indicator for every target and assemble analysis."""
    indicator_id = selection.selection.primary_indicator_id
    if indicator_id is None:
        raise ValueError("indicator selection has no primary indicator")
    definition = get_indicator(indicator_id)
    metric = plan_metric_type(definition.method_id)
    spec = METHOD_REGISTRY[definition.method_id]
    needs_ref = any(item.kind == "reference_point" for item in definition.requirements)
    analysis_type = "comparison" if len(target_rows) >= 2 else "single_target"

    computed: list[tuple[str, str, IndicatorComputationResult]] = []
    plan_targets: list[AnalysisTarget] = []
    result_targets: list[TargetAnalysisResult] = []

    for index, ((target_id, label, datasets), place) in enumerate(
        zip(target_rows, places, strict=True)
    ):
        refs = tuple(record.dataset_ref for record in datasets.values())
        dataset_ref = refs[0] if refs else f"osm_result_{index + 1}"
        reference_points: list[GeoPoint] = []
        if needs_ref and place is not None:
            reference_points = [GeoPoint(lat=place.latitude, lon=place.longitude)]
        result = compute_indicator(
            IndicatorComputeRequest(
                indicator_id=indicator_id,
                datasets=datasets,
                reference_points=tuple(reference_points),
            )
        )
        computed.append((target_id, label, result))
        plan_targets.append(
            AnalysisTarget(
                target_id=target_id,
                label=label[:80],
                dataset_ref=dataset_ref,
                reference_points=reference_points,
            )
        )
        record = next(iter(datasets.values())) if datasets else None
        metric_status = (
            "computed"
            if result.status == "computed"
            else "insufficient_data"
            if result.status == "insufficient_data"
            else "not_applicable"
        )
        provenance = CalculationProvenance(
            metric=metric,
            dataset_ref=dataset_ref,
            target_ref=target_id,
            implementation_id=spec.implementation_id,
            observation_count=result.observation_count,
            candidate_count=result.observation_count,
            missing_count=result.missing_count,
            invalid_count=0,
            unit=result.unit,
            analysis_area_km2=record.scope.area_km2 if record is not None else None,
            reference_point_count=len(reference_points),
            metric_catalog_version=METRIC_CATALOG_VERSION,
        )
        metric_result = MetricResult(
            metric=metric,
            role="primary",
            label=definition.display_label,
            status=metric_status,
            value=result.value,
            unit=result.unit,
            direction=definition.direction,
            observation_count=result.observation_count,
            candidate_count=result.observation_count,
            missing_count=result.missing_count,
            invalid_count=0,
            provenance=provenance,
            notes=list(result.warnings),
        )
        retrieved = record.feature_count if record is not None else 0
        scope_kind = record.scope.scope_kind if record is not None else "point"
        retrieved_at = (
            record.retrieved_at.isoformat()
            if record is not None
            else datetime.now(tz=timezone.utc).isoformat()
        )
        result_targets.append(
            TargetAnalysisResult(
                target_id=target_id,
                label=label[:80],
                dataset_ref=dataset_ref,
                data_provenance=DataProvenance(
                    feature_concept=definition.display_label[:80],
                    resolved_tags=list(tags),
                    dataset_ref=dataset_ref,
                    target_id=target_id,
                    analysis_scope=record.scope.summary if record is not None else label,
                    scope_kind=scope_kind,
                    analysis_area_km2=record.scope.area_km2 if record is not None else None,
                    retrieved_feature_count=retrieved,
                    valid_observation_count=result.observation_count,
                    missing_observation_count=result.missing_count,
                    invalid_observation_count=0,
                    truncated=record.truncated if record is not None else False,
                    effective_limit=record.effective_limit if record is not None else None,
                    limit_reached=bool(record.truncated) if record is not None else False,
                    retrieved_at=retrieved_at,
                ),
                metrics=[metric_result],
            )
        )

    comparison = compare_indicator_targets(definition, tuple(computed), analysis_type=analysis_type)
    report = assemble_indicator_report(
        definition=definition,
        selection=selection,
        comparison=comparison,
        per_target=tuple((label, result) for _tid, label, result in computed),
        tags=tags,
        extra_warnings=extra_warnings,
    )
    plan = AnalysisPlan(
        analysis_type=analysis_type,
        feature_concept=definition.display_label[:80],
        comparison_goal=selection.inferred_goal.replace("_", " ")[:200],
        targets=plan_targets,
        metrics=[
            MetricRequest(
                metric=metric,
                role="primary",
                inferred_goal=selection.inferred_goal,
                user_explicit=selection.goal_explicit,
            )
        ],
    )
    analysis_result = AnalysisResult(
        analysis_type=analysis_type,
        feature_concept=definition.display_label[:80],
        comparison_goal=plan.comparison_goal,
        targets=result_targets,
        metric_catalog_version=METRIC_CATALOG_VERSION,
        ruleset_version=RULESET_VERSION,
        warnings=list(extra_warnings),
        limitations=list(definition.limitations),
    )
    evidence = [
        MetricSelectionEvidence(
            basis_type=item.basis_type,
            rule_id=item.rule_id,
            statement=item.statement,
            source_ref=item.source_ref,
        )
        for item in selection.selection_evidence
    ]
    metric_trace = MetricSelectionTrace(
        metric=metric,
        role="primary",
        inferred_goal=selection.inferred_goal,
        target_ids=[row[0] for row in target_rows],
        evidence=evidence,
        feasibility_checks=[],
        final_status="executed",
        attempt_index=0,
    )
    decision = AnalysisDecisionTrace(
        inferred_analysis_type=analysis_type,
        inferred_comparison_goal=plan.comparison_goal,
        feature_concept=definition.display_label[:80],
        metric_selections=[metric_trace],
        plan_revision_count=0,
        final_primary_metric=metric,
        final_supporting_metrics=[],
        ruleset_version=RULESET_VERSION,
        metric_catalog_version=METRIC_CATALOG_VERSION,
        selected_indicator_id=definition.indicator_id,
        analysis_domain=",".join(selection.domain_selection.domains),
        candidate_indicators=list(selection.candidate_indicators),
        indicator_selection_reason=selection.selection_reason,
        required_data=list(selection.required_data),
        calculation_method=definition.method_id,
        indicator_catalog_version=INDICATOR_CATALOG_VERSION,
    )
    status = "completed"
    if all(item.status != "computed" for _t, _l, item in computed):
        status = "rejected"
    return AnalysisBlock(
        plan=plan,
        decision_trace=decision,
        result=analysis_result,
        comparison=comparison,
        report=report,
        status=status,
    )
