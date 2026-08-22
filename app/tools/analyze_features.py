"""``analyze_features`` — deterministic spatial analytics over request-scoped datasets."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.spatial_trust import coordinates_supported_by_user
from app.analytics.comparison import ComparisonEngine
from app.analytics.contracts import (
    MAX_ANALYSIS_REPLANS,
    AnalysisDecisionTrace,
    AnalysisPlan,
    AnalysisResult,
    ComparisonReport,
    ComparisonResult,
    MetricSelectionTrace,
    MetricType,
)
from app.analytics.engine import SpatialAnalyticsEngine
from app.analytics.feasibility import MetricFeasibilityValidator
from app.analytics.report import ReportAssembler
from app.core.errors import ToolArgumentError
from app.places.contracts import PlaceRegistry
from app.tools.context import ToolContext
from app.tools.contracts import ToolOutcome

TOOL_NAME = "analyze_features"
TOOL_DESCRIPTION = (
    "Deterministic statistics/comparisons over datasets already retrieved with "
    "query_osm in THIS request (dataset_ref like osm_result_1). Use ONLY for "
    "analytical questions after live data exists. Do NOT use for ordinary "
    "find/list/search requests. Never invent numbers or fetch new map data."
)

AnalyzeFeaturesArgs = AnalysisPlan


class AnalyzeFeaturesResult(BaseModel):
    """Structured analytics payload kept for the API/UI (not the LLM observation)."""

    model_config = ConfigDict(frozen=True)

    status: Literal["completed", "rejected", "abandoned"]
    plan: AnalysisPlan
    decision_trace: AnalysisDecisionTrace
    result: AnalysisResult | None = None
    comparison: ComparisonResult | None = None
    report: ComparisonReport | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def analysis_status(self) -> str:
        return self.status

    @property
    def metrics_computed(self) -> int:
        if self.result is None:
            return 0
        return sum(
            1
            for target in self.result.targets
            for metric in target.metrics
            if metric.status == "computed"
        )


class AnalyzeFeaturesTool:
    """Tool implementation over the deterministic analytics stack."""

    def __init__(
        self,
        engine: SpatialAnalyticsEngine,
        validator: MetricFeasibilityValidator,
        comparison: ComparisonEngine,
        reporter: ReportAssembler,
    ) -> None:
        self._engine = engine
        self._validator = validator
        self._comparison = comparison
        self._reporter = reporter

    @property
    def name(self) -> str:
        return TOOL_NAME

    @property
    def description(self) -> str:
        return TOOL_DESCRIPTION

    @property
    def args_model(self) -> type[AnalysisPlan]:
        return AnalyzeFeaturesArgs

    async def execute(
        self,
        args: AnalysisPlan,
        context: ToolContext,
    ) -> ToolOutcome[AnalyzeFeaturesResult]:
        if not context.datasets.refs():
            raise ToolArgumentError(
                "analyze_features requires at least one prior query_osm result "
                "in this request (dataset_ref such as osm_result_1)"
            )

        self._assert_trusted_reference_points(args, context)

        attempt_index = context.analysis.plan_revision_count
        if attempt_index > MAX_ANALYSIS_REPLANS:
            from app.analytics.catalog import METRIC_CATALOG_VERSION
            from app.analytics.rules import RULESET_VERSION

            decision = AnalysisDecisionTrace(
                inferred_analysis_type=args.analysis_type,
                inferred_comparison_goal=args.comparison_goal,
                feature_concept=args.feature_concept,
                metric_selections=list(context.analysis.rejected_traces),
                plan_revision_count=min(
                    context.analysis.plan_revision_count,
                    MAX_ANALYSIS_REPLANS + 1,
                ),
                final_primary_metric=None,
                final_supporting_metrics=[],
                ruleset_version=RULESET_VERSION,
                metric_catalog_version=METRIC_CATALOG_VERSION,
            )
            result = AnalyzeFeaturesResult(
                status="abandoned",
                plan=args,
                decision_trace=decision,
                warnings=["analytical re-plan budget exhausted"],
            )
            return ToolOutcome(
                observation=(
                    "status=abandoned reason=replan_budget_exhausted "
                    "Answer without analytics and state what could not be computed. "
                    "Do not invent metric values."
                ),
                payload=result,
            )

        outcome = self._validator.validate(
            args,
            context.datasets,
            attempt_index=attempt_index,
            plan_revision_count=context.analysis.plan_revision_count,
            prior_rejected=list(context.analysis.rejected_traces),
        )

        if not outcome.ok:
            rejected = [t for t in outcome.traces if t.final_status == "rejected"]
            context.analysis.rejected_traces.extend(
                t for t in rejected if t.attempt_index == attempt_index
            )
            context.analysis.plan_revision_count += 1
            status: Literal["rejected", "abandoned"] = (
                "abandoned"
                if context.analysis.plan_revision_count > MAX_ANALYSIS_REPLANS
                else "rejected"
            )
            remaining = max(0, MAX_ANALYSIS_REPLANS - context.analysis.plan_revision_count + 1)
            if status == "abandoned":
                remaining = 0
            feedback = outcome.feedback + f" replans_remaining={remaining}"
            if status == "abandoned":
                feedback = (
                    "status=abandoned "
                    + outcome.feedback
                    + " Answer without analytics and state what could not be computed."
                )
            # Link replacement on prior rejected when a later plan succeeds — handled
            # on the success path. Here just return rejection.
            payload = AnalyzeFeaturesResult(
                status=status,
                plan=args,
                decision_trace=outcome.decision_trace.model_copy(
                    update={"plan_revision_count": context.analysis.plan_revision_count}
                ),
                warnings=["analysis plan failed feasibility validation"],
            )
            observation = self._safe_observation(feedback)
            return ToolOutcome(observation=observation, payload=payload)

        # Mark prior rejected traces as superseded with replacement when applicable.
        primary = next(m.metric for m in args.metrics if m.role == "primary")
        self._link_replacements(context.analysis.rejected_traces, primary)

        analysis_result = self._engine.compute_plan(
            args,
            context.datasets,
            grounding_sources=[],
        )
        comparison = self._comparison.compare(args, analysis_result)
        report = self._reporter.assemble(
            args,
            outcome.decision_trace,
            analysis_result,
            comparison,
        )

        executed = [t for t in outcome.traces if t.final_status == "executed"]
        context.analysis.accepted_traces.extend(executed)
        context.analysis.last_plan_primary = primary

        decision = outcome.decision_trace.model_copy(
            update={
                "metric_selections": list(context.analysis.rejected_traces) + executed,
                "plan_revision_count": context.analysis.plan_revision_count,
            }
        )

        payload = AnalyzeFeaturesResult(
            status="completed",
            plan=args,
            decision_trace=decision,
            result=analysis_result,
            comparison=comparison,
            report=report,
            warnings=list(analysis_result.warnings),
        )
        return ToolOutcome(
            observation=self._observation(payload),
            payload=payload,
        )

    def _assert_trusted_reference_points(self, plan: AnalysisPlan, context: ToolContext) -> None:
        for target in plan.targets:
            for point in target.reference_points:
                if coordinates_supported_by_user(point.lat, point.lon, context.user_message):
                    continue
                if _matches_registered_place(point.lat, point.lon, context.places):
                    continue
                raise ToolArgumentError(
                    "reference_points require latitude and longitude explicitly "
                    "provided by the user or a trusted place_ref from resolve_place; "
                    "inventing coordinates is not allowed."
                )

    def _link_replacements(
        self,
        rejected: list[MetricSelectionTrace],
        final_primary: MetricType,
    ) -> None:
        for index, trace in enumerate(rejected):
            if trace.replacement_metric is None and trace.role == "primary":
                rejected[index] = trace.model_copy(
                    update={
                        "replacement_metric": final_primary,
                        "final_status": "superseded",
                    }
                )

    def _observation(self, payload: AnalyzeFeaturesResult) -> str:
        assert payload.result is not None and payload.comparison is not None
        primary = payload.comparison.primary
        target_bits = []
        if primary is not None:
            for item in primary.ranking:
                value = "n/a" if item.value is None else f"{item.value:g}"
                target_bits.append(f"{item.target_id}={value} (n={item.observation_count})")
        supporting_bits = []
        for comp in payload.comparison.supporting:
            parts = [
                f"{v.target_id}={v.value:g}" if v.value is not None else f"{v.target_id}=n/a"
                for v in comp.ranking
            ]
            supporting_bits.append(f"{comp.metric} " + " ".join(parts))
        rule_ids = sorted(
            {
                e.rule_id
                for sel in payload.decision_trace.metric_selections
                if sel.final_status in {"executed", "superseded"}
                for e in sel.evidence
                if e.rule_id
            }
        )
        limitations = ",".join(payload.result.limitations[:6])
        parts = [
            f"status=completed analysis={payload.plan.analysis_type} "
            f'concept="{payload.plan.feature_concept}" targets={len(payload.plan.targets)}',
        ]
        if primary is not None:
            parts.append(
                f"primary={primary.metric} unit={primary.unit} direction={primary.direction}"
            )
            parts.append(" ".join(target_bits))
            parts.append(
                f"verdict={primary.preferred_target_id or primary.leading_target_id or 'none'} "
                f"{primary.verdict}"
                + (
                    f" diff={primary.absolute_difference:g}{primary.unit}"
                    if primary.absolute_difference is not None
                    else ""
                )
                + (
                    f" rel={primary.relative_difference_percent:.1f}%"
                    if primary.relative_difference_percent is not None
                    else ""
                )
            )
        if supporting_bits:
            parts.append("supporting: " + " | ".join(supporting_bits))
        parts.append("rules=" + (",".join(rule_ids) if rule_ids else "none"))
        parts.append(f"limitations={limitations or 'none'}")
        parts.append(
            "Use these exact numbers. Do not recalculate, round differently, or add values."
        )
        return self._safe_observation(" ".join(parts))

    @staticmethod
    def _safe_observation(text: str) -> str:
        if "FeatureCollection" in text or '"coordinates"' in text:
            raise ToolArgumentError("internal error: GeoJSON leaked into observation")
        return text[:1200]


def _matches_registered_place(lat: float, lon: float, places: PlaceRegistry) -> bool:
    for ref in places.refs():
        record = places.get(ref)
        if record is None:
            continue
        if abs(record.latitude - lat) <= 1e-6 and abs(record.longitude - lon) <= 1e-6:
            return True
    return False
