"""Deterministic comparison of analysis results across targets."""

from __future__ import annotations

from app.analytics.catalog import get_metric_definition
from app.analytics.contracts import (
    AnalysisPlan,
    AnalysisResult,
    ComparisonResult,
    ComparisonVerdict,
    MetricComparison,
    MetricDirection,
    MetricResult,
    MetricRole,
    MetricType,
    TargetMetricValue,
)

_NORMATIVE_TOKENS = ("better", "worse", "superior", "best", "worst")


class ComparisonEngine:
    """Pure comparison logic — no model involvement."""

    def compare(self, plan: AnalysisPlan, result: AnalysisResult) -> ComparisonResult:
        primary_comp: MetricComparison | None = None
        supporting: list[MetricComparison] = []

        for metric_req in plan.metrics:
            values = self._collect(result, metric_req.metric, metric_req.role)
            direction = self._direction(metric_req.metric, values, metric_req.direction)
            comparison = self._compare_metric(
                metric=metric_req.metric,
                role=metric_req.role,
                direction=direction,
                values=values,
            )
            if metric_req.role == "primary":
                primary_comp = comparison
            else:
                supporting.append(comparison)

        statement, confidence = self._overall(
            plan,
            primary_comp,
            truncated=any(
                target.data_provenance.truncated or target.data_provenance.limit_reached
                for target in result.targets
            ),
        )
        return ComparisonResult(
            analysis_type=plan.analysis_type,
            primary=primary_comp,
            supporting=supporting,
            overall_statement=statement,
            overall_confidence=confidence,
        )

    def _collect(
        self,
        result: AnalysisResult,
        metric: MetricType,
        role: MetricRole,
    ) -> list[tuple[str, str, MetricResult]]:
        collected: list[tuple[str, str, MetricResult]] = []
        for target in result.targets:
            for metric_result in target.metrics:
                if metric_result.metric == metric and metric_result.role == role:
                    collected.append((target.target_id, target.label, metric_result))
        return collected

    def _direction(
        self,
        metric: MetricType,
        values: list[tuple[str, str, MetricResult]],
        override: MetricDirection | None,
    ) -> MetricDirection:
        if values:
            return values[0][2].direction
        definition = get_metric_definition(metric)
        return override or definition.default_direction

    def _compare_metric(
        self,
        *,
        metric: MetricType,
        role: MetricRole,
        direction: MetricDirection,
        values: list[tuple[str, str, MetricResult]],
    ) -> MetricComparison:
        definition = get_metric_definition(metric)
        ranking_values = [
            TargetMetricValue(
                target_id=tid,
                label=label,
                value=mr.value,
                status=mr.status,
                observation_count=mr.observation_count,
            )
            for tid, label, mr in values
        ]
        computed = [v for v in ranking_values if v.status == "computed" and v.value is not None]
        rest = [v for v in ranking_values if v not in computed]
        computed_sorted = sorted(computed, key=lambda v: float(v.value or 0.0), reverse=True)
        ranking = computed_sorted + rest

        unit = values[0][2].unit if values else definition.unit
        if len(computed_sorted) < 2:
            return MetricComparison(
                metric=metric,
                role=role,
                label=definition.display_label,
                unit=unit,
                direction=direction,
                ranking=ranking,
                verdict="insufficient_data",
                difference_unit="",
            )

        a = computed_sorted[0]
        b = computed_sorted[1]
        assert a.value is not None and b.value is not None
        abs_diff = abs(a.value - b.value)
        eps = definition.tie_relative_epsilon
        tied = abs_diff <= max(0.0, eps * max(abs(a.value), abs(b.value)))

        rel: float | None = None
        difference_unit = unit
        if metric == "coverage_percentage":
            difference_unit = "percentage_points"
            rel = None
        else:
            smaller = min(abs(a.value), abs(b.value))
            if smaller > 0:
                rel = 100.0 * abs_diff / smaller

        if tied:
            verdict: ComparisonVerdict = "tie"
            preferred = None
            leading = a.target_id
        else:
            leading = a.target_id
            if direction == "higher_is_better":
                preferred = a.target_id
                verdict = "higher"
            elif direction == "lower_is_better":
                preferred = computed_sorted[-1].target_id
                # Re-rank ascending for lower-is-better presentation of verdict.
                low = min(computed_sorted, key=lambda v: float(v.value or 0.0))
                preferred = low.target_id
                verdict = "lower"
            else:
                preferred = None
                verdict = "higher"

        return MetricComparison(
            metric=metric,
            role=role,
            label=definition.display_label,
            unit=unit,
            direction=direction,
            ranking=ranking,
            verdict=verdict,
            leading_target_id=leading,
            preferred_target_id=preferred,
            absolute_difference=abs_diff,
            relative_difference_percent=rel,
            difference_unit=difference_unit,
        )

    def _overall(
        self,
        plan: AnalysisPlan,
        primary: MetricComparison | None,
        *,
        truncated: bool = False,
    ) -> tuple[str, str]:
        if primary is None or primary.verdict == "insufficient_data":
            return (
                "Insufficient comparable data to rank the targets on the primary metric.",
                "insufficient_data",
            )

        labels = {v.target_id: v.label for v in primary.ranking}
        if primary.verdict == "tie":
            return (
                f"Targets are effectively tied on {primary.label} ({primary.unit}).",
                "close",
            )

        confidence = "clear"
        if (
            primary.relative_difference_percent is not None
            and primary.relative_difference_percent < 10.0
        ):
            confidence = "close"

        if primary.direction == "neutral":
            if primary.metric == "standard_deviation":
                # Lower sd → more consistent (descriptive, not better/worse).
                low = min(
                    (v for v in primary.ranking if v.value is not None),
                    key=lambda v: float(v.value or 0.0),
                    default=None,
                )
                high = max(
                    (v for v in primary.ranking if v.value is not None),
                    key=lambda v: float(v.value or 0.0),
                    default=None,
                )
                if low and high and low.target_id != high.target_id:
                    statement = (
                        f"{low.label}'s values are more tightly clustered "
                        f"({primary.label}: {low.value:g} vs {high.value:g} {primary.unit})."
                    )
                else:
                    statement = f"Targets differ on {primary.label}."
            else:
                a = primary.ranking[0]
                b = primary.ranking[1] if len(primary.ranking) > 1 else None
                if a.value is not None and b and b.value is not None:
                    statement = (
                        f"{a.label} differs from {b.label} on {primary.label} "
                        f"({a.value:g} vs {b.value:g} {primary.unit})."
                    )
                else:
                    statement = f"Targets differ on {primary.label}."
            for token in _NORMATIVE_TOKENS:
                assert token not in statement.lower()
            return statement, confidence

        preferred_id = primary.preferred_target_id
        preferred_label = labels.get(preferred_id or "", "A target")
        others = [v for v in primary.ranking if v.target_id != preferred_id and v.value is not None]
        other = others[0] if others else None
        pref_val = next(
            (v.value for v in primary.ranking if v.target_id == preferred_id),
            None,
        )
        if other is None or pref_val is None or other.value is None:
            return f"{preferred_label} leads on {primary.label}.", confidence

        direction_word = "lower" if primary.direction == "lower_is_better" else "higher"

        rel_part = ""
        if primary.relative_difference_percent is not None and not truncated:
            rel_part = f", {primary.relative_difference_percent:.1f}% difference"
        statement = (
            f"{preferred_label} has a {direction_word} {primary.label} "
            f"({pref_val:g} vs {other.value:g} {primary.unit}{rel_part})."
        )
        if truncated:
            statement += (
                " At least one target reached the feature retrieval limit, so these "
                "counts are lower bounds and not an exhaustive abundance comparison."
            )
            confidence = "close"
        return statement, confidence
