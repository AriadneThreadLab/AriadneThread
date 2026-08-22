"""Deterministic comparison report assembly and numeric fidelity guard."""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.analytics.catalog import METRIC_CATALOG_VERSION, get_metric_definition
from app.analytics.contracts import (
    AnalysisDecisionTrace,
    AnalysisPlan,
    AnalysisResult,
    ComparisonReport,
    ComparisonResult,
    ReportSection,
)
from app.analytics.limitations import limitation_text
from app.analytics.rules import RULESET_VERSION

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")


class ReportAssembler:
    """Assemble deterministic report sections after analytics complete."""

    def assemble(
        self,
        plan: AnalysisPlan,
        decision_trace: AnalysisDecisionTrace,
        result: AnalysisResult,
        comparison: ComparisonResult,
        *,
        narrative_interpretation: str | None = None,
        narrative_conclusion: str | None = None,
    ) -> ComparisonReport:
        primary = next(m for m in plan.metrics if m.role == "primary")
        definition = get_metric_definition(primary.metric)
        rule_ids = [
            e.rule_id
            for sel in decision_trace.metric_selections
            if sel.final_status == "executed" and sel.metric == primary.metric
            for e in sel.evidence
            if e.rule_id is not None
        ]
        why_lines = [
            e.statement
            for sel in decision_trace.metric_selections
            if sel.final_status == "executed" and sel.metric == primary.metric
            for e in sel.evidence
            if e.basis_type in {"semantic_rule", "statistical_guidance", "user_explicit"}
        ]
        scopes = "; ".join(f"{t.label}: {t.data_provenance.analysis_scope}" for t in result.targets)
        tags = ", ".join(result.targets[0].data_provenance.resolved_tags) if result.targets else ""
        obs_lines = []
        for target in result.targets:
            primary_result = next(
                (m for m in target.metrics if m.role == "primary"),
                None,
            )
            if primary_result is None:
                continue
            obs_lines.append(
                f"{target.label}: {target.data_provenance.retrieved_feature_count} features "
                f"({primary_result.observation_count} observations, "
                f"{primary_result.missing_count} missing)"
            )

        fallback_line = ""
        rejected = [s for s in decision_trace.metric_selections if s.final_status == "rejected"]
        if rejected and decision_trace.plan_revision_count > 0:
            first = rejected[0]
            fallback_line = (
                f"Originally proposed {first.metric} was rejected "
                f"({first.rejection_reason}). "
                f"A supported metric was selected after re-planning."
            )

        supporting_lines = []
        for comp in comparison.supporting:
            parts = [
                f"{v.label}={v.value:g}" if v.value is not None else f"{v.label}=n/a"
                for v in comp.ranking
            ]
            supporting_lines.append(f"{comp.label}: " + ", ".join(parts))

        limitation_lines = [limitation_text(code) for code in result.limitations]

        sections = [
            ReportSection(
                key="analysis_goal",
                title="Analysis Goal",
                body=plan.comparison_goal,
                origin="deterministic",
            ),
            ReportSection(
                key="selected_indicator",
                title="Selected Indicator",
                body=(
                    f"{definition.display_label} ({definition.unit}, "
                    f"{definition.default_direction.replace('_', ' ')})"
                ),
                origin="deterministic",
            ),
            ReportSection(
                key="why_this_indicator",
                title="Why This Indicator Was Selected",
                body="\n".join(why_lines) or "Selected from the bounded metric catalog.",
                origin="deterministic",
            ),
            ReportSection(
                key="analysis_method",
                title="Analysis Method",
                body=(
                    f"Scope: {scopes}\n"
                    f"Data: OpenStreetMap — {tags}\n"
                    f"Observations:\n"
                    + "\n".join(obs_lines)
                    + f"\nSelection rules: {', '.join(rule_ids) or 'none'}\n"
                    f"Catalog: {METRIC_CATALOG_VERSION}; Ruleset: {RULESET_VERSION}"
                    + (f"\n{fallback_line}" if fallback_line else "")
                ),
                origin="deterministic",
            ),
            ReportSection(
                key="comparison",
                title="Comparison",
                body=comparison.overall_statement,
                origin="deterministic",
            ),
            ReportSection(
                key="supporting_indicators",
                title="Supporting Indicators",
                body="\n".join(supporting_lines) or "None",
                origin="deterministic",
            ),
            ReportSection(
                key="interpretation",
                title="Interpretation",
                body=narrative_interpretation
                or "See the comparison statement for the deterministic ranking.",
                origin="model_narrative" if narrative_interpretation else "deterministic",
            ),
            ReportSection(
                key="data_limitations",
                title="Data Limitations",
                body="\n".join(f"- {line}" for line in limitation_lines)
                or "- No additional limitations were selected for this analysis.",
                origin="deterministic",
            ),
            ReportSection(
                key="conclusion",
                title="Conclusion",
                body=narrative_conclusion or comparison.overall_statement,
                origin="model_narrative" if narrative_conclusion else "deterministic",
            ),
        ]

        fidelity: str = "not_applicable"
        if narrative_interpretation or narrative_conclusion:
            fidelity = "verified"

        report = ComparisonReport(sections=sections, numeric_fidelity=fidelity)
        if narrative_interpretation or narrative_conclusion:
            report = verify_numeric_fidelity(
                report,
                result,
                comparison,
                narratives=[n for n in (narrative_interpretation, narrative_conclusion) if n],
            )
        return report


def verify_numeric_fidelity(
    report: ComparisonReport,
    result: AnalysisResult,
    comparison: ComparisonResult,
    narratives: Iterable[str],
) -> ComparisonReport:
    """Replace narrative sections if they contain unverified numeric tokens."""
    allowed = _allowed_numbers(result, comparison)
    unmatched: list[str] = []
    for narrative in narratives:
        for token in _NUMBER_RE.findall(narrative):
            if not _token_allowed(token, allowed):
                unmatched.append(token)

    if not unmatched:
        return report.model_copy(update={"numeric_fidelity": "verified"})

    fallback = (
        "The model narrative was withheld because it contained values that do not "
        "match the computed results."
    )
    sections = []
    for section in report.sections:
        if section.key in {"interpretation", "conclusion"} and section.origin == "model_narrative":
            sections.append(
                section.model_copy(update={"body": fallback, "origin": "deterministic"})
            )
        else:
            sections.append(section)
    return ComparisonReport(
        title=report.title,
        sections=sections,
        numeric_fidelity="unverified_numbers_removed",
    )


def _allowed_numbers(
    result: AnalysisResult,
    comparison: ComparisonResult,
) -> set[float]:
    allowed: set[float] = set()
    for target in result.targets:
        for metric in target.metrics:
            if metric.value is not None:
                allowed.add(float(metric.value))
                definition = get_metric_definition(metric.metric)
                allowed.add(round(float(metric.value), definition.display_precision))
            allowed.add(float(metric.observation_count))
            allowed.add(float(metric.candidate_count))
            allowed.add(float(metric.missing_count))
            allowed.add(float(metric.invalid_count))
        allowed.add(float(target.data_provenance.retrieved_feature_count))
    if comparison.primary is not None:
        if comparison.primary.absolute_difference is not None:
            allowed.add(float(comparison.primary.absolute_difference))
        if comparison.primary.relative_difference_percent is not None:
            allowed.add(float(comparison.primary.relative_difference_percent))
            allowed.add(round(float(comparison.primary.relative_difference_percent), 1))
    for i in range(0, len(result.targets) + 1):
        allowed.add(float(i))
    return allowed


def _token_allowed(token: str, allowed: set[float]) -> bool:
    cleaned = token.replace(",", "").rstrip("%")
    try:
        value = float(cleaned)
    except ValueError:
        return False
    if value in allowed:
        return True
    return any(abs(candidate - value) <= 0.05 for candidate in allowed)
