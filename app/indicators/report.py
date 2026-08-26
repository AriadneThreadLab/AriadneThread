"""User-facing comparison report for catalog indicators."""

from __future__ import annotations

from app.analytics.contracts import ComparisonReport, ComparisonResult, ReportSection
from app.analytics.limitations import limitation_text
from app.indicators.contracts import (
    IndicatorComputationResult,
    IndicatorDefinition,
    IndicatorSelectionTrace,
)


def assemble_indicator_report(
    *,
    definition: IndicatorDefinition,
    selection: IndicatorSelectionTrace,
    comparison: ComparisonResult,
    per_target: tuple[tuple[str, IndicatorComputationResult], ...],
    tags: tuple[str, ...],
    extra_warnings: tuple[str, ...] = (),
) -> ComparisonReport:
    """Compact report: goal, indicator, why, values, ranking, limitations."""
    why = selection.selection_reason
    evidence = [
        item.statement
        for item in selection.selection_evidence
        if item.basis_type in {"semantic_rule", "user_explicit", "metric_catalog"}
    ]
    if evidence:
        why = evidence[0] if why.startswith("Selected ") else why

    value_lines = []
    for label, result in per_target:
        if result.value is None:
            value_lines.append(f"{label}: not computed ({result.status})")
        else:
            value_lines.append(
                f"{label}: {result.value:g} {result.unit} ({result.observation_count} observations)"
            )
    comparison_body = comparison.overall_statement
    if value_lines:
        comparison_body = comparison_body + "\n" + "\n".join(value_lines)

    warning_lines = list(extra_warnings)
    for _label, result in per_target:
        warning_lines.extend(result.warnings)
    limitation_lines = [limitation_text(code) for code in definition.limitations]
    limitation_lines.extend(dict.fromkeys(warning_lines))
    if tags:
        limitation_lines.append(f"OpenStreetMap tags: {', '.join(tags)}.")

    sections = [
        ReportSection(
            key="analysis_goal",
            title="Analysis Goal",
            body=selection.inferred_goal.replace("_", " "),
            origin="deterministic",
        ),
        ReportSection(
            key="selected_indicator",
            title="Selected Indicator",
            body=(
                f"{definition.display_label} ({definition.indicator_id}), "
                f"{definition.method_id}, {definition.unit}"
            ),
            origin="deterministic",
        ),
        ReportSection(
            key="why_this_indicator",
            title="Why This Indicator Was Selected",
            body=why,
            origin="deterministic",
        ),
        ReportSection(
            key="comparison",
            title="Comparison",
            body=comparison_body,
            origin="deterministic",
        ),
        ReportSection(
            key="data_limitations",
            title="Data Limitations",
            body="\n".join(f"- {line}" for line in limitation_lines)
            or "- No additional limitations were selected for this analysis.",
            origin="deterministic",
        ),
    ]
    return ComparisonReport(sections=sections, numeric_fidelity="not_applicable")
