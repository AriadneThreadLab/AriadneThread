"""Security boundaries for the analytics package."""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_IMPORTS = {
    "os",
    "subprocess",
    "socket",
    "pathlib",
    "httpx",
    "sqlalchemy",
    "asyncio.subprocess",
}
FORBIDDEN_TOKENS = {"eval(", "exec(", "__import__("}


def test_analytics_package_has_no_io_or_eval() -> None:
    root = Path(__file__).resolve().parents[1] / "app" / "analytics"
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            assert token not in source, f"{path.name} contains {token}"
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in FORBIDDEN_IMPORTS
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in FORBIDDEN_IMPORTS


def test_analysis_block_serialization_has_no_think() -> None:
    from app.analytics.contracts import (
        AnalysisBlock,
        AnalysisDecisionTrace,
        AnalysisPlan,
        AnalysisTarget,
        MetricRequest,
        MetricSelectionTrace,
    )

    block = AnalysisBlock(
        plan=AnalysisPlan(
            analysis_type="single_target",
            feature_concept="parks",
            comparison_goal="Count parks",
            targets=[
                AnalysisTarget(target_id="a", label="A", dataset_ref="osm_result_1"),
            ],
            metrics=[
                MetricRequest(metric="count", role="primary", inferred_goal="abundance"),
            ],
        ),
        decision_trace=AnalysisDecisionTrace(
            inferred_analysis_type="single_target",
            inferred_comparison_goal="Count parks",
            feature_concept="parks",
            metric_selections=[
                MetricSelectionTrace(
                    metric="count",
                    role="primary",
                    inferred_goal="abundance",
                    target_ids=["a"],
                    evidence=[],
                    feasibility_checks=[],
                    final_status="executed",
                    attempt_index=0,
                )
            ],
            plan_revision_count=0,
            final_primary_metric="count",
            final_supporting_metrics=[],
            ruleset_version="metric-rules-1",
            metric_catalog_version="metric-catalog-1",
        ),
        status="completed",
    )
    payload = block.model_dump_json()
    assert "<think" not in payload.lower()
    assert "FeatureCollection" not in payload
