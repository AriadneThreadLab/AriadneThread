"""analyze_features tool through the Tool Registry."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from app.analytics.contracts import AnalysisPlan, AnalysisTarget, MetricRequest
from app.analytics.datasets import DatasetRegistry
from app.analytics.factory import build_analyze_features_tool
from app.llm.contracts import ToolCall
from app.osm.query_spec import OsmFeatureQuery, PointRadius, TagFilter
from app.places.contracts import PlaceRegistry
from app.tools.context import AnalysisRunState, GroundingState, ToolContext
from app.tools.factory import build_tool_registry
from app.tools.query_osm import OsmQuerySource, QueryOsmResult


def _ctx(message: str = "Compare parks at 35.7 51.4 and 35.8 51.5") -> ToolContext:
    return ToolContext(
        datasets=DatasetRegistry(),
        analysis=AnalysisRunState(),
        user_message=message,
        places=PlaceRegistry(),
        grounding=GroundingState(),
    )


def _seed(ctx: ToolContext, *, lat: float, lon: float) -> str:
    query = OsmFeatureQuery(
        point=PointRadius(lat=lat, lon=lon, radius_m=2000),
        tags=[TagFilter(key="leisure", value="park")],
        limit=20,
    )
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon + 0.001, lat + 0.001]},
                "properties": {"tags": {"leisure": "park"}},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon + 0.002, lat]},
                "properties": {"tags": {"leisure": "park"}},
            },
        ],
    }
    result = QueryOsmResult(
        feature_count=2,
        geojson=geojson,
        overpass_query="out;",
        source=OsmQuerySource(
            endpoint="https://overpass.test",
            retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        effective_limit=20,
        scope_summary=f"Analysis area: 2000 m around {lat}, {lon}",
    )
    return ctx.datasets.register(result, query)


async def test_registry_rejects_unknown_metric() -> None:
    tool = build_analyze_features_tool()
    registry = build_tool_registry(analyze_features_tool=tool)
    invocation = await registry.invoke(
        ToolCall(
            id="c1",
            name="analyze_features",
            arguments={
                "analysis_type": "single_target",
                "feature_concept": "parks",
                "comparison_goal": "Custom score",
                "targets": [{"target_id": "a", "label": "A", "dataset_ref": "osm_result_1"}],
                "metrics": [
                    {
                        "metric": "magic_score",
                        "role": "primary",
                        "inferred_goal": "abundance",
                    }
                ],
            },
        ),
        _ctx(),
    )
    assert not invocation.ok
    assert invocation.error_code == "tool_argument_error"


async def test_count_comparison_completes() -> None:
    tool = build_analyze_features_tool()
    ctx = _ctx()
    ref_a = _seed(ctx, lat=35.7, lon=51.4)
    ref_b = _seed(ctx, lat=35.8, lon=51.5)
    plan = AnalysisPlan(
        analysis_type="comparison",
        feature_concept="public parks",
        comparison_goal="Which area has more parks",
        targets=[
            AnalysisTarget(target_id="a", label="Area A", dataset_ref=ref_a),
            AnalysisTarget(target_id="b", label="Area B", dataset_ref=ref_b),
        ],
        metrics=[
            MetricRequest(metric="count", role="primary", inferred_goal="abundance"),
        ],
    )
    outcome = await tool.execute(plan, ctx)
    assert outcome.payload.status == "completed"
    assert outcome.payload.result is not None
    assert outcome.payload.comparison is not None
    assert outcome.payload.decision_trace.final_primary_metric == "count"
    assert "status=completed" in outcome.observation
    assert "FeatureCollection" not in outcome.observation
    assert '"coordinates"' not in outcome.observation
    for target in outcome.payload.result.targets:
        for metric in target.metrics:
            assert metric.provenance.implementation_id.startswith("SpatialAnalyticsEngine.")
            assert metric.provenance.metric_catalog_version == "metric-catalog-1"


async def test_rejection_then_abandon_path() -> None:
    tool = build_analyze_features_tool()
    ctx = _ctx()
    ref = _seed(ctx, lat=35.7, lon=51.4)
    bad = AnalysisPlan(
        analysis_type="single_target",
        feature_concept="parks",
        comparison_goal="Mean capacity",
        targets=[AnalysisTarget(target_id="a", label="A", dataset_ref=ref)],
        metrics=[
            MetricRequest(
                metric="mean",
                role="primary",
                inferred_goal="typical_value",
                property_key="capacity",
                user_explicit=True,
            )
        ],
    )
    first = await tool.execute(bad, ctx)
    assert first.payload.status == "rejected"
    assert ctx.analysis.plan_revision_count == 1
    second = await tool.execute(bad, ctx)
    assert second.payload.status in {"rejected", "abandoned"}
    # Third call after budget should abandon.
    ctx.analysis.plan_revision_count = 2
    third = await tool.execute(bad, ctx)
    assert third.payload.status == "abandoned"


async def test_untrusted_reference_points_rejected() -> None:
    tool = build_analyze_features_tool()
    ctx = _ctx(message="Compare parks near my house")  # no coordinates
    ref = _seed(ctx, lat=35.7, lon=51.4)
    plan = AnalysisPlan(
        analysis_type="single_target",
        feature_concept="parks",
        comparison_goal="Nearest park",
        targets=[
            AnalysisTarget(
                target_id="a",
                label="A",
                dataset_ref=ref,
                reference_points=[{"lat": 35.7, "lon": 51.4}],
            )
        ],
        metrics=[
            MetricRequest(
                metric="nearest_distance",
                role="primary",
                inferred_goal="accessibility",
            )
        ],
    )
    with pytest.raises(Exception, match="reference_points|inventing"):
        await tool.execute(plan, ctx)
