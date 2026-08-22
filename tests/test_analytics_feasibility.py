"""Metric feasibility validation and rejection traces."""

from __future__ import annotations

from datetime import datetime, timezone

from app.analytics.contracts import (
    AnalysisPlan,
    AnalysisTarget,
    MetricRequest,
)
from app.analytics.datasets import DatasetRegistry
from app.analytics.feasibility import MetricFeasibilityValidator
from app.osm.query_spec import OsmFeatureQuery, PointRadius, TagFilter
from app.tools.query_osm import OsmQuerySource, QueryOsmResult


def _fc(n: int = 2, *, with_capacity: bool = False) -> dict[str, object]:
    features = []
    for i in range(n):
        tags: dict[str, str] = {"leisure": "park"}
        if with_capacity:
            tags["capacity"] = str(i + 1)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [51.3 + i * 0.01, 35.7],
                            [51.31 + i * 0.01, 35.7],
                            [51.31 + i * 0.01, 35.71],
                            [51.3 + i * 0.01, 35.71],
                            [51.3 + i * 0.01, 35.7],
                        ]
                    ],
                },
                "properties": {"tags": tags},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _register_point(
    registry: DatasetRegistry,
    *,
    lat: float,
    lon: float,
    n: int = 3,
    with_capacity: bool = False,
) -> str:
    query = OsmFeatureQuery(
        point=PointRadius(lat=lat, lon=lon, radius_m=2000),
        tags=[TagFilter(key="leisure", value="park")],
        limit=50,
    )
    result = QueryOsmResult(
        feature_count=n,
        geojson=_fc(n, with_capacity=with_capacity),
        overpass_query="out geom;",
        source=OsmQuerySource(
            endpoint="https://overpass.test",
            retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        effective_limit=50,
        scope_summary=f"Analysis area: 2000 m around {lat}, {lon}",
    )
    return registry.register(result, query)


def test_mean_capacity_rejected_when_missing() -> None:
    registry = DatasetRegistry()
    ref = _register_point(registry, lat=35.7, lon=51.4, with_capacity=False)
    plan = AnalysisPlan(
        analysis_type="single_target",
        feature_concept="parks",
        comparison_goal="Mean park capacity",
        targets=[
            AnalysisTarget(target_id="a", label="A", dataset_ref=ref),
        ],
        metrics=[
            MetricRequest(
                metric="mean",
                role="primary",
                inferred_goal="typical_value",
                property_key="capacity",
                user_explicit=True,
            ),
        ],
    )
    outcome = MetricFeasibilityValidator().validate(
        plan, registry, attempt_index=0, plan_revision_count=0
    )
    assert not outcome.ok
    trace = next(t for t in outcome.traces if t.metric == "mean")
    assert trace.final_status == "rejected"
    assert trace.rejection_reason == "numeric_property_unavailable"
    assert any(c.status == "failed" for c in trace.feasibility_checks)
    assert "count" in trace.supported_alternatives or "total_area" in trace.supported_alternatives


def test_place_scope_density_rejected() -> None:
    registry = DatasetRegistry()
    query = OsmFeatureQuery(
        place="Tehran",
        tags=[TagFilter(key="leisure", value="park")],
        limit=50,
    )
    result = QueryOsmResult(
        feature_count=2,
        geojson=_fc(2),
        overpass_query="out geom;",
        source=OsmQuerySource(
            endpoint="https://overpass.test",
            retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        effective_limit=50,
        scope_summary="Search area: Tehran",
    )
    ref = registry.register(result, query)
    plan = AnalysisPlan(
        analysis_type="single_target",
        feature_concept="parks",
        comparison_goal="Park density",
        targets=[AnalysisTarget(target_id="a", label="A", dataset_ref=ref)],
        metrics=[
            MetricRequest(metric="density", role="primary", inferred_goal="concentration"),
        ],
    )
    outcome = MetricFeasibilityValidator().validate(
        plan, registry, attempt_index=0, plan_revision_count=0
    )
    assert not outcome.ok
    assert outcome.rejection_reason == "analysis_area_unavailable"


def test_median_nearest_needs_two_reference_points() -> None:
    registry = DatasetRegistry()
    ref = _register_point(registry, lat=35.7, lon=51.4)
    plan = AnalysisPlan(
        analysis_type="single_target",
        feature_concept="parks",
        comparison_goal="Typical access",
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
                metric="median_nearest_distance",
                role="primary",
                inferred_goal="accessibility",
            ),
        ],
    )
    outcome = MetricFeasibilityValidator().validate(
        plan, registry, attempt_index=0, plan_revision_count=0
    )
    assert not outcome.ok
    assert outcome.rejection_reason in {
        "reference_geometry_unavailable",
        "insufficient_observations",
    }
    assert "nearest_distance" in outcome.supported_alternatives
