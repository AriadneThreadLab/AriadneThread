"""Request-scoped dataset registry."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from app.analytics.datasets import DatasetRegistry, UnknownDatasetError
from app.osm.query_spec import OsmFeatureQuery, PointRadius, TagFilter
from app.tools.query_osm import OsmQuerySource, QueryOsmResult


def _result(n: int = 1) -> QueryOsmResult:
    return QueryOsmResult(
        feature_count=n,
        geojson={"type": "FeatureCollection", "features": []},
        overpass_query="out;",
        source=OsmQuerySource(
            endpoint="https://overpass.test",
            retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        effective_limit=20,
        scope_summary="Analysis area: 1000 m around 35.7, 51.4",
    )


def test_sequential_refs_and_isolation() -> None:
    a = DatasetRegistry()
    b = DatasetRegistry()
    query = OsmFeatureQuery(
        point=PointRadius(lat=35.7, lon=51.4, radius_m=1000),
        tags=[TagFilter(key="leisure", value="park")],
    )
    assert a.register(_result(), query) == "osm_result_1"
    assert a.register(_result(), query) == "osm_result_2"
    assert b.register(_result(), query) == "osm_result_1"
    assert a.get("osm_result_1").scope.area_km2 is not None
    with pytest.raises(UnknownDatasetError):
        a.get("osm_result_9")


def test_place_scope_has_no_area() -> None:
    registry = DatasetRegistry()
    query = OsmFeatureQuery(
        place="Tehran",
        tags=[TagFilter(key="leisure", value="park")],
    )
    ref = registry.register(
        _result().model_copy(update={"scope_summary": "Search area: Tehran"}),
        query,
    )
    assert registry.get(ref).scope.area_km2 is None
