"""Deterministic SpatialAnalyticsEngine metrics."""

from __future__ import annotations

from datetime import datetime, timezone

from app.analytics.datasets import DatasetRecord, DatasetScope
from app.analytics.engine import SpatialAnalyticsEngine
from app.analytics.observations import (
    AreaObservations,
    CountObservations,
    NumericObservations,
    extract_numeric,
    parse_numeric_tag,
)


def test_parse_numeric_missing_never_zero() -> None:
    assert parse_numeric_tag(None) is None
    assert parse_numeric_tag("12") == 12.0
    assert parse_numeric_tag("12 m") == "invalid"
    assert parse_numeric_tag("3,5") == "invalid"
    assert parse_numeric_tag("nan") == "invalid"


def test_count_and_empty_mean() -> None:
    engine = SpatialAnalyticsEngine()
    assert engine.count(CountObservations(0, {})).value == 0.0
    mean = engine.mean(
        NumericObservations(
            values=(), candidate_count=3, missing_count=3, invalid_count=0, geometry_counts={}
        )
    )
    assert mean.status == "insufficient_data"
    assert mean.value is None


def test_median_and_stdev() -> None:
    engine = SpatialAnalyticsEngine()
    obs = NumericObservations(
        values=(1.0, 2.0, 100.0),
        candidate_count=3,
        missing_count=0,
        invalid_count=0,
        geometry_counts={},
    )
    assert engine.median(obs).value == 2.0
    assert engine.mean(obs).value == (103.0 / 3.0)
    stdev = engine.standard_deviation(obs)
    assert stdev.status == "computed"
    assert stdev.value is not None and stdev.value > 0
    one = NumericObservations(
        values=(5.0,),
        candidate_count=1,
        missing_count=0,
        invalid_count=0,
        geometry_counts={},
    )
    assert engine.standard_deviation(one).status == "insufficient_data"


def test_area_metrics() -> None:
    engine = SpatialAnalyticsEngine()
    obs = AreaObservations(
        areas_m2=(100.0, 300.0),
        candidate_count=2,
        missing_count=0,
        invalid_count=0,
        geometry_counts={"Polygon": 2},
    )
    assert engine.total_area(obs).value == 400.0
    assert engine.mean_area(obs).value == 200.0
    assert engine.median_area(obs).value == 200.0


def test_coverage_clamp() -> None:
    engine = SpatialAnalyticsEngine()
    obs = AreaObservations(
        areas_m2=(2_000_000.0,),
        candidate_count=1,
        missing_count=0,
        invalid_count=0,
        geometry_counts={"Polygon": 1},
    )
    result = engine.coverage_percentage(obs, area_km2=1.0)
    assert result.status == "computed"
    assert result.value == 100.0
    assert any("overlapping" in n for n in result.notes)


def test_extract_numeric_missing_counts() -> None:
    record = DatasetRecord(
        dataset_ref="osm_result_1",
        feature_collection={
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"tags": {"capacity": "10"}},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1, 1]},
                    "properties": {"tags": {}},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [2, 2]},
                    "properties": {"tags": {"capacity": "12 m"}},
                },
            ],
        },
        feature_count=3,
        resolved_tags=("leisure=park",),
        scope=DatasetScope(scope_kind="place", summary="Search area: X"),
        effective_limit=100,
        retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        endpoint="https://overpass.test",
    )
    obs = extract_numeric(record, "capacity")
    assert obs.values == (10.0,)
    assert obs.missing_count == 1
    assert obs.invalid_count == 1
    assert obs.observation_count + obs.missing_count + obs.invalid_count == obs.candidate_count
