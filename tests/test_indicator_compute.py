"""Deterministic indicator executors on synthetic GeoJSON (offline)."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import pytest
from app.analytics.contracts import GeoPoint
from app.analytics.datasets import DatasetRecord, DatasetScope
from app.analytics.engine import SpatialAnalyticsEngine
from app.analytics.geometry import (
    bbox_area_m2,
    clipped_geometry_length_m,
    geometry_area_m2,
    haversine_m,
    polyline_length_m,
    spherical_cap_area_m2,
)
from app.analytics.network import count_network_intersections
from app.analytics.observations import CategoryObservations
from app.core.errors import UnknownIndicatorError
from app.indicators.categorize import assign_category
from app.indicators.compute import compute_indicator
from app.indicators.contracts import IndicatorComputeRequest, TagCategory
from app.osm.query_spec import MAX_RADIUS_METERS

_ENTROPY_BASE = 2.718281828459045

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_ENDPOINT = "https://overpass.test"


def _feature(
    geometry: dict[str, Any],
    *,
    osm_id: int,
    osm_type: str = "way",
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "osm_type": osm_type,
            "osm_id": osm_id,
            "tags": tags or {},
        },
    }


def _record(
    features: list[dict[str, Any]],
    *,
    dataset_ref: str = "osm_result_1",
    scope: DatasetScope | None = None,
) -> DatasetRecord:
    return DatasetRecord(
        dataset_ref=dataset_ref,
        feature_collection={"type": "FeatureCollection", "features": features},
        feature_count=len(features),
        resolved_tags=("test=yes",),
        scope=scope
        or DatasetScope(
            scope_kind="bbox",
            summary="bbox",
            bbox=(0.0, 0.0, 1.0, 1.0),
            area_km2=bbox_area_m2(0.0, 0.0, 1.0, 1.0) / 1_000_000.0,
        ),
        effective_limit=1000,
        retrieved_at=_NOW,
        endpoint=_ENDPOINT,
    )


def _point_scope(lat: float = 0.0, lon: float = 0.0, radius_m: int = 2000) -> DatasetScope:
    area_m2 = spherical_cap_area_m2(float(radius_m))
    return DatasetScope(
        scope_kind="point",
        summary="cap",
        center=GeoPoint(lat=lat, lon=lon),
        radius_m=radius_m,
        area_km2=area_m2 / 1_000_000.0,
    )


def test_equator_degree_is_geodesic_not_degree_units() -> None:
    length = polyline_length_m([[0.0, 0.0], [1.0, 0.0]])
    assert length is not None
    expected = haversine_m(0.0, 0.0, 0.0, 1.0)
    assert abs(length - expected) < 1e-6
    assert 110_000 < length < 113_000


def test_feature_count_deduplicates_osm_ids() -> None:
    point = {"type": "Point", "coordinates": [51.4, 35.7]}
    record = _record(
        [
            _feature(point, osm_id=1, osm_type="node"),
            _feature(point, osm_id=1, osm_type="node"),
            _feature({"type": "Point", "coordinates": [51.41, 35.71]}, osm_id=2, osm_type="node"),
        ]
    )
    result = compute_indicator(
        IndicatorComputeRequest(indicator_id="feature_count", datasets={"subject_features": record})
    )
    assert result.status == "computed"
    assert result.value == 2.0
    assert result.unit == "features"
    assert result.methodology == "count"
    assert result.dataset_refs == ("osm_result_1",)
    assert result.observation_count == 2


def test_green_space_ratio_known_bbox_coverage() -> None:
    ring = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
    polygon = {"type": "Polygon", "coordinates": [ring]}
    record = _record([_feature(polygon, osm_id=10, tags={"leisure": "park"})])
    result = compute_indicator(
        IndicatorComputeRequest(
            indicator_id="green_space_ratio", datasets={"green_polygons": record}
        )
    )
    assert result.status == "computed"
    assert result.methodology == "area_share"
    assert result.unit == "percent"
    assert result.value is not None
    assert abs(result.value - 100.0) < 2.5
    feature_area = geometry_area_m2(polygon)
    assert feature_area is not None
    assert abs(feature_area - bbox_area_m2(0.0, 0.0, 1.0, 1.0)) / feature_area < 0.02


def test_green_space_ratio_handles_multipolygon() -> None:
    left = [[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4], [0.0, 0.0]]
    right = [[0.6, 0.6], [1.0, 0.6], [1.0, 1.0], [0.6, 1.0], [0.6, 0.6]]
    geometry = {"type": "MultiPolygon", "coordinates": [[left], [right]]}
    record = _record([_feature(geometry, osm_id=11)])
    result = compute_indicator(
        IndicatorComputeRequest(
            indicator_id="green_space_ratio", datasets={"green_polygons": record}
        )
    )
    assert result.status == "computed"
    assert result.value is not None
    assert 0.0 < result.value < 50.0


def test_road_density_two_segment_way() -> None:
    line = {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]}
    record = _record([_feature(line, osm_id=20, tags={"highway": "residential"})])
    result = compute_indicator(
        IndicatorComputeRequest(indicator_id="road_density", datasets={"highway_features": record})
    )
    length_m = polyline_length_m([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    assert length_m is not None
    area_km2 = bbox_area_m2(0.0, 0.0, 1.0, 1.0) / 1_000_000.0
    expected = (length_m / 1000.0) / area_km2
    assert result.status == "computed"
    assert result.methodology == "length_density"
    assert result.unit == "km_per_km2"
    assert result.value is not None
    assert abs(result.value - expected) / expected < 0.02
    assert "clipped" in " ".join(result.warnings)


def test_road_density_clips_to_circular_scope() -> None:
    line = {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 0.0]]}
    record = _record(
        [_feature(line, osm_id=21, tags={"highway": "primary"})],
        scope=_point_scope(0.0, 0.0, 1000),
    )
    result = compute_indicator(
        IndicatorComputeRequest(indicator_id="road_density", datasets={"highway_features": record})
    )
    clipped, applied = clipped_geometry_length_m(line, center=(0.0, 0.0), radius_m=1000.0)
    assert applied is True
    assert clipped is not None
    assert 900.0 < clipped < 1100.0
    assert result.status == "computed"
    assert result.value is not None
    area_km2 = spherical_cap_area_m2(1000.0) / 1_000_000.0
    expected = (clipped / 1000.0) / area_km2
    assert abs(result.value - expected) / expected < 0.05


def test_road_density_handles_multilinestring() -> None:
    geometry = {
        "type": "MultiLineString",
        "coordinates": [
            [[0.0, 0.0], [0.2, 0.0]],
            [[0.4, 0.0], [0.6, 0.0]],
        ],
    }
    record = _record([_feature(geometry, osm_id=22)])
    result = compute_indicator(
        IndicatorComputeRequest(indicator_id="road_density", datasets={"highway_features": record})
    )
    assert result.status == "computed"
    assert result.value is not None
    assert result.value > 0.0


def test_road_accessibility_nearest_on_linestring() -> None:
    line = {"type": "LineString", "coordinates": [[1.0, 0.0], [2.0, 0.0]]}
    record = _record([_feature(line, osm_id=30, tags={"highway": "residential"})])
    result = compute_indicator(
        IndicatorComputeRequest(
            indicator_id="road_accessibility",
            datasets={"highway_features": record},
            reference_points=(GeoPoint(lat=0.0, lon=0.0),),
        )
    )
    expected = haversine_m(0.0, 0.0, 0.0, 1.0)
    assert result.status == "computed"
    assert result.methodology == "nearest_distance"
    assert result.unit == "m"
    assert result.value is not None
    assert abs(result.value - expected) / expected < 0.02
    assert any("geodesic" in item for item in result.warnings)


def test_poi_density_known_count_over_cap() -> None:
    features = [
        _feature(
            {"type": "Point", "coordinates": [0.001 * index, 0.0]},
            osm_id=40 + index,
            osm_type="node",
            tags={"amenity": "school"},
        )
        for index in range(4)
    ]
    record = _record(features, scope=_point_scope(0.0, 0.0, 2000))
    result = compute_indicator(
        IndicatorComputeRequest(indicator_id="poi_density", datasets={"service_pois": record})
    )
    area_km2 = spherical_cap_area_m2(2000.0) / 1_000_000.0
    assert result.status == "computed"
    assert result.methodology == "density"
    assert result.unit == "features_per_km2"
    assert result.value is not None
    assert abs(result.value - (4.0 / area_km2)) / result.value < 1e-9


def test_service_accessibility_nearest_point() -> None:
    features = [
        _feature(
            {"type": "Point", "coordinates": [0.01, 0.0]},
            osm_id=50,
            osm_type="node",
            tags={"amenity": "hospital"},
        ),
        _feature(
            {"type": "Point", "coordinates": [0.05, 0.0]},
            osm_id=51,
            osm_type="node",
            tags={"amenity": "school"},
        ),
    ]
    record = _record(features, scope=_point_scope())
    result = compute_indicator(
        IndicatorComputeRequest(
            indicator_id="service_accessibility",
            datasets={"service_pois": record},
            reference_points=(GeoPoint(lat=0.0, lon=0.0),),
        )
    )
    expected = haversine_m(0.0, 0.0, 0.0, 0.01)
    assert result.status == "computed"
    assert result.value is not None
    assert abs(result.value - expected) < 1.0


def test_green_accessibility_is_explicitly_unimplemented() -> None:
    ring = [[0.0, 0.0], [0.01, 0.0], [0.01, 0.01], [0.0, 0.01], [0.0, 0.0]]
    record = _record([_feature({"type": "Polygon", "coordinates": [ring]}, osm_id=60)])
    result = compute_indicator(
        IndicatorComputeRequest(
            indicator_id="green_accessibility",
            datasets={"green_polygons": record},
            reference_points=(GeoPoint(lat=0.0, lon=0.0),),
        )
    )
    assert result.status == "unimplemented"
    assert result.value is None
    assert result.methodology == "distance_decay_sum"
    assert any("pinned" in item or "Hansen" in item for item in result.warnings)


def test_unknown_indicator_still_fails() -> None:
    record = _record([])
    with pytest.raises(UnknownIndicatorError):
        compute_indicator(
            IndicatorComputeRequest(
                indicator_id="invented_score", datasets={"subject_features": record}
            )
        )


def test_empty_feature_count_is_zero() -> None:
    record = _record([])
    result = compute_indicator(
        IndicatorComputeRequest(indicator_id="feature_count", datasets={"subject_features": record})
    )
    assert result.status == "computed"
    assert result.value == 0.0


def test_point_radius_within_overpass_bound() -> None:
    assert MAX_RADIUS_METERS >= 2000


def _shannon_nats(counts: list[int], log_base: float = _ENTROPY_BASE) -> float:
    total = float(sum(counts))
    return -sum(
        (count / total) * (math.log(count / total) / math.log(log_base)) for count in counts
    )


def _green_point(osm_id: int, tags: dict[str, str] | None) -> dict[str, Any]:
    feature = _feature(
        {"type": "Point", "coordinates": [0.001 * osm_id, 0.0]},
        osm_id=osm_id,
        osm_type="node",
        tags=tags,
    )
    if tags is None:
        feature["properties"].pop("tags")
    return feature


def _service_point(osm_id: int, tags: dict[str, str]) -> dict[str, Any]:
    return _feature(
        {"type": "Point", "coordinates": [0.001 * osm_id, 0.0]},
        osm_id=osm_id,
        osm_type="node",
        tags=tags,
    )


def _network_scope() -> DatasetScope:
    south, west, north, east = -0.02, -0.02, 0.02, 0.02
    return DatasetScope(
        scope_kind="bbox",
        summary="network bbox",
        bbox=(south, west, north, east),
        area_km2=bbox_area_m2(south, west, north, east) / 1_000_000.0,
    )


def _place_scope(*, area_km2: float = 2.0) -> DatasetScope:
    return DatasetScope(
        scope_kind="place",
        summary="named place",
        place="Testville",
        area_km2=area_km2,
    )


def _highway(
    coords: list[list[float]],
    osm_id: int,
    *,
    nodes: list[int] | None = None,
) -> dict[str, Any]:
    feature = _feature(
        {"type": "LineString", "coordinates": coords},
        osm_id=osm_id,
        tags={"highway": "residential"},
    )
    if nodes is not None:
        feature["properties"]["nodes"] = nodes
    return feature


def test_shannon_entropy_two_equal_categories_is_ln2() -> None:
    obs = CategoryObservations(
        category_counts=(2, 2),
        candidate_count=4,
        missing_count=0,
        unknown_count=0,
        multi_match_count=0,
        geometry_counts={"Point": 4},
    )
    result = SpatialAnalyticsEngine().shannon_entropy(obs, log_base=_ENTROPY_BASE)
    assert result.status == "computed"
    assert result.value is not None
    assert abs(result.value - math.log(2)) < 1e-9
    assert abs(result.value - _shannon_nats([2, 2])) < 1e-15


def test_shannon_entropy_one_category_is_zero() -> None:
    obs = CategoryObservations(
        category_counts=(5, 0, 0, 0),
        candidate_count=5,
        missing_count=0,
        unknown_count=0,
        multi_match_count=0,
        geometry_counts={"Point": 5},
    )
    result = SpatialAnalyticsEngine().shannon_entropy(obs, log_base=_ENTROPY_BASE)
    assert result.status == "computed"
    assert result.value == 0.0


def test_shannon_entropy_empty_is_insufficient() -> None:
    obs = CategoryObservations(
        category_counts=(0, 0),
        candidate_count=0,
        missing_count=0,
        unknown_count=0,
        multi_match_count=0,
        geometry_counts={},
    )
    result = SpatialAnalyticsEngine().shannon_entropy(obs, log_base=_ENTROPY_BASE)
    assert result.status == "insufficient_data"
    assert result.value is None


def test_green_diversity_two_equal_types() -> None:
    record = _record(
        [
            _green_point(1, {"leisure": "park"}),
            _green_point(2, {"leisure": "park"}),
            _green_point(3, {"natural": "wood"}),
            _green_point(4, {"natural": "wood"}),
        ]
    )
    result = compute_indicator(
        IndicatorComputeRequest(indicator_id="green_diversity", datasets={"green_polygons": record})
    )
    assert result.status == "computed"
    assert result.unit == "nats"
    assert result.methodology == "shannon_entropy"
    assert result.observation_count == 4
    assert result.value is not None
    assert abs(result.value - math.log(2)) < 1e-9
    assert any("not normalised" in item for item in result.warnings)


def test_green_diversity_one_category_is_zero() -> None:
    record = _record(
        [
            _green_point(1, {"leisure": "park"}),
            _green_point(2, {"leisure": "park"}),
            _green_point(3, {"leisure": "park"}),
        ]
    )
    result = compute_indicator(
        IndicatorComputeRequest(indicator_id="green_diversity", datasets={"green_polygons": record})
    )
    assert result.status == "computed"
    assert result.value == 0.0
    assert result.observation_count == 3


def test_green_diversity_empty_is_insufficient() -> None:
    result = compute_indicator(
        IndicatorComputeRequest(
            indicator_id="green_diversity", datasets={"green_polygons": _record([])}
        )
    )
    assert result.status == "insufficient_data"
    assert result.value is None


def test_green_diversity_single_feature_is_insufficient() -> None:
    record = _record([_green_point(1, {"leisure": "park"})])
    result = compute_indicator(
        IndicatorComputeRequest(indicator_id="green_diversity", datasets={"green_polygons": record})
    )
    assert result.status == "insufficient_data"
    assert result.value is None
    assert result.observation_count == 1


def test_green_diversity_excludes_missing_tags() -> None:
    record = _record(
        [
            _green_point(1, {"leisure": "park"}),
            _green_point(2, {"leisure": "park"}),
            _green_point(3, {}),
            _green_point(4, None),
        ]
    )
    result = compute_indicator(
        IndicatorComputeRequest(indicator_id="green_diversity", datasets={"green_polygons": record})
    )
    assert result.status == "computed"
    assert result.value == 0.0
    assert result.observation_count == 2
    assert result.missing_count == 2
    assert any("no OSM tags" in item for item in result.warnings)


def test_green_diversity_excludes_unknown_categories() -> None:
    record = _record(
        [
            _green_point(1, {"leisure": "park"}),
            _green_point(2, {"leisure": "park"}),
            _green_point(3, {"leisure": "playground"}),
        ]
    )
    result = compute_indicator(
        IndicatorComputeRequest(indicator_id="green_diversity", datasets={"green_polygons": record})
    )
    assert result.status == "computed"
    assert result.value == 0.0
    assert result.observation_count == 2
    assert result.missing_count == 0
    assert any("no catalog category" in item for item in result.warnings)


def test_green_diversity_unknown_only_is_insufficient() -> None:
    record = _record(
        [
            _green_point(1, {"leisure": "playground"}),
            _green_point(2, {"leisure": "playground"}),
        ]
    )
    result = compute_indicator(
        IndicatorComputeRequest(indicator_id="green_diversity", datasets={"green_polygons": record})
    )
    assert result.status == "insufficient_data"
    assert result.value is None
    assert result.observation_count == 0


def test_service_diversity_known_entropy() -> None:
    record = _record(
        [
            _service_point(1, {"amenity": "school"}),
            _service_point(2, {"amenity": "school"}),
            _service_point(3, {"amenity": "hospital"}),
            _service_point(4, {"amenity": "hospital"}),
        ]
    )
    result = compute_indicator(
        IndicatorComputeRequest(indicator_id="service_diversity", datasets={"service_pois": record})
    )
    assert result.status == "computed"
    assert result.unit == "nats"
    assert result.value is not None
    assert abs(result.value - math.log(2)) < 1e-9


def test_first_matching_category_wins() -> None:
    park = TagCategory(id="park", label="Park", tags=("leisure=park",))
    leisure = TagCategory(id="leisure", label="Leisure", tags=("leisure",))
    primary, matched = assign_category({"leisure": "park"}, (park, leisure))
    assert primary == "park"
    assert matched == ("park", "leisure")


def test_plus_junction_is_one_intersection() -> None:
    features = [
        _highway([[-0.001, 0.0], [0.0, 0.0], [0.001, 0.0]], 1),
        _highway([[0.0, -0.001], [0.0, 0.0], [0.0, 0.001]], 2),
    ]
    census = count_network_intersections(features, _network_scope())
    assert census.intersection_count == 1
    assert census.way_count == 2
    assert any("degree ≥ 3" in item for item in census.notes)


def test_t_junction_is_one_intersection() -> None:
    features = [
        _highway([[-0.001, 0.0], [0.0, 0.0], [0.001, 0.0]], 1),
        _highway([[0.0, 0.0], [0.0, 0.001]], 2),
    ]
    census = count_network_intersections(features, _network_scope())
    assert census.intersection_count == 1


def test_dense_vertices_on_one_way_are_not_intersections() -> None:
    coords = [[index * 0.0001, 0.0] for index in range(40)]
    census = count_network_intersections([_highway(coords, 1)], _network_scope())
    assert census.intersection_count == 0
    assert census.vertex_count == 40


def test_end_to_end_ways_are_not_an_intersection() -> None:
    features = [
        _highway([[-0.001, 0.0], [0.0, 0.0]], 1),
        _highway([[0.0, 0.0], [0.001, 0.0]], 2),
    ]
    census = count_network_intersections(features, _network_scope())
    assert census.intersection_count == 0


def test_geometric_crossing_without_shared_vertex_is_not_an_intersection() -> None:
    features = [
        _highway([[-0.001, -0.001], [0.001, 0.001]], 1),
        _highway([[-0.001, 0.001], [0.001, -0.001]], 2),
    ]
    census = count_network_intersections(features, _network_scope())
    assert census.intersection_count == 0


def test_osm_node_ids_count_shared_degree_three() -> None:
    features = [
        _highway([[-0.001, 0.0], [0.0, 0.0], [0.001, 0.0]], 1, nodes=[10, 11, 12]),
        _highway([[0.0, 0.0], [0.0, 0.001]], 2, nodes=[11, 13]),
    ]
    census = count_network_intersections(features, _place_scope())
    assert census.used_osm_node_ids is True
    assert census.intersection_count == 1


def test_intersection_density_known_count_over_area() -> None:
    features = [
        _highway([[-0.001, 0.0], [0.0, 0.0], [0.001, 0.0]], 1),
        _highway([[0.0, -0.001], [0.0, 0.0], [0.0, 0.001]], 2),
    ]
    scope = _network_scope()
    record = _record(features, scope=scope)
    result = compute_indicator(
        IndicatorComputeRequest(
            indicator_id="intersection_density", datasets={"highway_features": record}
        )
    )
    assert result.status == "computed"
    assert result.unit == "intersections_per_km2"
    assert result.methodology == "intersection_density"
    assert result.observation_count == 1
    assert result.value is not None
    assert scope.area_km2 is not None
    assert abs(result.value - (1.0 / scope.area_km2)) / result.value < 1e-12
    assert any("degree ≥ 3" in item for item in result.warnings)


def test_intersection_density_zero_junctions_is_computed() -> None:
    record = _record(
        [_highway([[-0.001, 0.0], [0.001, 0.0]], 1)],
        scope=_network_scope(),
    )
    result = compute_indicator(
        IndicatorComputeRequest(
            indicator_id="intersection_density", datasets={"highway_features": record}
        )
    )
    assert result.status == "computed"
    assert result.value == 0.0
    assert result.observation_count == 0


def test_intersection_density_empty_network_is_zero() -> None:
    record = _record([], scope=_network_scope())
    result = compute_indicator(
        IndicatorComputeRequest(
            indicator_id="intersection_density", datasets={"highway_features": record}
        )
    )
    assert result.status == "computed"
    assert result.value == 0.0


def test_intersection_density_uses_spherical_cap_area() -> None:
    features = [
        _highway([[-0.001, 0.0], [0.0, 0.0], [0.001, 0.0]], 1),
        _highway([[0.0, -0.001], [0.0, 0.0], [0.0, 0.001]], 2),
    ]
    scope = _point_scope(0.0, 0.0, 2000)
    record = _record(features, scope=scope)
    result = compute_indicator(
        IndicatorComputeRequest(
            indicator_id="intersection_density", datasets={"highway_features": record}
        )
    )
    area_km2 = spherical_cap_area_m2(2000.0) / 1_000_000.0
    assert result.status == "computed"
    assert result.value is not None
    assert abs(result.value - (1.0 / area_km2)) / result.value < 1e-12
