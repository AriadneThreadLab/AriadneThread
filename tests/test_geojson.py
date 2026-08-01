"""Overpass → GeoJSON conversion for the MVP element subset."""

from __future__ import annotations

from app.osm.contracts import OSM_ATTRIBUTION
from app.osm.geojson import OverpassGeoJsonEncoder


def test_node_becomes_point_feature():
    result = OverpassGeoJsonEncoder().encode(
        (
            {
                "type": "node",
                "id": 42,
                "lat": 52.5,
                "lon": 13.4,
                "tags": {"amenity": "bench"},
            },
        )
    )
    feature = result.feature_collection["features"][0]
    assert feature["geometry"] == {"type": "Point", "coordinates": [13.4, 52.5]}
    assert feature["properties"]["osm_type"] == "node"
    assert feature["properties"]["osm_id"] == 42
    assert feature["properties"]["tags"] == {"amenity": "bench"}
    assert feature["properties"]["attribution"] == OSM_ATTRIBUTION
    assert result.warnings == ()


def test_way_geometry_becomes_linestring():
    result = OverpassGeoJsonEncoder().encode(
        (
            {
                "type": "way",
                "id": 7,
                "geometry": [
                    {"lat": 1.0, "lon": 2.0},
                    {"lat": 1.1, "lon": 2.1},
                    {"lat": 1.2, "lon": 2.2},
                ],
                "tags": {"highway": "path"},
            },
        )
    )
    geometry = result.feature_collection["features"][0]["geometry"]
    assert geometry["type"] == "LineString"
    assert geometry["coordinates"] == [[2.0, 1.0], [2.1, 1.1], [2.2, 1.2]]


def test_closed_way_geometry_becomes_polygon():
    result = OverpassGeoJsonEncoder().encode(
        (
            {
                "type": "way",
                "id": 8,
                "geometry": [
                    {"lat": 0.0, "lon": 0.0},
                    {"lat": 0.0, "lon": 1.0},
                    {"lat": 1.0, "lon": 1.0},
                    {"lat": 0.0, "lon": 0.0},
                ],
            },
        )
    )
    assert result.feature_collection["features"][0]["geometry"]["type"] == "Polygon"


def test_center_only_way_becomes_point():
    result = OverpassGeoJsonEncoder().encode(
        (
            {
                "type": "way",
                "id": 9,
                "center": {"lat": 52.52, "lon": 13.40},
                "tags": {"leisure": "park"},
            },
        )
    )
    assert result.feature_collection["features"][0]["geometry"] == {
        "type": "Point",
        "coordinates": [13.40, 52.52],
    }
    assert result.warnings == ()


def test_unsupported_relation_without_geometry_warns_and_skips():
    result = OverpassGeoJsonEncoder().encode(
        (
            {
                "type": "relation",
                "id": 100,
                "tags": {"type": "multipolygon", "leisure": "park"},
                "members": [{"type": "way", "ref": 1, "role": "outer"}],
            },
        )
    )
    assert result.feature_collection["features"] == []
    assert any("full relation geometry is not supported" in w for w in result.warnings)


def test_relation_with_center_returns_point_and_warning():
    result = OverpassGeoJsonEncoder().encode(
        (
            {
                "type": "relation",
                "id": 101,
                "center": {"lat": 52.5, "lon": 13.4},
                "tags": {"type": "multipolygon"},
            },
        )
    )
    assert len(result.feature_collection["features"]) == 1
    assert result.feature_collection["features"][0]["geometry"]["type"] == "Point"
    assert any("only centre point" in w for w in result.warnings)


def test_empty_elements_yield_empty_feature_collection():
    result = OverpassGeoJsonEncoder().encode(())
    assert result.feature_collection == {"type": "FeatureCollection", "features": []}
    assert result.warnings == ()
