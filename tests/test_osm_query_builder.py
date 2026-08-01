"""The Overpass builder must be deterministic, bounded and injection-proof."""

from __future__ import annotations

import pytest
from app.core.errors import OverpassQueryBuildError
from app.osm.query_builder import build_overpass_query
from app.osm.query_spec import (
    BoundingBox,
    OsmFeatureQuery,
    PointRadius,
    TagFilter,
)
from pydantic import ValidationError


def test_place_query_matches_expected_overpass_ql():
    query = OsmFeatureQuery(
        place="Berlin",
        tags=[TagFilter(key="leisure", value="park")],
        include_geometry=True,
        limit=1000,
    )
    assert build_overpass_query(query, timeout_seconds=60) == (
        "[out:json][timeout:60];\n"
        'area["name"="Berlin"]->.searchArea;\n'
        "(\n"
        '  node["leisure"="park"](area.searchArea);\n'
        '  way["leisure"="park"](area.searchArea);\n'
        '  relation["leisure"="park"](area.searchArea);\n'
        ");\n"
        "out geom 1000;"
    )


def test_point_radius_query_uses_around_filter():
    query = OsmFeatureQuery(
        point=PointRadius(lat=52.5219, lon=13.4132, radius_m=2000),
        tags=[TagFilter(key="amenity", value="pharmacy")],
        element_types=["node"],
        include_geometry=False,
        limit=50,
    )
    built = build_overpass_query(query, timeout_seconds=25)
    assert '  node["amenity"="pharmacy"](around:2000,52.5219,13.4132);' in built
    assert "out center 50;" in built
    assert "area" not in built


def test_bbox_query_renders_south_west_north_east():
    query = OsmFeatureQuery(
        bbox=BoundingBox(south=52.4, west=13.2, north=52.6, east=13.6),
        tags=[TagFilter(key="natural", value="wood")],
        element_types=["way"],
    )
    assert "(52.4,13.2,52.6,13.6);" in build_overpass_query(query, timeout_seconds=30)


def test_tag_without_value_matches_key_presence():
    query = OsmFeatureQuery(place="Berlin", tags=[TagFilter(key="amenity")])
    assert '  node["amenity"](area.searchArea);' in build_overpass_query(query, timeout_seconds=30)


def test_multiple_tags_are_combined_with_and():
    query = OsmFeatureQuery(
        place="Berlin",
        tags=[TagFilter(key="leisure", value="park"), TagFilter(key="access", value="yes")],
        element_types=["way"],
    )
    built = build_overpass_query(query, timeout_seconds=30)
    assert '  way["leisure"="park"]["access"="yes"](area.searchArea);' in built


def test_element_type_order_is_stable_regardless_of_input_order():
    tags = [TagFilter(key="leisure", value="park")]
    first = build_overpass_query(
        OsmFeatureQuery(place="Berlin", tags=tags, element_types=["relation", "node"]),
        timeout_seconds=30,
    )
    second = build_overpass_query(
        OsmFeatureQuery(place="Berlin", tags=tags, element_types=["node", "relation"]),
        timeout_seconds=30,
    )
    assert first == second
    assert first.index("  node") < first.index("  relation")


def test_builder_is_deterministic():
    query = OsmFeatureQuery(place="Berlin", tags=[TagFilter(key="leisure", value="park")])
    assert build_overpass_query(query, timeout_seconds=60) == build_overpass_query(
        query, timeout_seconds=60
    )


@pytest.mark.parametrize(
    "place",
    [
        'Berlin"];out;//',
        "Berlin\\",
        "Berlin\nout;",
    ],
)
def test_quote_and_escape_injection_in_place_is_rejected(place):
    with pytest.raises(ValidationError):
        OsmFeatureQuery(place=place, tags=[TagFilter(key="leisure", value="park")])


def test_injection_in_tag_value_is_rejected():
    with pytest.raises(ValidationError):
        TagFilter(key="leisure", value='park"];node["amenity"="fuel')


def test_tag_key_must_look_like_an_osm_key():
    with pytest.raises(ValidationError):
        TagFilter(key="leisure park!")


def test_exactly_one_spatial_scope_is_required():
    tags = [TagFilter(key="leisure", value="park")]
    with pytest.raises(ValidationError, match="exactly one"):
        OsmFeatureQuery(tags=tags)
    with pytest.raises(ValidationError, match="exactly one"):
        OsmFeatureQuery(
            place="Berlin",
            point=PointRadius(lat=52.5, lon=13.4, radius_m=100),
            tags=tags,
        )


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        OsmFeatureQuery.model_validate(
            {
                "place": "Berlin",
                "tags": [{"key": "leisure", "value": "park"}],
                "overpass_ql": "[out:json];node;out;",
            }
        )


def test_limit_and_radius_are_bounded():
    tags = [TagFilter(key="leisure", value="park")]
    with pytest.raises(ValidationError):
        OsmFeatureQuery(place="Berlin", tags=tags, limit=100_000)
    with pytest.raises(ValidationError):
        OsmFeatureQuery(point=PointRadius(lat=52.5, lon=13.4, radius_m=10_000_000), tags=tags)


def test_bbox_ordering_is_validated():
    with pytest.raises(ValidationError, match="south must be less than north"):
        BoundingBox(south=52.6, west=13.2, north=52.4, east=13.6)


def test_non_positive_timeout_is_rejected():
    query = OsmFeatureQuery(place="Berlin", tags=[TagFilter(key="leisure", value="park")])
    with pytest.raises(OverpassQueryBuildError):
        build_overpass_query(query, timeout_seconds=0)
