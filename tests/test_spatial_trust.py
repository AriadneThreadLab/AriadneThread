"""Named-place preference and rejection of invented coordinates."""

from __future__ import annotations

import pytest
from app.agent.spatial_trust import untrusted_spatial_scope_error
from app.osm.query_spec import MAX_LIMIT, OsmFeatureQuery, TagFilter
from pydantic import ValidationError


def test_named_place_scope_is_trusted_without_coordinates():
    assert (
        untrusted_spatial_scope_error(
            {"place": "Tehran, Iran", "tags": [["leisure", "park"]], "limit": 20},
            "Find public parks in Tehran, Iran. Return at most 20 features.",
        )
        is None
    )


def test_invented_bbox_is_rejected_when_user_did_not_provide_bounds():
    err = untrusted_spatial_scope_error(
        {
            "bbox": {
                "west": 59.937208,
                "east": 60.147208,
                "north": 35.950469,
                "south": 35.550469,
            },
            "tags": [["amenity", "park"]],
            "limit": 20,
        },
        "Find public parks in Tehran, Iran. Return at most 20 features.",
    )
    assert err is not None
    assert "bounding-box" in err
    assert "inventing" in err.lower()


def test_invented_point_radius_is_rejected():
    err = untrusted_spatial_scope_error(
        {
            "point": {"lat": 35.6997, "lon": 51.3380, "radius_m": 2000},
            "tags": [{"key": "leisure", "value": "park"}],
        },
        "Find parks near Azadi Square.",
    )
    assert err is not None
    assert "point-radius" in err


def test_explicit_user_coordinates_are_trusted():
    message = "Find pharmacies within 500 m of 35.6997, 51.3380"
    assert (
        untrusted_spatial_scope_error(
            {
                "point": {"lat": 35.6997, "lon": 51.3380, "radius_m": 500},
                "tags": [{"key": "amenity", "value": "pharmacy"}],
            },
            message,
        )
        is None
    )


def test_tag_pairs_coerce_to_tag_filters():
    query = OsmFeatureQuery(
        place="Tehran, Iran",
        tags=[["leisure", "park"]],
        limit=20,
    )
    assert query.tags[0].key == "leisure"
    assert query.tags[0].value == "park"
    assert query.scope_kind == "place"


def test_limit_above_application_maximum_is_rejected():
    with pytest.raises(ValidationError):
        OsmFeatureQuery(
            place="Tehran, Iran",
            tags=[TagFilter(key="leisure", value="park")],
            limit=MAX_LIMIT + 1,
        )


def test_limit_2000_cannot_override_schema_maximum():
    with pytest.raises(ValidationError):
        OsmFeatureQuery(
            place="Tehran, Iran",
            tags=[TagFilter(key="leisure", value="park")],
            limit=2000,
        )
