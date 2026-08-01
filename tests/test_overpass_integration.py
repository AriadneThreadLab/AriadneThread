"""Optional live Overpass check.

Skipped unless ``RUN_INTEGRATION=1``. Not part of the normal offline suite.
"""

from __future__ import annotations

import os

import pytest
from app.core.config import get_settings
from app.osm.client import HttpOverpassClient
from app.osm.query_builder import build_overpass_query
from app.osm.query_spec import OsmFeatureQuery, PointRadius, TagFilter

pytestmark = pytest.mark.integration


def _integration_enabled() -> bool:
    return os.environ.get("RUN_INTEGRATION") == "1"


@pytest.mark.asyncio
async def test_live_overpass_accepts_a_tiny_node_query():
    if not _integration_enabled():
        pytest.skip("Set RUN_INTEGRATION=1 to run a real Overpass request")

    get_settings.cache_clear()
    settings = get_settings()
    client = HttpOverpassClient(
        base_url=settings.overpass_base_url,
        timeout_seconds=min(settings.overpass_timeout_seconds, 30.0),
        max_response_bytes=settings.overpass_max_response_bytes,
        user_agent=settings.osm_wiki_user_agent,
    )
    query = build_overpass_query(
        OsmFeatureQuery(
            point=PointRadius(lat=52.5200, lon=13.4050, radius_m=40),
            tags=[TagFilter(key="amenity", value="bench")],
            element_types=["node"],
            include_geometry=False,
            limit=1,
        ),
        timeout_seconds=25,
    )
    try:
        response = await client.run(query)
    finally:
        await client.aclose()
        get_settings.cache_clear()

    # Empty is a valid live answer; the check is that Overpass accepted the query.
    assert isinstance(response.elements, tuple)
    assert response.query == query
