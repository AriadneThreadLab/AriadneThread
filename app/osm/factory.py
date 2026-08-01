"""Construct Overpass collaborators from settings."""

from __future__ import annotations

from app.core.config import Settings
from app.osm.client import HttpOverpassClient
from app.osm.geojson import OverpassGeoJsonEncoder
from app.tools.query_osm import QueryOsmTool


def build_query_osm_tool(settings: Settings) -> QueryOsmTool:
    """Build the registered ``query_osm`` tool with configured limits."""
    client = HttpOverpassClient(
        base_url=settings.overpass_base_url,
        timeout_seconds=settings.overpass_timeout_seconds,
        max_response_bytes=settings.overpass_max_response_bytes,
        user_agent=settings.osm_wiki_user_agent,
    )
    return QueryOsmTool(
        client,
        OverpassGeoJsonEncoder(),
        timeout_seconds=int(settings.overpass_timeout_seconds),
        max_results=settings.overpass_max_results,
    )
