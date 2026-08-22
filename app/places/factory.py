"""Construct place-resolution collaborators from settings."""

from __future__ import annotations

from app.core.config import Settings
from app.places.nominatim import NominatimPlaceResolver
from app.tools.resolve_place import ResolvePlaceTool


def build_resolve_place_tool(settings: Settings) -> ResolvePlaceTool:
    resolver = NominatimPlaceResolver(
        base_url=settings.nominatim_base_url,
        user_agent=settings.osm_wiki_user_agent,
        timeout_seconds=settings.nominatim_timeout_seconds,
    )
    return ResolvePlaceTool(resolver)
