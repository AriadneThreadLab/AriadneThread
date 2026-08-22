"""Compact Overpass failure observations for the LLM.

Keeps semantic grounding separate from live-service availability. Never includes
raw HTML bodies or stack traces.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.errors import OverpassError
from app.osm.query_spec import OsmFeatureQuery


def describe_tags(query: OsmFeatureQuery) -> list[str]:
    return [tag.key if tag.value is None else f"{tag.key}={tag.value}" for tag in query.tags]


def describe_scope(query: OsmFeatureQuery) -> str:
    if query.place is not None:
        return query.place
    if query.place_ref_scope is not None:
        return (
            f"{query.place_ref_scope.radius_m}m around place_ref={query.place_ref_scope.place_ref}"
        )
    if query.point is not None:
        return f"{query.point.radius_m}m around {query.point.lat:g},{query.point.lon:g}"
    box = query.bbox
    assert box is not None
    return f"bbox {box.south:g},{box.west:g},{box.north:g},{box.east:g}"


def format_overpass_failure_observation(
    query: OsmFeatureQuery,
    error: OverpassError,
) -> str:
    """Structured tool_error observation after a validated query_osm failure."""
    retryable = error.code in {
        "overpass_timeout",
        "overpass_upstream_error",
        "overpass_rate_limited",
    }
    payload: dict[str, Any] = {
        "tool_error": {
            "tool": "query_osm",
            "code": error.code,
            "retryable": retryable,
            "message": error.message,
            "grounding_remains_valid": True,
            "validated_tags": describe_tags(query),
            "scope": describe_scope(query),
            "effective_limit": query.limit,
            "instructions": [
                "Do not invent replacement OSM tags.",
                "Do not invent coordinates.",
                "Do not write Overpass QL.",
                "Do not claim the validated tag choice was semantically wrong.",
                "Distinguish query validity from external Overpass availability.",
                "State clearly that no live features were returned.",
            ],
        }
    }
    if error.upstream_status is not None:
        payload["tool_error"]["upstream_status"] = error.upstream_status
    if error.attempts:
        payload["tool_error"]["attempts"] = error.attempts
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def public_overpass_error_message(error: OverpassError) -> str:
    """Short UI/API error line without upstream HTML."""
    if error.upstream_status is not None:
        return f"{error.code}: {error.message} (upstream_status={error.upstream_status})"
    return f"{error.code}: {error.message}"
