"""Reject model-invented geographic coordinates.

Named-place scopes are always allowed. Point-radius and bounding-box scopes are
accepted only when the numeric coordinates appear in the original user message
(so the model cannot invent extents for Tehran, Azadi Square, etc.).

Trusted ``place_ref_scope`` uses backend-resolved coordinates; the LLM must not
supply latitude/longitude. The radius must still match the user request.
"""

from __future__ import annotations

import re
from typing import Any

from app.places.contracts import PlaceRegistry

_KM_RADIUS = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:km|kilometers?|kilometres?)\b",
    re.IGNORECASE,
)
_M_RADIUS = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\b",
    re.IGNORECASE,
)


def _number_mentioned(message: str, value: float) -> bool:
    """Whether ``value`` appears in the user message as a decimal/integer token."""
    # Compare against numeric tokens in the message so trailing zeros in either
    # side (35.6997 vs 35.69970) do not cause false rejections.
    for match in re.finditer(r"-?\d+(?:\.\d+)?", message):
        try:
            if abs(float(match.group(0)) - float(value)) <= 1e-9:
                return True
        except ValueError:
            continue
    return False


def coordinates_supported_by_user(lat: float, lon: float, user_message: str) -> bool:
    """True when both coordinates appear as numeric tokens in the user message."""
    return _number_mentioned(user_message, float(lat)) and _number_mentioned(
        user_message, float(lon)
    )


def radius_supported_by_user(radius_m: int, user_message: str) -> bool:
    """True when the metre radius matches an explicit user distance."""
    if _number_mentioned(user_message, float(radius_m)):
        return True
    for match in _M_RADIUS.finditer(user_message):
        try:
            if abs(float(match.group(1)) - float(radius_m)) <= 1e-6:
                return True
        except ValueError:
            continue
    for match in _KM_RADIUS.finditer(user_message):
        try:
            metres = float(match.group(1)) * 1000.0
            if abs(metres - float(radius_m)) <= 1e-3:
                return True
        except ValueError:
            continue
    return False


def untrusted_spatial_scope_error(
    arguments: dict[str, Any],
    user_message: str,
    *,
    places: PlaceRegistry | None = None,
) -> str | None:
    """Return a safe error message when the scope is untrusted, else ``None``."""
    place = arguments.get("place")
    point = arguments.get("point")
    bbox = arguments.get("bbox")
    place_ref_scope = arguments.get("place_ref_scope")

    scopes = [
        place is not None,
        point is not None,
        bbox is not None,
        place_ref_scope is not None,
    ]
    if sum(bool(flag) for flag in scopes) != 1:
        # Let OsmFeatureQuery validation report the exact schema problem.
        return None

    if place is not None:
        if isinstance(place, list):
            return (
                "place must be a single named area string; resolve each landmark "
                "with resolve_place and query_osm once per target via place_ref_scope"
            )
        return None

    if isinstance(place_ref_scope, dict):
        place_ref = place_ref_scope.get("place_ref")
        radius = place_ref_scope.get("radius_m")
        if not isinstance(place_ref, str) or not place_ref:
            return None
        if places is None or not places.has(place_ref):
            return (
                f"unknown place_ref '{place_ref}'; call resolve_place first and use "
                "only place_ref values present in tool observations"
            )
        if not isinstance(radius, int):
            return None
        if not radius_supported_by_user(radius, user_message):
            return (
                "place_ref_scope.radius_m must match a distance explicitly stated "
                "in the user request; inventing a buffer distance is not allowed"
            )
        return None

    if isinstance(point, dict):
        lat = point.get("lat")
        lon = point.get("lon")
        if not isinstance(lat, int | float) or not isinstance(lon, int | float):
            return None
        if coordinates_supported_by_user(float(lat), float(lon), user_message):
            return None
        return (
            "point-radius scope requires latitude and longitude explicitly provided "
            "by the user or a trusted place_ref_scope from resolve_place; inventing "
            "coordinates is not allowed. Use resolve_place for landmarks or a named "
            "place for cities."
        )

    if isinstance(bbox, dict):
        for key in ("south", "west", "north", "east"):
            value = bbox.get(key)
            if not isinstance(value, int | float):
                return None
            if not _number_mentioned(user_message, float(value)):
                return (
                    "bounding-box scope requires south/west/north/east values explicitly "
                    "provided by the user; inventing coordinates is not allowed. Use a "
                    "named place or ask the user for coordinates."
                )
        return None

    return None
