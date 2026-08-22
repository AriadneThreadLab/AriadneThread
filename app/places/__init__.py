"""Trusted place resolution (Nominatim behind Tool Registry)."""

from __future__ import annotations

from app.places.contracts import (
    GeocoderHit,
    PlaceRegistry,
    PlaceResolver,
    ResolvedPlaceRecord,
)

__all__ = [
    "GeocoderHit",
    "PlaceRegistry",
    "PlaceResolver",
    "ResolvedPlaceRecord",
]
