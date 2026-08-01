"""Overpass transport contracts.

The HTTP implementation is deliberately absent in this phase; only the shape of
the boundary is fixed, so the tool layer and its tests can be written against a
fake client.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

#: Overpass returns a JSON document with an ``elements`` array. Element shapes
#: differ per type and per ``out`` mode, so they stay untyped until the GeoJSON
#: converter is implemented.
OverpassElement = dict[str, Any]

#: A GeoJSON ``FeatureCollection`` object, ready to hand to a map client.
GeoJsonFeatureCollection = dict[str, Any]

#: Attribution required whenever OSM-derived data is shown to a user.
OSM_ATTRIBUTION = "© OpenStreetMap contributors (ODbL)"


@dataclass(frozen=True, slots=True)
class OverpassResponse:
    """Raw, successful Overpass result plus transport metadata."""

    elements: tuple[OverpassElement, ...]
    query: str
    endpoint: str
    retrieved_at: datetime
    response_bytes: int
    truncated: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class OverpassClient(Protocol):
    """Executes a pre-built Overpass QL query.

    Implementations must enforce an explicit request timeout and a maximum
    response size, and must raise :class:`app.core.errors.OverpassError` (or
    :class:`app.core.errors.ToolTimeoutError`) on failure. They never build or
    modify the query text.
    """

    @property
    def endpoint(self) -> str: ...

    async def run(self, query: str) -> OverpassResponse: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class OsmGeoJsonEncoder(Protocol):
    """Converts raw Overpass elements into a GeoJSON ``FeatureCollection``.

    Kept separate from the transport so the conversion rules (centre points,
    ways, relation members, tag mapping) can be developed and tested on stored
    fixtures without any network access.
    """

    def encode(self, elements: Sequence[OverpassElement]) -> GeoJsonFeatureCollection: ...
