"""Trusted place-resolution contracts.

Coordinates originate only from a configured geocoder behind Tool Registry —
never from LLM memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class GeocoderHit:
    """One ranked geocoder candidate (provider-native, pre-validation)."""

    display_name: str
    latitude: float
    longitude: float
    source_id: str
    raw_class: str | None = None
    raw_type: str | None = None
    importance: float | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ResolvedPlaceRecord:
    """Request-scoped trusted place used for point-radius OSM queries."""

    place_ref: str
    query: str
    label: str
    display_name: str
    latitude: float
    longitude: float
    source: str
    source_id: str


@runtime_checkable
class PlaceResolver(Protocol):
    """Looks up human-readable place queries via a configured endpoint only."""

    @property
    def source_name(self) -> str: ...

    async def search(self, query: str, *, limit: int) -> tuple[GeocoderHit, ...]: ...

    async def aclose(self) -> None: ...


class PlaceRegistry:
    """One instance per agent run. Not shared across requests."""

    MAX_PLACES = 8

    def __init__(self) -> None:
        self._records: dict[str, ResolvedPlaceRecord] = {}
        self._order: list[str] = []

    def register(
        self,
        *,
        query: str,
        label: str,
        display_name: str,
        latitude: float,
        longitude: float,
        source: str,
        source_id: str,
    ) -> ResolvedPlaceRecord:
        if len(self._order) >= self.MAX_PLACES:
            raise ValueError(f"at most {self.MAX_PLACES} places may be resolved per request")
        place_ref = f"place_{len(self._order) + 1}"
        record = ResolvedPlaceRecord(
            place_ref=place_ref,
            query=query,
            label=label,
            display_name=display_name,
            latitude=latitude,
            longitude=longitude,
            source=source,
            source_id=source_id,
        )
        self._records[place_ref] = record
        self._order.append(place_ref)
        return record

    def get(self, place_ref: str) -> ResolvedPlaceRecord | None:
        return self._records.get(place_ref)

    def refs(self) -> tuple[str, ...]:
        return tuple(self._order)

    def has(self, place_ref: str) -> bool:
        return place_ref in self._records
