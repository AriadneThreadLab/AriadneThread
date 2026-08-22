"""Nominatim HTTP place resolver.

Configured endpoint only. The model never supplies URLs. Offline tests inject
``httpx.MockTransport`` clients.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import httpx

from app.core.errors import PlaceResolutionError
from app.places.contracts import GeocoderHit


class NominatimPlaceResolver:
    """GET search against a fixed Nominatim-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        user_agent: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
                # Prefer Latin/English labels so English landmark queries can match.
                "Accept-Language": "en",
            },
        )

    @property
    def source_name(self) -> str:
        return "nominatim"

    async def search(self, query: str, *, limit: int) -> tuple[GeocoderHit, ...]:
        text = query.strip()
        if len(text) < 2:
            raise PlaceResolutionError("place query must be at least 2 characters")
        if limit < 1:
            raise PlaceResolutionError("limit must be at least 1")
        params = {
            "q": text,
            "format": "jsonv2",
            "limit": str(min(limit, 5)),
            "addressdetails": "0",
            "accept-language": "en",
        }
        url = f"{self._base_url}?{urlencode(params)}"
        try:
            response = await self._client.get(url)
        except httpx.TimeoutException as exc:
            raise PlaceResolutionError("place resolution timed out") from exc
        except httpx.HTTPError as exc:
            raise PlaceResolutionError("place resolution request failed") from exc

        if response.status_code == 429:
            raise PlaceResolutionError("place resolution rate limited (HTTP 429)")
        if response.status_code >= 400:
            raise PlaceResolutionError(
                f"place resolution upstream error HTTP {response.status_code}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise PlaceResolutionError("place resolution returned non-JSON body") from exc
        if not isinstance(payload, list):
            raise PlaceResolutionError("place resolution JSON root must be an array")
        hits: list[GeocoderHit] = []
        for entry in payload:
            hit = _parse_hit(entry)
            if hit is not None:
                hits.append(hit)
        return tuple(hits)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _parse_hit(entry: Any) -> GeocoderHit | None:
    if not isinstance(entry, dict):
        return None
    try:
        lat = float(entry["lat"])
        lon = float(entry["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    display = entry.get("display_name")
    if not isinstance(display, str) or not display.strip():
        return None
    source_id = str(entry.get("osm_id") or entry.get("place_id") or display)
    importance = entry.get("importance")
    return GeocoderHit(
        display_name=display.strip(),
        latitude=lat,
        longitude=lon,
        source_id=source_id,
        raw_class=str(entry["class"]) if isinstance(entry.get("class"), str) else None,
        raw_type=str(entry["type"]) if isinstance(entry.get("type"), str) else None,
        importance=float(importance) if isinstance(importance, int | float) else None,
    )
