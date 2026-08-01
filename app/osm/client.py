"""Async Overpass HTTP client.

The client executes a pre-built query against a *configured* endpoint. It never
accepts model-controlled endpoints, timeouts or response limits.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.errors import OverpassError, ToolTimeoutError
from app.osm.contracts import OverpassElement, OverpassResponse


class HttpOverpassClient:
    """POST Overpass QL to a fixed interpreter URL with size and timeout guards."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        max_response_bytes: int,
        user_agent: str = "OSM-GeoAgent/0.1 (research; local)",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_response_bytes < 1024:
            raise ValueError("max_response_bytes must be at least 1024")
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=min(10.0, timeout_seconds),
                read=timeout_seconds,
                write=timeout_seconds,
                pool=timeout_seconds,
            ),
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
        )

    @property
    def endpoint(self) -> str:
        return self._base_url

    async def run(self, query: str) -> OverpassResponse:
        if not query.strip():
            raise OverpassError("Overpass query must not be empty")

        try:
            async with self._client.stream(
                "POST",
                self._base_url,
                data={"data": query},
            ) as response:
                body = await self._read_limited(response)
                status = response.status_code
        except ToolTimeoutError:
            raise
        except OverpassError:
            raise
        except httpx.TimeoutException as exc:
            raise ToolTimeoutError(
                f"Overpass request timed out after {self._timeout_seconds}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise OverpassError(f"Overpass request failed: {exc}") from exc

        self._raise_for_status(status, body)
        payload = self._parse_json(body)
        elements, warnings = self._extract_elements(payload)
        return OverpassResponse(
            elements=elements,
            query=query,
            endpoint=self._base_url,
            retrieved_at=datetime.now(tz=timezone.utc),
            response_bytes=len(body),
            truncated=False,
            warnings=warnings,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _read_limited(self, response: httpx.Response) -> bytes:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = 0
            if declared > self._max_response_bytes:
                raise OverpassError(
                    f"Overpass Content-Length {declared} exceeds "
                    f"limit {self._max_response_bytes} bytes"
                )

        chunks: list[bytes] = []
        total = 0
        try:
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > self._max_response_bytes:
                    raise OverpassError(
                        f"Overpass response exceeded limit of {self._max_response_bytes} bytes"
                    )
                chunks.append(chunk)
        except httpx.TimeoutException as exc:
            raise ToolTimeoutError(
                f"Overpass request timed out after {self._timeout_seconds}s"
            ) from exc
        return b"".join(chunks)

    def _raise_for_status(self, status: int, body: bytes) -> None:
        snippet = body[:300].decode("utf-8", errors="replace")
        if status == 429:
            raise OverpassError("Overpass rate limited the request (HTTP 429)")
        if 400 <= status < 500:
            raise OverpassError(f"Overpass client error HTTP {status}: {snippet}")
        if status >= 500:
            raise OverpassError(f"Overpass server error HTTP {status}: {snippet}")

    def _parse_json(self, body: bytes) -> dict[str, Any]:
        if not body.strip():
            raise OverpassError("Overpass returned an empty body")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OverpassError("Overpass returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise OverpassError("Overpass JSON root must be an object")
        return payload

    def _extract_elements(
        self, payload: dict[str, Any]
    ) -> tuple[tuple[OverpassElement, ...], tuple[str, ...]]:
        warnings: list[str] = []
        remark = payload.get("remark")
        if isinstance(remark, str) and remark.strip():
            text = remark.strip()
            if "error" in text.lower():
                raise OverpassError(f"Overpass reported an error: {text}")
            warnings.append(f"Overpass remark: {text}")

        raw_elements = payload.get("elements", [])
        if raw_elements is None:
            raw_elements = []
        if not isinstance(raw_elements, list):
            raise OverpassError("Overpass JSON 'elements' must be an array")

        elements: list[OverpassElement] = []
        for entry in raw_elements:
            if isinstance(entry, dict):
                elements.append(entry)
            else:
                warnings.append("Skipped a non-object Overpass element")
        return tuple(elements), tuple(warnings)
