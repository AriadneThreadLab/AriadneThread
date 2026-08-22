"""Async Overpass HTTP client.

The client executes a pre-built query against a *configured* endpoint. It never
accepts model-controlled endpoints, timeouts or response limits. Transient
upstream failures (502/503/504 and client read timeouts) are retried once by
default; the model never controls retries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.errors import (
    OverpassBadResponseError,
    OverpassError,
    OverpassRateLimitedError,
    OverpassTimeoutError,
    OverpassUpstreamError,
)
from app.osm.contracts import OverpassElement, OverpassResponse

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({502, 503, 504})
_HTML_RE = re.compile(r"<\s*/?\s*(html|body|head|title|div|p|br|pre)\b", re.I)


class HttpOverpassClient:
    """POST Overpass QL to a fixed interpreter URL with size and timeout guards."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        max_response_bytes: int,
        user_agent: str = "Ariadne-Thread/0.1 (research)",
        max_attempts: int = 2,
        retry_backoff_seconds: float = 1.5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_response_bytes < 1024:
            raise ValueError("max_response_bytes must be at least 1024")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_backoff_seconds <= 0:
            raise ValueError("retry_backoff_seconds must be positive")
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._max_attempts = max_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
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
            raise OverpassError("Overpass query must not be empty", attempts=0)

        last_error: OverpassError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._run_once(query, attempt=attempt)
                if attempt > 1:
                    logger.info(
                        "overpass_attempt=%s status=200 final=ok endpoint=%s",
                        attempt,
                        self._base_url,
                    )
                return response
            except OverpassError as exc:
                last_error = exc
                retryable = self._is_retryable(exc) and attempt < self._max_attempts
                logger.info(
                    "overpass_attempt=%s status=%s retryable=%s code=%s",
                    attempt,
                    exc.upstream_status if exc.upstream_status is not None else "n/a",
                    retryable,
                    exc.code,
                )
                if not retryable:
                    raise
                await asyncio.sleep(self._retry_backoff_seconds)

        assert last_error is not None
        raise last_error

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _run_once(self, query: str, *, attempt: int) -> OverpassResponse:
        try:
            async with self._client.stream(
                "POST",
                self._base_url,
                data={"data": query},
            ) as response:
                body = await self._read_limited(response)
                status = response.status_code
        except OverpassError as exc:
            raise type(exc)(
                exc.message,
                upstream_status=exc.upstream_status,
                attempts=attempt,
            ) from exc
        except httpx.TimeoutException as exc:
            raise OverpassTimeoutError(
                f"Overpass request timed out after {self._timeout_seconds:g}s",
                attempts=attempt,
            ) from exc
        except httpx.HTTPError as exc:
            raise OverpassUpstreamError(
                "Overpass request failed due to a transport error",
                attempts=attempt,
            ) from exc

        self._raise_for_status(status, body, attempts=attempt)
        payload = self._parse_json(body, attempts=attempt)
        elements, warnings = self._extract_elements(payload, attempts=attempt)
        return OverpassResponse(
            elements=elements,
            query=query,
            endpoint=self._base_url,
            retrieved_at=datetime.now(tz=timezone.utc),
            response_bytes=len(body),
            truncated=False,
            warnings=warnings,
            attempts=attempt,
        )

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
            raise OverpassTimeoutError(
                f"Overpass request timed out after {self._timeout_seconds:g}s"
            ) from exc
        return b"".join(chunks)

    def _raise_for_status(self, status: int, body: bytes, *, attempts: int) -> None:
        if status == 429:
            raise OverpassRateLimitedError(
                "Overpass rate limited the request (HTTP 429)",
                upstream_status=429,
                attempts=attempts,
            )
        if status == 504:
            raise OverpassTimeoutError(
                "The configured Overpass service returned HTTP 504",
                upstream_status=504,
                attempts=attempts,
            )
        if status in _RETRYABLE_STATUS:
            raise OverpassUpstreamError(
                f"The configured Overpass service returned HTTP {status}",
                upstream_status=status,
                attempts=attempts,
            )
        if 400 <= status < 500:
            raise OverpassBadResponseError(
                f"Overpass client error HTTP {status}",
                upstream_status=status,
                attempts=attempts,
            )
        if status >= 500:
            raise OverpassUpstreamError(
                f"The configured Overpass service returned HTTP {status}",
                upstream_status=status,
                attempts=attempts,
            )
        # Successful statuses still reject HTML bodies (gateway soft-fail pages).
        if _looks_like_html(body):
            raise OverpassBadResponseError(
                "Overpass returned an HTML error page instead of JSON",
                upstream_status=status,
                attempts=attempts,
            )

    def _parse_json(self, body: bytes, *, attempts: int) -> dict[str, Any]:
        if not body.strip():
            raise OverpassBadResponseError(
                "Overpass returned an empty body",
                attempts=attempts,
            )
        if _looks_like_html(body):
            raise OverpassBadResponseError(
                "Overpass returned an HTML error page instead of JSON",
                attempts=attempts,
            )
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OverpassBadResponseError(
                "Overpass returned malformed JSON",
                attempts=attempts,
            ) from exc
        if not isinstance(payload, dict):
            raise OverpassBadResponseError(
                "Overpass JSON root must be an object",
                attempts=attempts,
            )
        return payload

    def _extract_elements(
        self,
        payload: dict[str, Any],
        *,
        attempts: int,
    ) -> tuple[tuple[OverpassElement, ...], tuple[str, ...]]:
        warnings: list[str] = []
        remark = payload.get("remark")
        if isinstance(remark, str) and remark.strip():
            text = remark.strip()
            if "error" in text.lower():
                lowered = text.lower()
                if "timed out" in lowered or "timeout" in lowered:
                    raise OverpassTimeoutError(
                        "Overpass reported a query timeout",
                        attempts=attempts,
                    )
                raise OverpassUpstreamError(
                    "Overpass reported an error in the JSON remark",
                    attempts=attempts,
                )
            warnings.append(f"Overpass remark: {text}")

        raw_elements = payload.get("elements", [])
        if raw_elements is None:
            raw_elements = []
        if not isinstance(raw_elements, list):
            raise OverpassBadResponseError(
                "Overpass JSON 'elements' must be an array",
                attempts=attempts,
            )

        elements: list[OverpassElement] = []
        for entry in raw_elements:
            if isinstance(entry, dict):
                elements.append(entry)
            else:
                warnings.append("Skipped a non-object Overpass element")
        return tuple(elements), tuple(warnings)

    @staticmethod
    def _is_retryable(exc: OverpassError) -> bool:
        if isinstance(exc, OverpassRateLimitedError | OverpassBadResponseError):
            return False
        if isinstance(exc, OverpassTimeoutError):
            return True
        if isinstance(exc, OverpassUpstreamError):
            return exc.upstream_status in _RETRYABLE_STATUS or exc.upstream_status is None
        return False


def _looks_like_html(body: bytes) -> bool:
    sample = body[:800].decode("utf-8", errors="replace").lstrip().lower()
    if sample.startswith("<!doctype html") or sample.startswith("<html"):
        return True
    return bool(_HTML_RE.search(sample[:400]))
