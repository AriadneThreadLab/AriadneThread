"""Controlled MediaWiki API client for the OSM wiki.

Fetches only explicitly requested page titles. There is no link following and
no search crawl.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.errors import WikiFetchError, WikiParseError


@dataclass(frozen=True, slots=True)
class FetchedWikiPage:
    """Raw page payload as returned by the MediaWiki API."""

    title: str
    wikitext: str
    page_id: int | None
    retrieved_at: datetime


class MediaWikiClient:
    """Minimal async MediaWiki client with timeouts and bounded retries."""

    def __init__(
        self,
        *,
        api_url: str,
        user_agent: str,
        timeout_seconds: float,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_url = api_url
        self._max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"User-Agent": user_agent, "Accept": "application/json"},
        )

    async def fetch_page(self, title: str) -> FetchedWikiPage:
        """Fetch one page's main-slot wikitext by title."""
        if not title.strip():
            raise WikiFetchError("wiki page title must not be empty")

        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "redirects": "1",
            "titles": title,
        }

        body = await self._get_json(params)
        return self._decode_page(body, requested_title=title)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get_json(self, params: dict[str, str]) -> dict[str, Any]:
        attempts = self._max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = await self._client.get(self._api_url, params=params)
            except httpx.TimeoutException as exc:
                last_error = WikiFetchError(
                    f"OSM wiki request timed out for title {params.get('titles')!r}"
                )
                last_error.__cause__ = exc
                if attempt + 1 >= attempts:
                    raise last_error from exc
                continue
            except httpx.HTTPError as exc:
                last_error = WikiFetchError(f"OSM wiki request failed: {exc}")
                if attempt + 1 >= attempts:
                    raise last_error from exc
                continue

            if response.status_code in {429, 500, 502, 503, 504}:
                last_error = WikiFetchError(
                    f"OSM wiki returned HTTP {response.status_code} for "
                    f"title {params.get('titles')!r}"
                )
                if attempt + 1 >= attempts:
                    raise last_error
                continue

            if response.status_code >= 400:
                raise WikiFetchError(
                    f"OSM wiki returned HTTP {response.status_code} for "
                    f"title {params.get('titles')!r}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise WikiParseError("OSM wiki returned a non-JSON body") from exc
            if not isinstance(payload, dict):
                raise WikiParseError("OSM wiki returned an unexpected JSON payload")
            return payload

        assert last_error is not None
        raise last_error

    def _decode_page(self, body: dict[str, Any], *, requested_title: str) -> FetchedWikiPage:
        query = body.get("query")
        if not isinstance(query, dict):
            raise WikiParseError("MediaWiki response missing 'query'")

        pages = query.get("pages")
        if not isinstance(pages, list) or not pages:
            raise WikiParseError(f"MediaWiki response contained no pages for {requested_title!r}")

        page = pages[0]
        if not isinstance(page, dict):
            raise WikiParseError("MediaWiki page entry was not an object")
        if page.get("missing") is True or page.get("invalid") is True:
            raise WikiFetchError(f"OSM wiki page not found: {requested_title!r}")

        title = str(page.get("title") or requested_title)
        page_id_raw = page.get("pageid")
        page_id = int(page_id_raw) if isinstance(page_id_raw, int) else None

        revisions = page.get("revisions")
        if not isinstance(revisions, list) or not revisions:
            raise WikiParseError(f"MediaWiki page {title!r} has no revisions")

        revision = revisions[0]
        if not isinstance(revision, dict):
            raise WikiParseError(f"MediaWiki revision for {title!r} was not an object")

        slots = revision.get("slots")
        wikitext: str | None = None
        if isinstance(slots, dict):
            main = slots.get("main")
            if isinstance(main, dict):
                content = main.get("content")
                if isinstance(content, str):
                    wikitext = content
        # Older response shape fallback.
        if wikitext is None and isinstance(revision.get("content"), str):
            wikitext = revision["content"]

        if wikitext is None or not wikitext.strip():
            raise WikiParseError(f"MediaWiki page {title!r} has empty wikitext")

        return FetchedWikiPage(
            title=title,
            wikitext=wikitext,
            page_id=page_id,
            retrieved_at=datetime.now(tz=timezone.utc),
        )
