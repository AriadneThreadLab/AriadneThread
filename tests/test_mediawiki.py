"""MediaWiki client behaviour with mocked HTTP."""

from __future__ import annotations

import json

import httpx
import pytest
from app.core.errors import WikiFetchError, WikiParseError
from app.rag.mediawiki import MediaWikiClient


def _client(handler: object, *, max_retries: int = 1) -> MediaWikiClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    http = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(5.0),
        headers={"User-Agent": "OSM-GeoAgent-test"},
    )
    return MediaWikiClient(
        api_url="https://wiki.openstreetmap.org/w/api.php",
        user_agent="OSM-GeoAgent-test",
        timeout_seconds=5.0,
        max_retries=max_retries,
        client=http,
    )


def _ok_payload(title: str, wikitext: str) -> dict[str, object]:
    return {
        "query": {
            "pages": [
                {
                    "pageid": 1,
                    "title": title,
                    "revisions": [{"slots": {"main": {"content": wikitext}}}],
                }
            ]
        }
    }


async def test_fetch_page_reads_main_slot_wikitext():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["action"] == "query"
        assert request.url.params["titles"] == "Tag:leisure=park"
        assert "OSM-GeoAgent-test" in request.headers["User-Agent"]
        payload = _ok_payload("Tag:leisure=park", "== Description ==\nA park.")
        return httpx.Response(200, json=payload)

    client = _client(handler)
    page = await client.fetch_page("Tag:leisure=park")
    await client.aclose()
    assert page.title == "Tag:leisure=park"
    assert "A park." in page.wikitext
    assert page.retrieved_at.tzinfo is not None


async def test_timeout_becomes_wiki_fetch_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client = _client(handler, max_retries=0)
    with pytest.raises(WikiFetchError, match="timed out"):
        await client.fetch_page("Map_Features")
    await client.aclose()


async def test_http_429_retries_then_fails():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, text="rate limited")

    client = _client(handler, max_retries=1)
    with pytest.raises(WikiFetchError, match="429"):
        await client.fetch_page("Map_Features")
    await client.aclose()
    assert calls["n"] == 2


async def test_missing_page_is_a_fetch_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"query": {"pages": [{"title": "Nope", "missing": True}]}},
        )

    client = _client(handler)
    with pytest.raises(WikiFetchError, match="not found"):
        await client.fetch_page("Nope")
    await client.aclose()


async def test_malformed_json_is_a_parse_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json", headers={"Content-Type": "text/plain"})

    client = _client(handler)
    with pytest.raises(WikiParseError, match="non-JSON"):
        await client.fetch_page("Map_Features")
    await client.aclose()


async def test_empty_wikitext_is_a_parse_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_payload("Map_Features", "   "))

    client = _client(handler)
    with pytest.raises(WikiParseError, match="empty wikitext"):
        await client.fetch_page("Map_Features")
    await client.aclose()


async def test_server_error_payload_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _client(handler, max_retries=0)
    with pytest.raises(WikiFetchError, match="500"):
        await client.fetch_page("Map_Features")
    await client.aclose()


def test_ok_payload_helper_is_json_serialisable():
    # Guards the fixture shape used across ingest tests.
    assert json.loads(json.dumps(_ok_payload("Tag", "text")))["query"]["pages"][0]["title"] == "Tag"
