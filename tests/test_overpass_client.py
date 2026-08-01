"""HttpOverpassClient error mapping and response guards (offline)."""

from __future__ import annotations

import httpx
import pytest
from app.core.errors import OverpassError, ToolTimeoutError
from app.osm.client import HttpOverpassClient


def _client(
    handler: object,
    *,
    max_response_bytes: int = 10_000,
    timeout_seconds: float = 5.0,
) -> HttpOverpassClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    http = httpx.AsyncClient(transport=transport, base_url="https://overpass.test")
    return HttpOverpassClient(
        base_url="https://overpass.test/api/interpreter",
        timeout_seconds=timeout_seconds,
        max_response_bytes=max_response_bytes,
        client=http,
    )


async def test_successful_empty_elements_are_valid():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert b"data=" in request.content
        return httpx.Response(200, json={"elements": []})

    client = _client(handler)
    try:
        response = await client.run('[out:json][timeout:5];node["amenity"="bench"];out;')
    finally:
        await client.aclose()

    assert response.elements == ()
    assert response.endpoint.endswith("/api/interpreter")
    assert response.warnings == ()


async def test_http_429_is_mapped():
    client = _client(lambda request: httpx.Response(429, text="slow down"))
    try:
        with pytest.raises(OverpassError, match="429"):
            await client.run("query")
    finally:
        await client.aclose()


async def test_http_4xx_is_mapped():
    client = _client(lambda request: httpx.Response(400, text="bad query"))
    try:
        with pytest.raises(OverpassError, match="HTTP 400"):
            await client.run("query")
    finally:
        await client.aclose()


async def test_http_5xx_is_mapped():
    client = _client(lambda request: httpx.Response(503, text="unavailable"))
    try:
        with pytest.raises(OverpassError, match="HTTP 503"):
            await client.run("query")
    finally:
        await client.aclose()


async def test_malformed_json_is_mapped():
    client = _client(lambda request: httpx.Response(200, text="not-json"))
    try:
        with pytest.raises(OverpassError, match="malformed JSON"):
            await client.run("query")
    finally:
        await client.aclose()


async def test_overpass_error_remark_is_mapped():
    client = _client(
        lambda request: httpx.Response(
            200,
            json={"remark": "runtime error: Query timed out", "elements": []},
        )
    )
    try:
        with pytest.raises(OverpassError, match="Overpass reported an error"):
            await client.run("query")
    finally:
        await client.aclose()


async def test_non_error_remark_becomes_a_warning():
    client = _client(
        lambda request: httpx.Response(
            200,
            json={"remark": "runtime note: slow", "elements": []},
        )
    )
    try:
        response = await client.run("query")
    finally:
        await client.aclose()
    assert any("slow" in warning for warning in response.warnings)


async def test_timeout_becomes_tool_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = _client(handler, timeout_seconds=1.0)
    try:
        with pytest.raises(ToolTimeoutError, match="timed out"):
            await client.run("query")
    finally:
        await client.aclose()


async def test_content_length_over_limit_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"elements":[]}',
            headers={"Content-Length": "20000"},
        )

    client = _client(handler, max_response_bytes=1024)
    try:
        with pytest.raises(OverpassError, match="Content-Length"):
            await client.run("query")
    finally:
        await client.aclose()


async def test_streamed_body_over_limit_is_rejected():
    async def stream():
        yield b"x" * 600
        yield b"x" * 600

    def handler(request: httpx.Request) -> httpx.Response:
        # No Content-Length so the client must enforce the limit while reading.
        return httpx.Response(200, content=stream())

    client = _client(handler, max_response_bytes=1024)
    try:
        with pytest.raises(OverpassError, match="exceeded limit"):
            await client.run("query")
    finally:
        await client.aclose()


async def test_empty_query_is_rejected():
    client = _client(lambda request: httpx.Response(200, json={"elements": []}))
    try:
        with pytest.raises(OverpassError, match="must not be empty"):
            await client.run("   ")
    finally:
        await client.aclose()
