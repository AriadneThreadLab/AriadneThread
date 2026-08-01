"""Ollama provider, exercised against an in-process transport (no network)."""

from __future__ import annotations

import json

import httpx
import pytest
from app.core.errors import LLMError, LLMProtocolError, LLMTimeoutError
from app.llm.contracts import ChatMessage, LLMOptions, ToolDefinition
from app.llm.ollama import OllamaConfig, OllamaProvider

_TOOLS = [
    ToolDefinition(
        name="query_osm",
        description="live OSM features",
        parameters_schema={"type": "object", "properties": {}},
    )
]


def _provider(handler, **overrides) -> OllamaProvider:
    config = OllamaConfig(
        base_url="http://127.0.0.1:11434",
        model="deepseek-r1:7b",
        num_ctx=4096,
        request_timeout_seconds=5.0,
        **overrides,
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=config.base_url,
        timeout=httpx.Timeout(config.request_timeout_seconds),
    )
    return OllamaProvider(config, client=client)


def _reply(content: str, **extra) -> httpx.Response:
    body = {"model": "deepseek-r1:7b", "message": {"role": "assistant", "content": content}}
    body.update(extra)
    return httpx.Response(200, json=body)


async def test_prompted_mode_injects_tool_instructions_into_the_system_prompt():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _reply('{"final_answer": "done"}')

    provider = _provider(handler)
    response = await provider.chat(
        [ChatMessage(role="system", content="Base rules."), ChatMessage(role="user", content="hi")],
        tools=_TOOLS,
    )
    await provider.aclose()

    payload = captured["payload"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert "Base rules." in messages[0]["content"]
    assert "query_osm" in messages[0]["content"]
    assert "tools" not in payload
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] == 4096
    assert response.content == "done"


async def test_prompted_mode_parses_tool_calls_and_strips_reasoning():
    raw = (
        "<think>internal deliberation</think>"
        '{"tool_calls": [{"name": "query_osm", "arguments": {"place": "Berlin"}}]}'
    )
    provider = _provider(lambda request: _reply(raw))
    response = await provider.chat([ChatMessage(role="user", content="parks?")], tools=_TOOLS)
    await provider.aclose()

    assert response.has_tool_calls
    assert response.tool_calls[0].name == "query_osm"
    assert "internal deliberation" not in response.content


async def test_native_mode_sends_tool_definitions_and_reads_tool_calls():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "some-tool-model",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "query_osm", "arguments": {"place": "Berlin"}}}
                    ],
                },
            },
        )

    provider = _provider(handler, tool_call_mode="native")
    response = await provider.chat([ChatMessage(role="user", content="parks?")], tools=_TOOLS)
    await provider.aclose()

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["tools"][0]["function"]["name"] == "query_osm"
    assert response.tool_calls[0].arguments == {"place": "Berlin"}


async def test_options_override_defaults():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _reply("ok")

    provider = _provider(handler)
    await provider.chat(
        [ChatMessage(role="user", content="hi")],
        options=LLMOptions(temperature=0.0, num_ctx=2048, max_tokens=256, stop=("</s>",)),
    )
    await provider.aclose()

    options = captured["payload"]["options"]  # type: ignore[index]
    assert options == {
        "num_ctx": 2048,
        "temperature": 0.0,
        "num_predict": 256,
        "stop": ["</s>"],
    }


async def test_without_tools_the_reply_is_plain_text():
    provider = _provider(lambda request: _reply("<think>hmm</think>Parks are leisure=park."))
    response = await provider.chat([ChatMessage(role="user", content="tag for park?")])
    await provider.aclose()
    assert response.content == "Parks are leisure=park."
    assert not response.has_tool_calls


async def test_http_error_becomes_a_domain_error():
    provider = _provider(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(LLMError, match="HTTP 500"):
        await provider.chat([ChatMessage(role="user", content="hi")])
    await provider.aclose()


async def test_timeout_becomes_a_domain_timeout_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider = _provider(handler)
    with pytest.raises(LLMTimeoutError):
        await provider.chat([ChatMessage(role="user", content="hi")])
    await provider.aclose()


async def test_unexpected_body_becomes_a_protocol_error():
    provider = _provider(lambda request: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(LLMProtocolError):
        await provider.chat([ChatMessage(role="user", content="hi")])
    await provider.aclose()


async def test_list_models_reads_installed_tags():
    provider = _provider(
        lambda request: httpx.Response(200, json={"models": [{"name": "deepseek-r1:7b"}]})
    )
    assert await provider.list_models() == ("deepseek-r1:7b",)
    await provider.aclose()
