"""AvalAI OpenAI-compatible provider, mocked at HTTP (no network, no real key)."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from app.core.errors import (
    AvalAIAuthenticationError,
    AvalAIBillingError,
    AvalAIError,
    AvalAIInvalidModelError,
    AvalAIInvalidResponseError,
    AvalAIRateLimitError,
    AvalAIUnavailableError,
    ConfigurationError,
    LLMTimeoutError,
)
from app.llm.avalai import AvalAIConfig, AvalAIProvider
from app.llm.contracts import ChatMessage, ToolDefinition
from app.llm.factory import build_avalai_provider, build_llm_provider, build_ollama_provider
from app.llm.ollama import OllamaProvider
from app.llm.tool_schema import normalize_tool_parameters_schema
from app.tools.search_osm_knowledge import SearchOsmKnowledgeArgs
from openai import AsyncOpenAI

_API_KEY = "sk-test-avalai-secret-do-not-leak"
_MODEL = "gemini-3.6-flash"
_TOOLS = [
    ToolDefinition(
        name="search_osm_knowledge",
        description="Search OSM Wiki documentation.",
        parameters_schema=SearchOsmKnowledgeArgs.model_json_schema(),
    )
]


def _completion(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, object]] | None = None,
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    return {
        "id": "chatcmpl-test-1",
        "object": "chat.completion",
        "created": 1,
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": message,
            }
        ],
        "usage": usage or {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def _tool_call_payload(
    name: str, arguments: dict[str, object], *, call_id: str
) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _provider(handler, **overrides) -> AvalAIProvider:
    config = AvalAIConfig(
        api_key=overrides.pop("api_key", _API_KEY),
        base_url="https://api.avalai.ir/v1",
        model=overrides.pop("model", _MODEL),
        request_timeout_seconds=5.0,
        tool_call_mode=overrides.pop("tool_call_mode", "native"),
        max_attempts=overrides.pop("max_attempts", 2),
        retry_backoff_seconds=0.01,
    )
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=config.base_url,
    )
    return AvalAIProvider(config, http_client=http)


def _settings(**overrides: object):
    from app.core.config import Settings

    base: dict[str, object] = {
        "database_url": "postgresql+asyncpg://user:pass@127.0.0.1:5433/osm_geoagent",
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


async def test_provider_selection_ollama_is_the_default():
    settings = _settings()
    provider = build_llm_provider(settings)
    assert isinstance(provider, OllamaProvider)
    assert provider.provider_name == "ollama"
    assert provider.model_name == "deepseek-r1:7b"
    await provider.aclose()


async def test_provider_selection_avalai():
    settings = _settings(
        llm_provider="avalai",
        avalai_api_key=_API_KEY,
        avalai_model=_MODEL,
    )
    provider = build_llm_provider(settings)
    assert provider.provider_name == "avalai"
    assert provider.model_name == _MODEL
    await provider.aclose()


def test_missing_avalai_api_key_fails_settings_validation():
    with pytest.raises(Exception, match="AVALAI_API_KEY"):
        _settings(llm_provider="avalai")


def test_build_avalai_provider_requires_a_key():
    settings = _settings()
    with pytest.raises(ConfigurationError, match="AVALAI_API_KEY"):
        build_avalai_provider(settings)


def test_ollama_builder_still_constructs_the_local_provider():
    provider = build_ollama_provider(_settings())
    assert isinstance(provider, OllamaProvider)


async def test_basic_completion_parsing_and_usage(caplog):
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json=_completion(content="Ariadne AvalAI connection OK"),
        )

    caplog.set_level(logging.INFO)
    provider = _provider(handler)
    response = await provider.chat(
        [ChatMessage(role="user", content="Reply with exactly: Ariadne AvalAI connection OK")]
    )
    await provider.aclose()

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == _MODEL
    assert set(payload.keys()) <= {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "stop",
        "tools",
        "tool_choice",
        "response_format",
    }
    assert "format" not in payload
    assert "think" not in payload
    assert "options" not in payload
    assert "num_ctx" not in payload
    assert "tools" not in payload
    assert response.content == "Ariadne AvalAI connection OK"
    assert response.model == _MODEL
    assert response.usage is not None
    assert response.usage.prompt_tokens == 11
    assert response.usage.total_tokens == 18
    assert response.usage.provider_request_id == "chatcmpl-test-1"
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert _API_KEY not in joined
    assert "provider=avalai" in joined
    assert f"model={_MODEL}" in joined
    assert "endpoint_host=api.avalai.ir" in joined
    assert "prompt_tokens=11" in joined


async def test_native_single_tool_call_parsing():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_completion(
                tool_calls=[
                    _tool_call_payload(
                        "search_osm_knowledge",
                        {"query": "public park OSM tag"},
                        call_id="call_1",
                    )
                ]
            ),
        )

    provider = _provider(handler)
    response = await provider.chat(
        [ChatMessage(role="user", content="What tag is a public park?")],
        tools=_TOOLS,
    )
    await provider.aclose()

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["tools"][0]["function"]["name"] == "search_osm_knowledge"
    expected_schema = normalize_tool_parameters_schema(SearchOsmKnowledgeArgs.model_json_schema())
    assert payload["tools"][0]["function"]["parameters"] == expected_schema
    assert payload["tool_choice"] == "auto"
    assert "format" not in payload
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "search_osm_knowledge"
    assert response.tool_calls[0].arguments == {"query": "public park OSM tag"}


async def test_multiple_native_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion(
                tool_calls=[
                    _tool_call_payload(
                        "resolve_place",
                        {"query": "University of Tehran"},
                        call_id="a",
                    ),
                    _tool_call_payload(
                        "resolve_place",
                        {"query": "Sharif University of Technology"},
                        call_id="b",
                    ),
                ]
            ),
        )

    provider = _provider(handler)
    response = await provider.chat(
        [ChatMessage(role="user", content="resolve both campuses")],
        tools=_TOOLS,
    )
    await provider.aclose()
    assert [call.id for call in response.tool_calls] == ["a", "b"]
    assert [call.arguments["query"] for call in response.tool_calls] == [
        "University of Tehran",
        "Sharif University of Technology",
    ]


async def test_tool_result_round_trip_uses_openai_tool_role():
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.append(payload)
        if len(captured) == 1:
            return httpx.Response(
                200,
                json=_completion(
                    tool_calls=[
                        _tool_call_payload(
                            "search_osm_knowledge",
                            {"query": "park"},
                            call_id="call_1",
                        )
                    ]
                ),
            )
        return httpx.Response(200, json=_completion(content="Parks are leisure=park."))

    provider = _provider(handler)
    first = await provider.chat(
        [ChatMessage(role="user", content="park tag?")],
        tools=_TOOLS,
    )
    follow_up = [
        ChatMessage(role="user", content="park tag?"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=first.tool_calls,
        ),
        ChatMessage(
            role="tool",
            content="status=ok leisure=park",
            name="search_osm_knowledge",
            tool_call_id="call_1",
        ),
    ]
    second = await provider.chat(follow_up, tools=_TOOLS)
    await provider.aclose()

    assert second.content == "Parks are leisure=park."
    replay = captured[1]["messages"]
    assert isinstance(replay, list)
    assistant = next(message for message in replay if message["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "call_1"
    tool = next(message for message in replay if message["role"] == "tool")
    assert tool["tool_call_id"] == "call_1"
    assert "GeoJSON" not in json.dumps(replay)


async def test_prompted_fallback_when_native_tools_are_rejected(caplog):
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.append(payload)
        if "tools" in payload:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "tools is not supported for this model",
                        "type": "invalid_request_error",
                    }
                },
            )
        return httpx.Response(
            200,
            json=_completion(content='{"final_answer":"done"}'),
        )

    caplog.set_level(logging.WARNING)
    provider = _provider(handler)
    response = await provider.chat(
        [ChatMessage(role="user", content="hi")],
        tools=_TOOLS,
    )
    await provider.aclose()

    assert response.content == "done"
    assert provider.tool_call_mode == "prompted_fallback"
    assert "tools" in captured[0]
    assert "tools" not in captured[1]
    assert captured[1]["response_format"] == {"type": "json_object"}
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert "tool_mode=prompted_fallback" in joined or "prompted_fallback" in joined
    assert _API_KEY not in joined


async def test_authentication_failure_mapping():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "Incorrect API key provided",
                    "type": "invalid_request_error",
                }
            },
        )

    provider = _provider(handler)
    with pytest.raises(AvalAIAuthenticationError):
        await provider.chat([ChatMessage(role="user", content="hi")])
    await provider.aclose()


async def test_rate_limit_mapping():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
        )

    provider = _provider(handler, max_attempts=1)
    with pytest.raises(AvalAIRateLimitError):
        await provider.chat([ChatMessage(role="user", content="hi")])
    await provider.aclose()


async def test_timeout_mapping():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider = _provider(handler, max_attempts=1)
    with pytest.raises(LLMTimeoutError):
        await provider.chat([ChatMessage(role="user", content="hi")])
    await provider.aclose()


async def test_malformed_provider_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "x", "choices": []})

    provider = _provider(handler)
    with pytest.raises(AvalAIInvalidResponseError):
        await provider.chat([ChatMessage(role="user", content="hi")])
    await provider.aclose()


async def test_billing_failure_mapping():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            json={"error": {"message": "insufficient_quota", "type": "insufficient_quota"}},
        )

    provider = _provider(handler, max_attempts=1)
    with pytest.raises((AvalAIBillingError, AvalAIError)):
        await provider.chat([ChatMessage(role="user", content="hi")])
    await provider.aclose()


async def test_invalid_model_mapping():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "error": {
                    "message": "The model does not exist",
                    "type": "invalid_request_error",
                }
            },
        )

    provider = _provider(handler, max_attempts=1)
    with pytest.raises(AvalAIInvalidModelError):
        await provider.chat([ChatMessage(role="user", content="hi")])
    await provider.aclose()


async def test_bounded_retry_on_unavailable_then_success():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json=_completion(content="recovered"))

    provider = _provider(handler, max_attempts=2)
    response = await provider.chat([ChatMessage(role="user", content="hi")])
    await provider.aclose()
    assert response.content == "recovered"
    assert calls["n"] == 2


async def test_auth_failures_are_not_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    provider = _provider(handler, max_attempts=3)
    with pytest.raises(AvalAIAuthenticationError):
        await provider.chat([ChatMessage(role="user", content="hi")])
    await provider.aclose()
    assert calls["n"] == 1


async def test_api_key_never_appears_in_logs(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": f"invalid key {_API_KEY}", "type": "invalid_request_error"}},
        )

    caplog.set_level(logging.DEBUG)
    provider = _provider(handler)
    with pytest.raises(AvalAIAuthenticationError):
        await provider.chat([ChatMessage(role="user", content="hi")])
    await provider.aclose()
    text = "\n".join(record.getMessage() for record in caplog.records)
    text += "\n".join(str(record.exc_text or "") for record in caplog.records)
    assert _API_KEY not in text


async def test_json_mode_uses_response_format_not_ollama_fields():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_completion(content='{"analysis_type":"comparison"}'),
        )

    from app.llm.contracts import LLMOptions

    provider = _provider(handler)
    await provider.chat(
        [ChatMessage(role="user", content="plan")],
        options=LLMOptions(json_mode=True, temperature=0.1, max_tokens=400),
    )
    await provider.aclose()
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 400
    assert "format" not in payload
    assert "num_ctx" not in payload


async def test_injected_client_is_not_recreated_per_turn():
    created = {"n": 0}

    class CountingClient(AsyncOpenAI):
        def __init__(self, *args, **kwargs):
            created["n"] += 1
            super().__init__(*args, **kwargs)

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_completion(content="ok"))
        ),
        base_url="https://api.avalai.ir/v1",
    )
    client = CountingClient(
        api_key=_API_KEY,
        base_url="https://api.avalai.ir/v1",
        max_retries=0,
        http_client=http,
    )
    provider = AvalAIProvider(
        AvalAIConfig(
            api_key=_API_KEY,
            base_url="https://api.avalai.ir/v1",
            model=_MODEL,
            request_timeout_seconds=5.0,
            max_attempts=1,
        ),
        client=client,
    )
    await provider.chat([ChatMessage(role="user", content="a")])
    await provider.chat([ChatMessage(role="user", content="b")])
    await http.aclose()
    assert created["n"] == 1


async def test_unavailable_exhausted_retries():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    provider = _provider(handler, max_attempts=2)
    with pytest.raises(AvalAIUnavailableError):
        await provider.chat([ChatMessage(role="user", content="hi")])
    await provider.aclose()
    assert calls["n"] == 2
