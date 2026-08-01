"""Ollama implementation of :class:`~app.llm.contracts.LLMProvider`.

This is the only module allowed to know about Ollama's HTTP shape. It supports
both tool-calling paths:

* ``native``   - Ollama's ``tools`` request field and ``message.tool_calls``.
* ``prompted`` - a JSON protocol injected into the system prompt, required for
  models such as ``deepseek-r1:7b`` that do not advertise the ``tools``
  capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx

from app.core.errors import LLMError, LLMProtocolError, LLMTimeoutError
from app.llm.contracts import (
    ChatMessage,
    LLMOptions,
    LLMResponse,
    ToolCall,
    ToolDefinition,
)
from app.llm.tool_protocol import parse_reply, render_tool_instructions, strip_reasoning

ToolCallMode = Literal["native", "prompted"]

_CHAT_PATH = "/api/chat"
_TAGS_PATH = "/api/tags"


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    """Connection and default generation settings for one Ollama model."""

    base_url: str
    model: str
    num_ctx: int
    request_timeout_seconds: float
    tool_call_mode: ToolCallMode = "prompted"
    temperature: float = 0.1


class OllamaProvider:
    """Chat provider backed by a local Ollama server.

    The HTTP client is created eagerly but holds no connections until first use;
    call :meth:`aclose` (wired to the application lifespan) to release it.
    """

    def __init__(self, config: OllamaConfig, *, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(config.request_timeout_seconds),
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDefinition] | None = None,
        options: LLMOptions | None = None,
    ) -> LLMResponse:
        prompted = self._config.tool_call_mode == "prompted" and bool(tools)
        outgoing = self._with_tool_instructions(messages, tools) if prompted else messages
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [self._encode_message(message) for message in outgoing],
            "stream": False,
            "options": self._encode_options(options),
        }
        if tools and self._config.tool_call_mode == "native":
            payload["tools"] = [self._encode_tool(tool) for tool in tools]

        body = await self._post(_CHAT_PATH, payload)
        return self._decode_response(body, prompted=prompted)

    async def list_models(self) -> tuple[str, ...]:
        """Return the model tags installed on the server."""
        body = await self._get(_TAGS_PATH)
        models = body.get("models")
        if not isinstance(models, list):
            raise LLMProtocolError("Ollama /api/tags returned no model list")
        return tuple(
            str(entry["name"]) for entry in models if isinstance(entry, dict) and "name" in entry
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # --- internals ---

    def _with_tool_instructions(
        self, messages: list[ChatMessage], tools: list[ToolDefinition] | None
    ) -> list[ChatMessage]:
        if not tools:
            return messages
        instructions = render_tool_instructions(tools)
        head, *rest = messages if messages else [ChatMessage(role="system", content="")]
        if head.role == "system":
            merged = f"{head.content}\n\n{instructions}".strip()
            return [ChatMessage(role="system", content=merged), *rest]
        return [ChatMessage(role="system", content=instructions), *messages]

    def _encode_options(self, options: LLMOptions | None) -> dict[str, Any]:
        encoded: dict[str, Any] = {
            "num_ctx": self._config.num_ctx,
            "temperature": self._config.temperature,
        }
        if options is None:
            return encoded
        if options.temperature is not None:
            encoded["temperature"] = options.temperature
        if options.num_ctx is not None:
            encoded["num_ctx"] = options.num_ctx
        if options.max_tokens is not None:
            encoded["num_predict"] = options.max_tokens
        if options.stop:
            encoded["stop"] = list(options.stop)
        return encoded

    @staticmethod
    def _encode_message(message: ChatMessage) -> dict[str, Any]:
        # Ollama has no dedicated tool-result role in prompted mode; observations
        # are delivered as user-visible content tagged with the tool name.
        role = "tool" if message.role == "tool" else message.role
        encoded: dict[str, Any] = {"role": role, "content": message.content}
        if message.name:
            encoded["name"] = message.name
        return encoded

    @staticmethod
    def _encode_tool(tool: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            },
        }

    def _decode_response(self, body: dict[str, Any], *, prompted: bool) -> LLMResponse:
        message = body.get("message")
        if not isinstance(message, dict):
            raise LLMProtocolError("Ollama response contained no message object")
        raw_content = str(message.get("content") or "")
        model = str(body.get("model") or self._config.model)
        finish_reason = body.get("done_reason")

        native_calls = self._decode_native_tool_calls(message)
        if native_calls:
            return LLMResponse(
                content=strip_reasoning(raw_content),
                tool_calls=native_calls,
                model=model,
                finish_reason=str(finish_reason) if finish_reason else None,
            )
        if prompted:
            parsed = parse_reply(raw_content)
            return LLMResponse(
                content=parsed.content,
                tool_calls=parsed.tool_calls,
                model=model,
                finish_reason=str(finish_reason) if finish_reason else None,
            )
        return LLMResponse(
            content=strip_reasoning(raw_content),
            model=model,
            finish_reason=str(finish_reason) if finish_reason else None,
        )

    @staticmethod
    def _decode_native_tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            return ()
        calls: list[ToolCall] = []
        for index, entry in enumerate(raw_calls):
            function = entry.get("function") if isinstance(entry, dict) else None
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, dict):
                continue
            calls.append(
                ToolCall(
                    id=str(entry.get("id") or f"call_{index + 1}"),
                    name=name,
                    arguments=dict(arguments),
                )
            )
        return tuple(calls)

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Ollama request timed out after {self._config.request_timeout_seconds}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Ollama returned HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        except ValueError as exc:
            raise LLMProtocolError("Ollama returned a non-JSON body") from exc
        if not isinstance(body, dict):
            raise LLMProtocolError("Ollama returned an unexpected JSON payload")
        return body

    async def _get(self, path: str) -> dict[str, Any]:
        try:
            response = await self._client.get(path)
            response.raise_for_status()
            body = response.json()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Ollama request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        except ValueError as exc:
            raise LLMProtocolError("Ollama returned a non-JSON body") from exc
        if not isinstance(body, dict):
            raise LLMProtocolError("Ollama returned an unexpected JSON payload")
        return body
