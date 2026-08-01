"""Prompted JSON tool-call protocol.

``/api/tags`` for the configured local model reports ``completion`` and
``thinking`` without advertising ``tools``. Prompted mode is therefore the safe
default: tools are described in the system prompt and the model answers with a
JSON object, which is parsed back into :class:`ToolCall` values. Native Ollama
tool calling remains available for models that support it.

It also strips the model's hidden reasoning. Reasoning is discarded here and
never reaches the orchestrator, the trace, the API, or the database.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.llm.contracts import ToolCall, ToolDefinition

_REASONING_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_REASONING = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)
_FENCED_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

TOOL_CALL_KEY = "tool_calls"
FINAL_ANSWER_KEY = "final_answer"


def strip_reasoning(text: str) -> str:
    """Remove hidden chain-of-thought blocks from raw model output."""
    without_blocks = _REASONING_BLOCK.sub("", text)
    return _UNCLOSED_REASONING.sub("", without_blocks).strip()


@dataclass(frozen=True, slots=True)
class ParsedReply:
    """Result of interpreting one raw model reply."""

    content: str
    tool_calls: tuple[ToolCall, ...]


def render_tool_instructions(tools: list[ToolDefinition]) -> str:
    """Build the system-prompt fragment that defines the JSON tool protocol."""
    specs = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        }
        for tool in tools
    ]
    catalogue = json.dumps(specs, ensure_ascii=False, indent=2)
    return (
        "You can call tools. The available tools are:\n"
        f"{catalogue}\n\n"
        "To call tools, reply with ONLY a JSON object of this exact shape:\n"
        '{"tool_calls": [{"name": "<tool name>", "arguments": {...}}]}\n'
        "To answer the user, reply with ONLY a JSON object of this exact shape:\n"
        '{"final_answer": "<your answer>"}\n'
        "Never invent tool names. Never invent map data: geographic features may "
        "only come from the query_osm tool, and tagging guidance only from the "
        "search_osm_knowledge tool."
    )


def _candidate_payloads(text: str) -> list[str]:
    """Return JSON-looking fragments of a reply, most specific first."""
    candidates = [block.strip() for block in _FENCED_BLOCK.findall(text)]
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
        start, end = stripped.find("{"), stripped.rfind("}")
        if 0 <= start < end:
            candidates.append(stripped[start : end + 1])
    return [candidate for candidate in candidates if candidate]


def _coerce_arguments(raw: Any) -> dict[str, Any] | None:
    """Accept arguments as an object or as a JSON-encoded string."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(decoded, dict):
            return decoded
    if raw is None:
        return {}
    return None


def _tool_calls_from_payload(payload: dict[str, Any], id_prefix: str) -> tuple[ToolCall, ...]:
    raw_calls = payload.get(TOOL_CALL_KEY)
    if not isinstance(raw_calls, list):
        return ()
    calls: list[ToolCall] = []
    for index, entry in enumerate(raw_calls):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        arguments = _coerce_arguments(entry.get("arguments"))
        if not isinstance(name, str) or not name or arguments is None:
            continue
        calls.append(ToolCall(id=f"{id_prefix}{index + 1}", name=name, arguments=arguments))
    return tuple(calls)


def parse_reply(raw_content: str, *, id_prefix: str = "call_") -> ParsedReply:
    """Interpret one raw model reply.

    Returns tool calls when the model emitted a valid ``tool_calls`` payload,
    otherwise the visible answer text (unwrapping ``final_answer`` if present).
    Malformed JSON is not an error: it is treated as plain prose so the loop can
    still make progress.
    """
    visible = strip_reasoning(raw_content)
    for candidate in _candidate_payloads(visible):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        calls = _tool_calls_from_payload(payload, id_prefix)
        if calls:
            return ParsedReply(content="", tool_calls=calls)
        final = payload.get(FINAL_ANSWER_KEY)
        if isinstance(final, str):
            return ParsedReply(content=final.strip(), tool_calls=())
    return ParsedReply(content=visible, tool_calls=())
