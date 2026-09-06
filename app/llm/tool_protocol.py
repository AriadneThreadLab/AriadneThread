"""Prompted JSON tool-call protocol.

Prompted mode is the safe default for local models that may accept native
``tools`` at the HTTP layer without reliably emitting native tool calls.

Hidden reasoning is stripped here and never reaches the orchestrator, trace,
API, or database.

Allowed normalisation only:
* strip whitespace
* remove at most one complete outer Markdown fence (json or untyped)
* parse the entire remaining string as one JSON object
* coerce ``arguments`` from a JSON object string when it is exactly one object

There is no prose-fragment scraping.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.llm.contracts import ToolCall, ToolDefinition
from app.llm.prompted_catalogue import render_prompted_tool_instructions

_REASONING_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_REASONING = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)
# Exactly one outer fence: optional language tag (json), then body, then close.
_OUTER_FENCE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)
_ATTEMPTED_PROTOCOL = re.compile(
    r"("
    r"```|"
    r"\btool_calls\b|"
    r"\bfinal_answer\b|"
    r"\bsearch_osm_knowledge\b|"
    r"\bresolve_place\b|"
    r"\bquery_osm\b|"
    r"\banalyze_features\b|"
    r"\bsimbench_query\b|"
    r"\banalyze_energy_grid\b"
    r")",
    re.IGNORECASE,
)

TOOL_CALL_KEY = "tool_calls"
FINAL_ANSWER_KEY = "final_answer"
_ALLOWED_TOP_LEVEL = frozenset({TOOL_CALL_KEY, FINAL_ANSWER_KEY})

logger = logging.getLogger(__name__)


class PromptedProtocolError(ValueError):
    """The model reply was not a valid prompted tool protocol object."""

    def __init__(self, message: str, *, code: str = "llm_protocol_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def strip_reasoning(text: str) -> str:
    """Remove hidden chain-of-thought blocks from raw model output."""
    without_blocks = _REASONING_BLOCK.sub("", text)
    return _UNCLOSED_REASONING.sub("", without_blocks).strip()


def normalize_prompted_text(text: str) -> str:
    """Strip whitespace and at most one outer Markdown JSON fence."""
    stripped = text.strip()
    if not stripped:
        return ""
    match = _OUTER_FENCE.match(stripped)
    if match is not None:
        return match.group(1).strip()
    return stripped


@dataclass(frozen=True, slots=True)
class ParsedReply:
    """Result of interpreting one raw model reply."""

    content: str
    tool_calls: tuple[ToolCall, ...]
    protocol_error: str | None = None
    response_kind: str = "invalid"

    @property
    def is_protocol_error(self) -> bool:
        return self.protocol_error is not None


def render_tool_instructions(tools: list[ToolDefinition]) -> str:
    """Build the system-prompt fragment that defines the JSON tool protocol."""
    return render_prompted_tool_instructions(tools)


def _coerce_arguments(raw: Any) -> dict[str, Any] | None:
    """Accept arguments as an object or as a JSON-encoded string of one object."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(decoded, dict):
            return decoded
        return None
    if raw is None:
        return {}
    return None


def _tool_calls_from_payload(payload: dict[str, Any], id_prefix: str) -> tuple[ToolCall, ...]:
    raw_calls = payload.get(TOOL_CALL_KEY)
    if not isinstance(raw_calls, list):
        raise PromptedProtocolError("'tool_calls' must be a JSON array")
    if not raw_calls:
        raise PromptedProtocolError("'tool_calls' must not be empty")
    calls: list[ToolCall] = []
    for index, entry in enumerate(raw_calls):
        if not isinstance(entry, dict):
            raise PromptedProtocolError(f"tool_calls[{index}] must be an object")
        # Reject native-style envelopes and wrong key names without guessing.
        if "function" in entry and "name" not in entry:
            raise PromptedProtocolError(
                f"tool_calls[{index}] must use name/arguments, not a native function envelope"
            )
        if "args" in entry and "arguments" not in entry:
            raise PromptedProtocolError(f"tool_calls[{index}] must use 'arguments' (not 'args')")
        if "parameters" in entry and "arguments" not in entry:
            raise PromptedProtocolError(
                f"tool_calls[{index}] must use 'arguments' (not 'parameters')"
            )
        name = entry.get("name")
        arguments = _coerce_arguments(entry.get("arguments"))
        if not isinstance(name, str) or not name.strip():
            raise PromptedProtocolError(f"tool_calls[{index}].name must be a non-empty string")
        if arguments is None:
            raise PromptedProtocolError(f"tool_calls[{index}].arguments must be a JSON object")
        calls.append(ToolCall(id=f"{id_prefix}{index + 1}", name=name.strip(), arguments=arguments))
    return tuple(calls)


def _looks_like_attempted_protocol(text: str) -> bool:
    if not text:
        return False
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("[") or stripped.startswith("```"):
        return True
    return _ATTEMPTED_PROTOCOL.search(text) is not None


def _loads_protocol_json(text: str) -> Any | None:
    """Parse one JSON value; allow a single over-escape repair for ``\\"``."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # deepseek-r1:7b sometimes emits {\\"key\\":\\"value\\"} inside already-JSON text.
    if '\\"' not in text:
        return None
    repaired = text.replace('\\"', '"')
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


_DIAGNOSTIC_PREFIX_CHARS = 400


def _classify_invalid_shape(visible: str, normalised: str, payload: Any | None) -> str:
    """Compact failure category for local diagnostics (no credentials/CoT)."""
    if not visible.strip():
        return "empty_after_reasoning_strip"
    if payload is None:
        if normalised.lstrip().startswith("```") or visible.lstrip().startswith("```"):
            return "markdown_fence_or_malformed_json"
        if normalised.lstrip().startswith("{") or normalised.lstrip().startswith("["):
            # Distinguish trailing prose after an otherwise JSON-looking object.
            decoder = json.JSONDecoder()
            try:
                _, end = decoder.raw_decode(normalised.lstrip())
                if normalised.lstrip()[end:].strip():
                    return "json_with_trailing_prose"
            except json.JSONDecodeError:
                pass
            return "malformed_json"
        if _ATTEMPTED_PROTOCOL.search(visible):
            return "prose_with_protocol_markers"
        return "prose_instead_of_json"
    if not isinstance(payload, dict):
        return "non_object_json"
    keys = set(payload)
    if keys & {
        "search_osm_knowledge",
        "query_osm",
        "analyze_features",
        "simbench_query",
        "analyze_energy_grid",
    }:
        return "tool_names_as_top_level_keys"
    if "tool_call" in keys and TOOL_CALL_KEY not in keys:
        return "singular_tool_call"
    if TOOL_CALL_KEY in keys and FINAL_ANSWER_KEY in keys:
        return "both_tool_calls_and_final_answer"
    if TOOL_CALL_KEY not in keys and FINAL_ANSWER_KEY not in keys:
        if "answer" in keys:
            return "wrong_envelope_answer_not_final_answer"
        return "wrong_top_level_envelope"
    return "invalid_tool_call_object"


def _log_invalid_diagnostic(visible: str, category: str) -> None:
    # Truncated sanitized visible prefix only — never system prompts or thinking.
    prefix = " ".join(visible.split())[:_DIAGNOSTIC_PREFIX_CHARS]
    logger.info(
        "prompted_protocol_invalid visible_output_shape=%s prefix=%s",
        category,
        prefix,
    )


def _log_parse_result(
    parsed: ParsedReply,
    *,
    visible: str = "",
    normalised: str = "",
    payload: Any | None = None,
) -> None:
    names = [call.name for call in parsed.tool_calls]
    logger.info(
        "prompted_protocol response_kind=%s tool_call_count=%s tool_names=%s protocol_error=%s",
        parsed.response_kind,
        len(parsed.tool_calls),
        names,
        parsed.protocol_error is not None,
    )
    if parsed.is_protocol_error:
        category = _classify_invalid_shape(visible, normalised, payload)
        _log_invalid_diagnostic(visible or normalised, category)


def parse_reply(raw_content: str, *, id_prefix: str = "call_") -> ParsedReply:
    """Interpret one raw model reply under the prompted protocol.

    Successful tool-call turns return empty ``content`` and never expose the
    protocol object. Successful final-answer turns return only the answer text.
    Structured-looking replies that fail validation become protocol errors
    rather than user-facing answers.
    """
    visible = strip_reasoning(raw_content)
    if not visible:
        parsed = ParsedReply(
            content="",
            tool_calls=(),
            protocol_error="model reply was empty after removing hidden reasoning",
            response_kind="invalid",
        )
        _log_parse_result(parsed, visible=visible)
        return parsed

    normalised = normalize_prompted_text(visible)
    if not normalised:
        parsed = ParsedReply(
            content="",
            tool_calls=(),
            protocol_error="model reply contained no prompted protocol payload",
            response_kind="invalid",
        )
        _log_parse_result(parsed, visible=visible, normalised=normalised)
        return parsed

    # Fence/prose mixtures: only a complete outer fence is normalised away.
    # Prose before/after a fence is rejected below (no fragment scraping).
    payload = _loads_protocol_json(normalised)
    if payload is None:
        if _looks_like_attempted_protocol(visible) or _looks_like_attempted_protocol(normalised):
            parsed = ParsedReply(
                content="",
                tool_calls=(),
                protocol_error=(
                    "model reply was not a single JSON tool_calls/final_answer object "
                    "(no prose; optional one outer ```json fence only)"
                ),
                response_kind="invalid",
            )
            _log_parse_result(parsed, visible=visible, normalised=normalised, payload=None)
            return parsed
        # Plain prose without protocol markers — soft final answer for non-tool turns.
        parsed = ParsedReply(content=normalised, tool_calls=(), response_kind="final_answer")
        _log_parse_result(parsed, visible=visible, normalised=normalised)
        return parsed

    if not isinstance(payload, dict):
        parsed = ParsedReply(
            content="",
            tool_calls=(),
            protocol_error="prompted protocol payload must be a JSON object",
            response_kind="invalid",
        )
        _log_parse_result(parsed, visible=visible, normalised=normalised, payload=payload)
        return parsed

    keys = set(payload)
    unknown = keys - _ALLOWED_TOP_LEVEL
    if unknown:
        # Common DeepSeek mistake: {"query_osm": {...}} instead of tool_calls.
        if unknown & {
            "search_osm_knowledge",
            "query_osm",
            "analyze_features",
            "simbench_query",
            "analyze_energy_grid",
        }:
            parsed = ParsedReply(
                content="",
                tool_calls=(),
                protocol_error=(
                    "prompted protocol must use "
                    '{"tool_calls":[{"name":"...","arguments":{...}}]} '
                    "not tool names as top-level keys"
                ),
                response_kind="invalid",
            )
            _log_parse_result(parsed, visible=visible, normalised=normalised, payload=payload)
            return parsed
        if "tool_call" in unknown and TOOL_CALL_KEY not in keys:
            parsed = ParsedReply(
                content="",
                tool_calls=(),
                protocol_error="use 'tool_calls' (plural), not 'tool_call'",
                response_kind="invalid",
            )
            _log_parse_result(parsed, visible=visible, normalised=normalised, payload=payload)
            return parsed
        parsed = ParsedReply(
            content="",
            tool_calls=(),
            protocol_error=(
                "prompted protocol payload has unknown top-level field(s): "
                + ", ".join(sorted(unknown))
            ),
            response_kind="invalid",
        )
        _log_parse_result(parsed, visible=visible, normalised=normalised, payload=payload)
        return parsed

    has_tools = TOOL_CALL_KEY in payload
    has_final = FINAL_ANSWER_KEY in payload
    if has_tools and has_final:
        parsed = ParsedReply(
            content="",
            tool_calls=(),
            protocol_error="prompted protocol payload must not include both "
            "tool_calls and final_answer",
            response_kind="invalid",
        )
        _log_parse_result(parsed, visible=visible, normalised=normalised, payload=payload)
        return parsed
    if not has_tools and not has_final:
        parsed = ParsedReply(
            content="",
            tool_calls=(),
            protocol_error="prompted protocol payload must include tool_calls or final_answer",
            response_kind="invalid",
        )
        _log_parse_result(parsed, visible=visible, normalised=normalised, payload=payload)
        return parsed

    if has_tools:
        try:
            calls = _tool_calls_from_payload(payload, id_prefix)
        except PromptedProtocolError as exc:
            parsed = ParsedReply(
                content="",
                tool_calls=(),
                protocol_error=exc.message,
                response_kind="invalid",
            )
            _log_parse_result(parsed, visible=visible, normalised=normalised, payload=payload)
            return parsed
        parsed = ParsedReply(content="", tool_calls=calls, response_kind="tool_calls")
        _log_parse_result(parsed, visible=visible, normalised=normalised, payload=payload)
        return parsed

    final = payload.get(FINAL_ANSWER_KEY)
    if not isinstance(final, str):
        parsed = ParsedReply(
            content="",
            tool_calls=(),
            protocol_error="'final_answer' must be a string",
            response_kind="invalid",
        )
        _log_parse_result(parsed, visible=visible, normalised=normalised, payload=payload)
        return parsed
    answer = final.strip()
    if not answer:
        parsed = ParsedReply(
            content="",
            tool_calls=(),
            protocol_error="'final_answer' must not be empty",
            response_kind="invalid",
        )
        _log_parse_result(parsed, visible=visible, normalised=normalised, payload=payload)
        return parsed
    parsed = ParsedReply(content=answer, tool_calls=(), response_kind="final_answer")
    _log_parse_result(parsed, visible=visible, normalised=normalised, payload=payload)
    return parsed


def recover_prompted_envelope(raw_content: str) -> ParsedReply | None:
    """Accept prompted JSON tool_calls/final_answer when native tool_calls are empty.

    Native providers often follow the system prompt and write the JSON envelope
    into ``content`` with ``finish_reason=stop`` instead of emitting API tool calls.
    """
    parsed = parse_reply(raw_content)
    if parsed.is_protocol_error:
        return None
    if parsed.tool_calls or parsed.response_kind == "final_answer":
        return parsed
    return None
