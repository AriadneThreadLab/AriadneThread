"""Safe planning-turn size diagnostics (no prompt body, no credentials)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.llm.contracts import ChatMessage, ToolDefinition


@dataclass(frozen=True, slots=True)
class PlanningDiagnostics:
    """Compact metrics for one LLM planning request."""

    system_prompt_chars: int
    rendered_tool_catalog_chars: int
    message_count: int
    total_message_chars: int
    eligible_tool_count: int
    estimated_tokens: int
    model: str
    num_ctx: int | None
    temperature: float | None


def estimate_tokens(char_count: int) -> int:
    """Cheap token estimate (~4 chars/token). Not a tokenizer."""
    return max(1, char_count // 4) if char_count else 0


def measure_planning_messages(
    messages: list[ChatMessage],
    *,
    tools: list[ToolDefinition] | None,
    tool_catalog_text: str,
    model: str,
    num_ctx: int | None = None,
    temperature: float | None = None,
) -> PlanningDiagnostics:
    system_chars = 0
    total = 0
    for message in messages:
        total += len(message.content or "")
        if message.role == "system":
            system_chars += len(message.content or "")
    catalog_chars = len(tool_catalog_text or "")
    # Prompted mode merges catalogue into the system message sent to Ollama.
    outbound_total = total + catalog_chars
    return PlanningDiagnostics(
        system_prompt_chars=system_chars,
        rendered_tool_catalog_chars=catalog_chars,
        message_count=len(messages),
        total_message_chars=outbound_total,
        eligible_tool_count=len(tools or []),
        estimated_tokens=estimate_tokens(outbound_total),
        model=model,
        num_ctx=num_ctx,
        temperature=temperature,
    )


def diagnostics_log_fields(diag: PlanningDiagnostics) -> dict[str, Any]:
    return {
        "system_prompt_chars": diag.system_prompt_chars,
        "system_prompt_estimated_tokens": estimate_tokens(diag.system_prompt_chars),
        "rendered_tool_catalog_chars": diag.rendered_tool_catalog_chars,
        "message_count": diag.message_count,
        "total_message_chars": diag.total_message_chars,
        "estimated_tokens": diag.estimated_tokens,
        "eligible_tool_count": diag.eligible_tool_count,
        "model": diag.model,
        "num_ctx": diag.num_ctx,
        "temperature": diag.temperature,
    }
