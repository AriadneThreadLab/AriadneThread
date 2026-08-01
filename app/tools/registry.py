"""Tool registry.

The registry is the single gate between model output and execution. A tool call
runs only if the tool is registered *and* its arguments validate against the
tool's own schema. Everything else becomes a structured tool error that the
agent can observe and recover from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.core.errors import (
    ToolArgumentError,
    ToolError,
    ToolNotRegisteredError,
)
from app.llm.contracts import ToolCall, ToolDefinition
from app.tools.contracts import Tool, ToolOutcome, tool_definition


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """The outcome of one attempted tool call, successful or not."""

    call_id: str
    tool_name: str
    ok: bool
    observation: str
    payload: Any | None = None
    error_code: str | None = None

    @classmethod
    def failure(cls, call: ToolCall, error: ToolError) -> ToolInvocation:
        return cls(
            call_id=call.id,
            tool_name=call.name,
            ok=False,
            observation=f"Tool '{call.name}' failed ({error.code}): {error.message}",
            error_code=error.code,
        )


class ToolRegistry:
    """An immutable-after-setup collection of approved tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any, Any]] = {}

    def register(self, tool: Tool[Any, Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any, Any]:
        try:
            return self._tools[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._tools)) or "none"
            raise ToolNotRegisteredError(
                f"unknown tool '{name}'; available tools: {known}"
            ) from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def definitions(self) -> list[ToolDefinition]:
        """Model-facing descriptions of every registered tool."""
        return [tool_definition(self._tools[name]) for name in self.names]

    def validate_arguments(self, call: ToolCall) -> Any:
        """Coerce raw model arguments into the tool's argument model."""
        tool = self.get(call.name)
        try:
            return tool.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            raise ToolArgumentError(
                f"invalid arguments for '{call.name}': {_summarise(exc)}"
            ) from exc

    async def invoke(self, call: ToolCall) -> ToolInvocation:
        """Validate and run one tool call, never raising for tool-level failures."""
        try:
            tool = self.get(call.name)
            args = self.validate_arguments(call)
            outcome: ToolOutcome[Any] = await tool.execute(args)
        except ToolError as error:
            return ToolInvocation.failure(call, error)
        return ToolInvocation(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            observation=outcome.observation,
            payload=outcome.payload,
        )


def _summarise(error: ValidationError, *, max_errors: int = 3) -> str:
    """Render validation problems compactly enough to feed back to the model."""
    parts = []
    for detail in error.errors()[:max_errors]:
        location = ".".join(str(item) for item in detail["loc"]) or "<root>"
        parts.append(f"{location}: {detail['msg']}")
    return "; ".join(parts)
