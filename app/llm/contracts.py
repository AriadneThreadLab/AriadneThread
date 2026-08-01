"""Provider-neutral language model contracts.

These types are the only vocabulary the agent uses to talk to a model. Adding a
cloud OpenAI-compatible provider later must not require changes here or in any
tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single conversation turn.

    ``tool_call_id`` and ``name`` are only meaningful for ``role="tool"``
    messages, which carry a structured observation back to the model.
    """

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A tool advertised to the model.

    ``parameters_schema`` is a JSON Schema object describing the tool arguments.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model request to run one tool with the given raw arguments.

    Arguments are unvalidated at this point: validation is the registry's job.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """One model reply.

    ``content`` holds the visible answer text only. Any hidden reasoning the
    model emits is stripped by the provider and is never stored or returned.
    """

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    model: str = ""
    finish_reason: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass(frozen=True, slots=True)
class LLMOptions:
    """Per-request model configuration.

    ``None`` means "use the provider default", which comes from settings.
    """

    temperature: float | None = None
    num_ctx: int | None = None
    max_tokens: int | None = None
    stop: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal chat interface with optional tool advertising."""

    @property
    def model_name(self) -> str: ...

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDefinition] | None = None,
        options: LLMOptions | None = None,
    ) -> LLMResponse:
        """Send a conversation and return a single reply.

        Implementations must apply an explicit network timeout and raise
        :class:`app.core.errors.LLMError` subclasses on failure.
        """
        ...

    async def aclose(self) -> None:
        """Release provider resources (connection pools, sessions)."""
        ...
