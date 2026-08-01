"""Bounds for the planner-executor tool loop.

:class:`LoopBudget` is the single enforcement point for
``AGENT_MAX_TOOL_ROUNDS`` and ``AGENT_MAX_TOOL_CALLS``. The orchestrator in
``app.agent.orchestrator`` consumes these counters each tool round.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent.contracts import StopReason
from app.llm.contracts import ToolCall


@dataclass(frozen=True, slots=True)
class LoopLimits:
    """Hard ceilings for one agent run."""

    max_tool_rounds: int
    max_tool_calls: int

    def __post_init__(self) -> None:
        if self.max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")


class LoopBudget:
    """Mutable per-run counter enforcing :class:`LoopLimits`.

    A run may perform at most ``max_tool_rounds`` model turns that request
    tools, and at most ``max_tool_calls`` tool executions in total. When a round
    requests more calls than remain, the surplus is dropped rather than the run
    being aborted, so the agent still observes partial results.
    """

    def __init__(self, limits: LoopLimits) -> None:
        self._limits = limits
        self._rounds_used = 0
        self._calls_used = 0

    @property
    def rounds_used(self) -> int:
        return self._rounds_used

    @property
    def calls_used(self) -> int:
        return self._calls_used

    @property
    def calls_remaining(self) -> int:
        return max(0, self._limits.max_tool_calls - self._calls_used)

    @property
    def rounds_remaining(self) -> int:
        return max(0, self._limits.max_tool_rounds - self._rounds_used)

    def can_run_tools(self) -> bool:
        return self.rounds_remaining > 0 and self.calls_remaining > 0

    def admit(self, calls: tuple[ToolCall, ...]) -> tuple[ToolCall, ...]:
        """Consume one round and return the calls that fit in the budget."""
        if not self.can_run_tools():
            return ()
        self._rounds_used += 1
        admitted = calls[: self.calls_remaining]
        self._calls_used += len(admitted)
        return admitted

    def exhausted_reason(self) -> StopReason | None:
        """Why the loop must stop, or ``None`` while budget remains."""
        if self.rounds_remaining == 0:
            return "max_tool_rounds"
        if self.calls_remaining == 0:
            return "max_tool_calls"
        return None
