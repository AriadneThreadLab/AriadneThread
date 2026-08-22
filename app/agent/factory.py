"""Construct the planner-executor agent from settings and collaborators."""

from __future__ import annotations

from app.agent.loop import LoopLimits
from app.agent.orchestrator import PlannerExecutorAgent
from app.core.config import Settings
from app.execution_memory.service import ExecutionMemoryService
from app.llm.contracts import LLMProvider
from app.tools.registry import ToolRegistry


def build_loop_limits(settings: Settings) -> LoopLimits:
    return LoopLimits(
        max_tool_rounds=settings.agent_max_tool_rounds,
        max_tool_calls=settings.agent_max_tool_calls,
    )


def build_geo_agent(
    llm: LLMProvider,
    registry: ToolRegistry,
    settings: Settings,
    execution_memory: ExecutionMemoryService | None = None,
) -> PlannerExecutorAgent:
    """Build the bounded GeoAgent over the given registry and LLM provider."""
    return PlannerExecutorAgent(
        llm,
        registry,
        build_loop_limits(settings),
        execution_memory,
    )
