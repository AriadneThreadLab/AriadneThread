"""Tool contracts.

A tool is a named, schema-described capability the agent may invoke. Every tool
declares a Pydantic argument model and a Pydantic result model; the registry
validates arguments before execution, so tool implementations never see
unvalidated model output.

Results are split in two on purpose:

* ``observation`` - a short factual string appended to the conversation. It is
  what the model reads, so it must be compact and must state what the data is.
* ``payload``     - the full structured result kept for the API response
  (GeoJSON, passages, provenance). The model never sees it verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from app.llm.contracts import ToolDefinition
from app.tools.context import ToolContext

ArgsT = TypeVar("ArgsT", bound=BaseModel)
ResultT = TypeVar("ResultT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ToolOutcome(Generic[ResultT]):
    """What one successful tool execution produced."""

    observation: str
    payload: ResultT


@runtime_checkable
class Tool(Protocol[ArgsT, ResultT]):
    """A registered capability.

    Implementations must be side-effect free with respect to conversation state
    and must translate upstream failures into
    :class:`app.core.errors.ToolError` subclasses.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def args_model(self) -> type[ArgsT]: ...

    async def execute(self, args: ArgsT, context: ToolContext) -> ToolOutcome[ResultT]: ...


def tool_definition(tool: Tool[Any, Any]) -> ToolDefinition:
    """Derive the model-facing tool description from its argument model."""
    return ToolDefinition(
        name=tool.name,
        description=tool.description,
        parameters_schema=tool.args_model.model_json_schema(),
    )
