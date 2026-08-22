"""The registry is the gate between model output and execution."""

from __future__ import annotations

import pytest
from app.analytics.datasets import DatasetRegistry
from app.core.errors import ToolArgumentError, ToolExecutionError, ToolNotRegisteredError
from app.llm.contracts import ToolCall
from app.places.contracts import PlaceRegistry
from app.tools.context import AnalysisRunState, GroundingState, ToolContext
from app.tools.contracts import ToolOutcome, tool_definition
from app.tools.registry import ToolRegistry
from pydantic import BaseModel, ConfigDict, Field


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, description="Text to echo.")
    times: int = Field(default=1, ge=1, le=3)


class EchoResult(BaseModel):
    echoed: str


def _ctx() -> ToolContext:
    return ToolContext(
        datasets=DatasetRegistry(),
        analysis=AnalysisRunState(),
        user_message="hi",
        places=PlaceRegistry(),
        grounding=GroundingState(),
    )


class EchoTool:
    name = "echo"
    description = "Echo the given text."
    args_model = EchoArgs

    def __init__(self) -> None:
        self.calls: list[EchoArgs] = []

    async def execute(self, args: EchoArgs, context: ToolContext) -> ToolOutcome[EchoResult]:
        del context
        self.calls.append(args)
        echoed = " ".join([args.text] * args.times)
        return ToolOutcome(observation=f"echoed '{echoed}'", payload=EchoResult(echoed=echoed))


class ExplodingTool:
    name = "explode"
    description = "Always fails."
    args_model = EchoArgs

    async def execute(self, args: EchoArgs, context: ToolContext) -> ToolOutcome[EchoResult]:
        del args, context
        raise ToolExecutionError("upstream refused the request")


def _registry(*tools: object) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)  # type: ignore[arg-type]
    return registry


async def test_registered_tool_runs_with_validated_arguments():
    tool = EchoTool()
    registry = _registry(tool)
    result = await registry.invoke(
        ToolCall(id="c1", name="echo", arguments={"text": "hi", "times": 2}),
        _ctx(),
    )

    assert result.ok
    assert result.observation == "echoed 'hi hi'"
    assert result.payload == EchoResult(echoed="hi hi")
    assert tool.calls[0].times == 2


async def test_unknown_tool_is_reported_without_execution():
    registry = _registry(EchoTool())
    result = await registry.invoke(ToolCall(id="c1", name="rm_rf", arguments={}), _ctx())

    assert not result.ok
    assert result.error_code == "tool_not_registered"
    assert "echo" in result.observation


async def test_invalid_arguments_are_reported_with_the_failing_field():
    registry = _registry(EchoTool())
    result = await registry.invoke(
        ToolCall(id="c1", name="echo", arguments={"times": 99}),
        _ctx(),
    )

    assert not result.ok
    assert result.error_code == "tool_argument_error"
    assert "text" in result.observation


async def test_unexpected_arguments_are_rejected():
    registry = _registry(EchoTool())
    result = await registry.invoke(
        ToolCall(id="c1", name="echo", arguments={"text": "hi", "shell": "rm -rf /"}),
        _ctx(),
    )
    assert not result.ok
    assert result.error_code == "tool_argument_error"


async def test_tool_failures_become_structured_observations():
    registry = _registry(ExplodingTool())
    result = await registry.invoke(
        ToolCall(id="c1", name="explode", arguments={"text": "x"}),
        _ctx(),
    )

    assert not result.ok
    assert result.error_code == "tool_execution_error"
    assert "upstream refused" in result.observation


def test_duplicate_registration_is_rejected():
    registry = _registry(EchoTool())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoTool())  # type: ignore[arg-type]


def test_definitions_expose_json_schema_for_every_tool():
    registry = _registry(EchoTool())
    definitions = registry.definitions()

    assert [definition.name for definition in definitions] == ["echo"]
    schema = definitions[0].parameters_schema
    assert schema["properties"]["text"]["description"] == "Text to echo."
    assert schema["required"] == ["text"]


def test_lookup_of_unknown_tool_raises():
    registry = ToolRegistry()
    assert registry.names == ()
    assert not registry.has("echo")
    with pytest.raises(ToolNotRegisteredError):
        registry.get("echo")


def test_validate_arguments_raises_for_bad_input():
    registry = _registry(EchoTool())
    with pytest.raises(ToolArgumentError):
        registry.validate_arguments(ToolCall(id="c1", name="echo", arguments={"text": ""}))


def test_tool_definition_is_derived_from_the_argument_model():
    definition = tool_definition(EchoTool())  # type: ignore[arg-type]
    assert definition.name == "echo"
    assert definition.parameters_schema["type"] == "object"
