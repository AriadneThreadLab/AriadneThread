"""Agent request, response and execution-trace contracts.

The trace records operational events only - which tool ran, what came back, why
the loop stopped. Model reasoning is stripped at the provider boundary and is
never traced, returned or stored.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.osm.contracts import GeoJsonFeatureCollection
from app.rag.contracts import RetrievedPassage

TraceEventKind = Literal[
    "request_received",
    "tool_call",
    "tool_result",
    "tool_error",
    "final_answer",
    "stopped",
]

StopReason = Literal[
    "final_answer",
    "max_tool_rounds",
    "max_tool_calls",
    "llm_error",
]


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class TraceEvent(BaseModel):
    """One operational step of a run."""

    model_config = ConfigDict(frozen=True)

    kind: TraceEventKind
    message: str = Field(description="Human-readable operational summary.")
    round_index: int = Field(default=0, ge=0)
    tool_name: str | None = None
    at: datetime = Field(default_factory=_now)


class SourceReference(BaseModel):
    """A citable source behind an answer."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["osm_documentation", "osm_features"]
    title: str
    url: str | None = None


class GeoAgentRequest(BaseModel):
    """A natural-language GIS request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str = Field(min_length=2, max_length=2000)
    conversation_id: str | None = Field(
        default=None,
        max_length=64,
        description="Optional client-supplied identifier for correlating runs.",
    )


class GeoAgentResponse(BaseModel):
    """The agent's answer plus everything needed to verify it."""

    model_config = ConfigDict(frozen=True)

    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    geojson: GeoJsonFeatureCollection | None = Field(
        default=None,
        description="Live OSM features, present only when query_osm ran successfully.",
    )
    feature_count: int | None = Field(default=None, ge=0)
    passages: list[RetrievedPassage] = Field(default_factory=list)
    overpass_query: str | None = None
    stop_reason: StopReason = "final_answer"
    model: str = ""


@runtime_checkable
class GeoAgent(Protocol):
    """Turns one natural-language request into a grounded, traced answer."""

    async def run(self, request: GeoAgentRequest) -> GeoAgentResponse: ...


@runtime_checkable
class TraceRecorder(Protocol):
    """Collects trace events during a run."""

    def record(
        self,
        kind: TraceEventKind,
        message: str,
        *,
        round_index: int = 0,
        tool_name: str | None = None,
    ) -> None: ...

    @property
    def events(self) -> tuple[TraceEvent, ...]: ...


class ListTraceRecorder:
    """In-memory recorder; one instance per run, so there is no shared state."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def record(
        self,
        kind: TraceEventKind,
        message: str,
        *,
        round_index: int = 0,
        tool_name: str | None = None,
    ) -> None:
        self._events.append(
            TraceEvent(
                kind=kind,
                message=message,
                round_index=round_index,
                tool_name=tool_name,
            )
        )

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def as_list(self) -> list[TraceEvent]:
        return list(self._events)


def describe_payload(payload: Any) -> str:
    """Short, safe description of a tool payload for tracing."""
    feature_count = getattr(payload, "feature_count", None)
    if isinstance(feature_count, int):
        return f"received {feature_count} feature(s)"
    passage_count = getattr(payload, "passage_count", None)
    if isinstance(passage_count, int):
        return f"received {passage_count} documentation passage(s)"
    return "received a structured result"
