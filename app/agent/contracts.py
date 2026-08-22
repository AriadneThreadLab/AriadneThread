"""Agent request, response and execution-trace contracts.

The trace records operational events only - which tool ran, what came back, why
the loop stopped. Model reasoning is stripped at the provider boundary and is
never traced, returned or stored.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.analytics.contracts import AnalysisBlock
from app.execution_memory.contracts import ExecutionMemoryTrace
from app.osm.contracts import GeoJsonFeatureCollection
from app.rag.contracts import RetrievedPassage

TraceEventKind = Literal[
    "request_received",
    "llm_turn",
    "memory_reuse",
    "protocol_repair",
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
    error_code: str | None = None
    # Safe operational metadata only (scope, tags, counts, durations, codes).
    details: dict[str, Any] | None = None
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
        description="Client-supplied identifier that correlates follow-up executions.",
    )
    request_id: str | None = Field(
        default=None,
        max_length=64,
        description="HTTP correlation id when the request arrived via the API.",
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
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    stop_reason: StopReason = "final_answer"
    model: str = ""
    effective_limit: int | None = Field(default=None, ge=0)
    scope_summary: str | None = None
    validated_tags: list[str] = Field(default_factory=list)
    live_query_failed: bool = False
    live_error_code: str | None = None
    analysis: AnalysisBlock | None = Field(
        default=None,
        description="Present only when analyze_features ran in this request.",
    )
    conversation_id: str | None = None
    execution_memory: ExecutionMemoryTrace | None = Field(
        default=None,
        description="Structured Execution Memory provenance; never chain-of-thought.",
    )


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
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
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
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._events.append(
            TraceEvent(
                kind=kind,
                message=message,
                round_index=round_index,
                tool_name=tool_name,
                error_code=error_code,
                details=details,
            )
        )

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def as_list(self) -> list[TraceEvent]:
        return list(self._events)


def describe_payload(payload: Any) -> str:
    """Short, safe description of a tool payload for tracing."""
    place_ref = getattr(payload, "place_ref", None)
    status = getattr(payload, "status", None)
    label = getattr(payload, "label", None)
    if isinstance(place_ref, str) and status == "resolved":
        source = getattr(payload, "source", None)
        target = label if isinstance(label, str) else place_ref
        source_bit = f" source={source}" if isinstance(source, str) else ""
        return f"Resolved comparison target target={target}{source_bit} status=completed"
    feature_count = getattr(payload, "feature_count", None)
    if isinstance(feature_count, int):
        analysis_target = getattr(payload, "analysis_target", None)
        if isinstance(analysis_target, str) and analysis_target:
            return f"received {feature_count} feature(s) for {analysis_target}"
        return f"received {feature_count} feature(s)"
    passage_count = getattr(payload, "passage_count", None)
    if isinstance(passage_count, int):
        return f"received {passage_count} documentation passage(s)"
    metrics_computed = getattr(payload, "metrics_computed", None)
    if isinstance(metrics_computed, int):
        analysis_status = getattr(payload, "analysis_status", None)
        if analysis_status:
            return f"analytics {analysis_status}; computed {metrics_computed} metric(s)"
        return f"computed {metrics_computed} metric(s)"
    return "received a structured result"


def summarise_arguments(arguments: dict[str, Any], *, max_chars: int = 180) -> str:
    """Compact argument summary for the operational trace (never full payloads)."""
    text = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
