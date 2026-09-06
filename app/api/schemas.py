"""Public HTTP schemas for the agent API.

Thin serialization models mapped explicitly from domain contracts. They do not
duplicate OsmFeatureQuery or Overpass transport types.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.active_learning.contracts import FeedbackSentiment
from app.agent.contracts import GeoAgentResponse, StopReason, TraceEvent
from app.analytics.charts import AnalysisChart
from app.analytics.contracts import AnalysisBlock
from app.execution_memory.contracts import ExecutionMemoryTrace
from app.llm.tool_protocol import FINAL_ANSWER_KEY, TOOL_CALL_KEY
from app.osm.contracts import OSM_ATTRIBUTION, GeoJsonFeatureCollection
from app.rag.contracts import RetrievedPassage
from app.tools.energy_report import EnergyAnalysisReport
from app.tools.energy_tools import ENERGY_ANALYSIS_ATTRIBUTION
from app.tools.simbench_tools import SIMBENCH_ATTRIBUTION

StopReasonOut = StopReason

ResultStatus = Literal[
    "completed",
    "completed_with_warnings",
    "no_matching_features",
    "failed",
    "timed_out",
    "rate_limited",
    "invalid_model_response",
    "dependency_unavailable",
]

_PROTOCOL_LEAK_RE = re.compile(
    rf"({TOOL_CALL_KEY}|{FINAL_ANSWER_KEY}|\breasoning_content\b|<think\b|```)",
    re.IGNORECASE,
)

_SAFE_PROTOCOL_ANSWER = (
    "The language model returned an invalid structured response. No geographic query was executed."
)


class AgentQueryRequest(BaseModel):
    """Natural-language GIS request accepted by the public API."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    message: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Natural-language GIS question or map request.",
        examples=[
            "Find up to 20 leisure=park features in Tehran, Iran.",
            "Find public parks in Tehran, Iran. Return at most 20 features.",
            "پارک‌های عمومی تهران را پیدا کن و حداکثر ۲۰ نتیجه نشان بده.",
        ],
    )
    conversation_id: str | None = Field(
        default=None,
        max_length=64,
        description="Stable id for follow-up analytical executions in one conversation.",
    )

    @field_validator("message")
    @classmethod
    def _reject_whitespace_only(cls, value: str) -> str:
        text = value.strip()
        if len(text) < 2:
            raise ValueError("message must not be empty or whitespace-only")
        return text


class AgentFeedbackRequest(BaseModel):
    """Feedback about one completed run, keyed by its correlation id."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    request_id: str = Field(min_length=1, max_length=64)
    sentiment: FeedbackSentiment
    failure_category: str | None = Field(
        default=None,
        max_length=64,
        description="Optional short label such as 'wrong_metric' or 'wrong_target'.",
    )
    corrected_output: dict[str, Any] | None = Field(
        default=None,
        description="Optional corrected plan; validated against the authoritative schema.",
    )
    note: str | None = Field(default=None, max_length=2000)


class AgentFeedbackResponse(BaseModel):
    """Outcome of a feedback submission (never a training trigger)."""

    model_config = ConfigDict(frozen=True)

    accepted: bool
    candidate_id: str | None = None
    review_status: str | None = None
    detail: str = ""


class KnowledgeSourceOut(BaseModel):
    """One OSM documentation passage citation (never live map data)."""

    model_config = ConfigDict(frozen=True)

    title: str
    section: str | None = None
    url: str
    score: float


class ExecutionTraceStepOut(BaseModel):
    """One operational trace step for API clients."""

    model_config = ConfigDict(frozen=True)

    step: int = Field(ge=1)
    event: str
    tool: str | None = None
    status: Literal["info", "completed", "failed", "warning"] = "info"
    message: str | None = None
    error_code: str | None = None
    details: dict[str, Any] | None = None


class AgentQueryResponse(BaseModel):
    """Stable public agent result."""

    model_config = ConfigDict(frozen=True)

    answer: str
    status: ResultStatus = "completed"
    knowledge_sources: list[KnowledgeSourceOut] = Field(default_factory=list)
    execution_trace: list[ExecutionTraceStepOut] = Field(default_factory=list)
    overpass_query: str | None = None
    geojson: GeoJsonFeatureCollection | None = None
    feature_count: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    stop_reason: StopReasonOut
    model: str = ""
    attribution: str | None = None
    request_id: str | None = None
    effective_limit: int | None = Field(default=None, ge=0)
    scope_summary: str | None = None
    validated_tags: list[str] = Field(default_factory=list)
    live_query_executed: bool = False
    live_data_available: bool = False
    live_query_failed: bool = False
    live_error_code: str | None = None
    analysis: AnalysisBlock | None = Field(
        default=None,
        description="Present only when analyze_features ran successfully or was rejected.",
    )
    energy_analysis: EnergyAnalysisReport | None = Field(
        default=None,
        description="Structured GeoLoadST analysis results for the Energy Analysis Results UI.",
    )
    charts: list[AnalysisChart] = Field(
        default_factory=list,
        description="Structured chart specs from the analysis engine. Empty when absent.",
    )
    conversation_id: str | None = None
    execution_memory: ExecutionMemoryTrace | None = None


def _trace_status(event: TraceEvent) -> Literal["info", "completed", "failed", "warning"]:
    if event.kind == "tool_result":
        return "completed"
    if event.kind == "tool_error":
        return "failed"
    if event.kind == "protocol_repair":
        return "warning"
    if event.kind == "stopped" and event.error_code:
        return "failed"
    if event.details:
        status = event.details.get("status")
        if status == "info":
            return "info"
        if status == "completed":
            return "completed"
        if status == "failed":
            return "failed"
        if status == "warning":
            return "warning"
    return "info"


#: Compact UI citation list; model grounding uses the tool's own top_k.
_UI_SOURCE_DISPLAY_LIMIT = 8


def knowledge_sources_from_passages(
    passages: list[RetrievedPassage],
) -> list[KnowledgeSourceOut]:
    ranked = sorted(passages, key=lambda item: item.score, reverse=True)
    return [
        KnowledgeSourceOut(
            title=passage.document_title,
            section=passage.section,
            url=passage.source_url,
            score=passage.score,
        )
        for passage in ranked[:_UI_SOURCE_DISPLAY_LIMIT]
    ]


def execution_trace_from_events(events: list[Any]) -> list[ExecutionTraceStepOut]:
    steps: list[ExecutionTraceStepOut] = []
    for index, event in enumerate(events, start=1):
        if not isinstance(event, TraceEvent):
            continue
        message = event.message
        if (
            message
            and _PROTOCOL_LEAK_RE.search(message)
            and event.kind in {"final_answer", "llm_turn", "stopped"}
        ):
            # Keep operational tool_call arg summaries; strip other leaks.
            message = "operational detail omitted"
        steps.append(
            ExecutionTraceStepOut(
                step=index,
                event=event.kind,
                tool=event.tool_name,
                status=_trace_status(event),
                message=message,
                error_code=event.error_code,
                details=event.details,
            )
        )
    return steps


def sanitize_public_answer(answer: str, *, errors: list[str], feature_count: int | None) -> str:
    """Defence-in-depth: never return prompted protocol objects as the answer."""
    text = (answer or "").strip()
    if not text:
        if any("llm_protocol_error" in err for err in errors):
            return _SAFE_PROTOCOL_ANSWER
        if feature_count == 0:
            return "The request completed, but no matching OpenStreetMap features were found."
        return "No final answer was produced."
    if _PROTOCOL_LEAK_RE.search(text) or text.lstrip().startswith("{"):
        if any("llm_protocol_error" in err for err in errors):
            return _SAFE_PROTOCOL_ANSWER
        return "No final answer was produced because the model response was invalid."
    return text


def derive_result_status(result: GeoAgentResponse) -> ResultStatus:
    error_blob = " ".join(result.errors).lower()
    answer_l = result.answer.lower()
    live_code = (result.live_error_code or "").lower()

    if result.stop_reason == "llm_error":
        if "llm_timeout" in error_blob:
            return "timed_out"
        if "llm_protocol_error" in error_blob or "invalid structured" in answer_l:
            return "invalid_model_response"
        if "overpass_timeout" in error_blob:
            return "timed_out"
        if "overpass_rate_limited" in error_blob or "429" in error_blob:
            return "rate_limited"
        if "place_resolution" in error_blob or "place_ambiguous" in error_blob:
            return "failed"
        return "dependency_unavailable"

    if result.live_query_failed or live_code.startswith("overpass_"):
        if live_code == "overpass_timeout" or "overpass_timeout" in error_blob:
            return "timed_out"
        if live_code == "overpass_rate_limited" or "overpass_rate_limited" in error_blob:
            return "rate_limited"
        return "failed"

    if result.stop_reason in {"max_tool_rounds", "max_tool_calls"}:
        return "failed" if result.errors else "completed_with_warnings"
    if result.errors and result.geojson is None and result.feature_count is None:
        if "overpass_timeout" in error_blob:
            return "timed_out"
        if "overpass_rate_limited" in error_blob:
            return "rate_limited"
        return "failed"
    if result.feature_count == 0 and result.overpass_query:
        return "no_matching_features"
    if (
        result.analysis is not None
        and result.analysis.status != "completed"
        and not result.errors
        and result.stop_reason == "final_answer"
    ):
        return "completed_with_warnings"
    if result.warnings:
        return "completed_with_warnings"
    return "completed"


def to_agent_query_response(
    result: GeoAgentResponse,
    *,
    request_id: str | None,
) -> AgentQueryResponse:
    """Map a domain agent result to the public HTTP contract."""
    geojson = result.geojson
    live_data_available = geojson is not None
    osm_live = (
        any(source.kind == "osm_features" for source in result.sources)
        or bool(result.overpass_query)
        or result.live_query_failed
    )
    simbench_live = any(source.kind == "simbench_network" for source in result.sources)
    energy_live = any(source.kind == "energy_analysis" for source in result.sources)
    live_query_executed = osm_live
    attribution_parts: list[str] = []
    if osm_live:
        attribution_parts.append(OSM_ATTRIBUTION)
    if simbench_live:
        attribution_parts.append(SIMBENCH_ATTRIBUTION)
    if energy_live:
        attribution_parts.append(ENERGY_ANALYSIS_ATTRIBUTION)
    attribution = "; ".join(attribution_parts) or None
    answer = sanitize_public_answer(
        result.answer,
        errors=list(result.errors),
        feature_count=result.feature_count,
    )
    return AgentQueryResponse(
        answer=answer,
        status=derive_result_status(result),
        knowledge_sources=knowledge_sources_from_passages(list(result.passages)),
        execution_trace=execution_trace_from_events(list(result.trace)),
        overpass_query=result.overpass_query,
        geojson=geojson,
        feature_count=result.feature_count,
        warnings=list(result.warnings),
        errors=list(result.errors),
        stop_reason=result.stop_reason,
        model=result.model,
        attribution=attribution,
        request_id=request_id,
        effective_limit=result.effective_limit,
        scope_summary=result.scope_summary,
        validated_tags=list(result.validated_tags),
        live_query_executed=live_query_executed,
        live_data_available=live_data_available,
        live_query_failed=result.live_query_failed,
        live_error_code=result.live_error_code,
        analysis=result.analysis,
        energy_analysis=result.energy_analysis,
        charts=list(result.charts),
        conversation_id=result.conversation_id,
        execution_memory=result.execution_memory,
    )
