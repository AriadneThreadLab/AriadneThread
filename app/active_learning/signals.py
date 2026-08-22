"""Trigger extraction: turn a finished run into bounded, structured signals.

Everything here is derived from the operational trace and the public agent
result. No prompts, no model reasoning and no raw upstream payloads are read.

The single most important distinction made in this module is *attribution*: an
Overpass gateway timeout says nothing about the model, while a rejected
``place_ref`` or an invented coordinate says a great deal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.active_learning.contracts import (
    ErrorAttribution,
    MetricSelectionSummary,
    PlaceResolutionEvent,
    RagEvidenceSummary,
    SelectionReason,
    TaskType,
    ToolValidationEvent,
)
from app.agent.comparison_workflow import is_multi_target_landmark_comparison
from app.agent.contracts import GeoAgentResponse
from app.tools.resolve_place import TOOL_NAME as RESOLVE_PLACE

#: Failures owned by an external dependency. On their own these are an
#: infrastructure signal, never evidence that the model needs fine-tuning.
EXTERNAL_ERROR_CODES: frozenset[str] = frozenset(
    {
        "overpass_error",
        "overpass_timeout",
        "overpass_rate_limited",
        "overpass_upstream_error",
        "overpass_bad_response",
        "ollama_unreachable",
        "ollama_model_not_found",
        "avalai_authentication_error",
        "avalai_billing_error",
        "avalai_rate_limited",
        "avalai_unavailable",
        "avalai_invalid_model",
        "embedding_error",
        "dependency_unavailable",
        "agent_request_timeout",
        "tool_execution_error",
        "tool_timeout",
        "tool_not_registered",
    }
)

#: Failures caused by what the model asked for.
MODEL_ERROR_CODES: frozenset[str] = frozenset(
    {
        "tool_argument_error",
        "tool_not_eligible",
        "llm_protocol_error",
        "llm_tool_protocol_error",
        "ollama_protocol_error",
        "ollama_invalid_response",
        "avalai_invalid_response",
        "invalid_model_response",
        "llm_timeout",
        "overpass_query_build_error",
    }
)

#: Place resolution can fail for either side; the message decides.
_TRANSPORT_PLACE_FAILURE = re.compile(
    r"(timed out|rate limited|upstream error|request failed|non-JSON|not registered"
    r"|must be an array|at least 2 characters)",
    re.IGNORECASE,
)

_RESOLVED_TARGET = re.compile(r"target=(.+?)(?:\s+source=|\s+status=|$)")

_GUARDRAIL_PATTERNS: tuple[tuple[re.Pattern[str], tuple[SelectionReason, ...]], ...] = (
    (
        re.compile(r"inventing coordinates is not allowed", re.IGNORECASE),
        (SelectionReason.HALLUCINATED_COORDINATE_BLOCKED, SelectionReason.GUARDRAIL_TRIGGERED),
    ),
    (
        re.compile(r"inventing a buffer distance is not allowed", re.IGNORECASE),
        (SelectionReason.GUARDRAIL_TRIGGERED,),
    ),
    (
        re.compile(r"unknown (?:place_ref|dataset_ref)", re.IGNORECASE),
        (SelectionReason.DEPENDENCY_ORDER_VIOLATION, SelectionReason.GUARDRAIL_TRIGGERED),
    ),
    (
        re.compile(r"silent substitution is not allowed|grounded tag selection", re.IGNORECASE),
        (SelectionReason.HALLUCINATED_TAG_BLOCKED, SelectionReason.GROUNDING_CONFLICT),
    ),
)

_PROTOCOL_CODES = frozenset(
    {
        "llm_protocol_error",
        "llm_tool_protocol_error",
        "ollama_protocol_error",
        "ollama_invalid_response",
        "avalai_invalid_response",
        "invalid_model_response",
    }
)


@dataclass(frozen=True, slots=True)
class RunSignals:
    """Deterministic, bounded description of one completed run."""

    task_type: TaskType
    reasons: frozenset[SelectionReason]
    tool_sequence: tuple[str, ...] = ()
    tool_validation_events: tuple[ToolValidationEvent, ...] = ()
    place_resolution_events: tuple[PlaceResolutionEvent, ...] = ()
    rag_evidence: RagEvidenceSummary | None = None
    metric_selection: MetricSelectionSummary | None = None
    model_error_codes: tuple[str, ...] = ()
    external_error_codes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default=())
    outcome_successful: bool = False
    feature_concept: str | None = None
    scope_kind: str = "none"

    @property
    def has_model_signal(self) -> bool:
        """Whether anything here implicates model behaviour."""
        return bool(self.reasons - {SelectionReason.NOVEL_QUERY})


def split_code(entry: str) -> tuple[str, str]:
    """Split an accumulated ``"code: message"`` entry."""
    code, separator, message = entry.partition(":")
    if not separator:
        return entry.strip(), ""
    return code.strip(), message.strip()


def classify_error(code: str, message: str) -> ErrorAttribution:
    """Attribute one error code to the model or to an external dependency."""
    if code in MODEL_ERROR_CODES:
        return ErrorAttribution.MODEL
    if code in EXTERNAL_ERROR_CODES:
        return ErrorAttribution.EXTERNAL
    if code in {"place_resolution_error", "place_ambiguous"}:
        if _TRANSPORT_PLACE_FAILURE.search(message):
            return ErrorAttribution.EXTERNAL
        # Nominatim answered, but nothing acceptable was selected for the
        # model's query: that is a planning signal.
        return ErrorAttribution.MODEL
    return ErrorAttribution.NONE


def _reasons_for_error(code: str, message: str) -> set[SelectionReason]:
    reasons: set[SelectionReason] = set()
    if code == "tool_argument_error":
        reasons.add(SelectionReason.TOOL_VALIDATION_FAILED)
        for pattern, mapped in _GUARDRAIL_PATTERNS:
            if pattern.search(message):
                reasons.update(mapped)
    elif code == "tool_not_eligible":
        reasons.add(SelectionReason.WRONG_OR_INELIGIBLE_TOOL)
        reasons.add(SelectionReason.GUARDRAIL_TRIGGERED)
    elif code in _PROTOCOL_CODES:
        reasons.add(SelectionReason.LLM_PROTOCOL_ERROR)
        reasons.add(SelectionReason.INVALID_STRUCTURED_OUTPUT)
    elif code == "llm_timeout":
        reasons.add(SelectionReason.MODEL_TIMEOUT)
    elif code == "place_ambiguous":
        reasons.add(SelectionReason.PLACE_RESOLUTION_AMBIGUOUS)
    elif code == "place_resolution_error":
        reasons.add(SelectionReason.PLACE_RESOLUTION_FAILED)
    return reasons


def _tool_events(
    response: GeoAgentResponse,
) -> tuple[
    tuple[str, ...],
    tuple[ToolValidationEvent, ...],
    tuple[PlaceResolutionEvent, ...],
]:
    sequence: list[str] = []
    validations: list[ToolValidationEvent] = []
    places: list[PlaceResolutionEvent] = []
    for event in response.trace:
        name = event.tool_name
        if event.kind == "tool_call" and name:
            sequence.append(name)
        elif event.kind == "tool_result" and name:
            validations.append(
                ToolValidationEvent(
                    tool_name=name,
                    status="passed",
                    round_index=event.round_index,
                )
            )
            if name == RESOLVE_PLACE:
                match = _RESOLVED_TARGET.search(event.message)
                places.append(
                    PlaceResolutionEvent(
                        label=(match.group(1).strip() if match else "resolved place")[:160],
                        status="resolved",
                        source="nominatim",
                    )
                )
        elif event.kind == "tool_error" and name:
            validations.append(
                ToolValidationEvent(
                    tool_name=name,
                    status="failed",
                    error_code=event.error_code,
                    round_index=event.round_index,
                    detail=event.message[:400],
                )
            )
            if name == RESOLVE_PLACE:
                ambiguous = event.error_code == "place_ambiguous"
                places.append(
                    PlaceResolutionEvent(
                        label="unresolved place",
                        status="ambiguous" if ambiguous else "failed",
                        error_code=event.error_code,
                    )
                )
    return tuple(sequence[:32]), tuple(validations[:32]), tuple(places[:8])


def _metric_summary(
    response: GeoAgentResponse,
) -> tuple[MetricSelectionSummary | None, set[SelectionReason]]:
    reasons: set[SelectionReason] = set()
    block = response.analysis
    if block is None:
        return None, reasons

    trace = block.decision_trace
    selections = trace.metric_selections
    rejections = [item.rejection_reason for item in selections if item.rejection_reason is not None]
    superseded = any(item.final_status == "superseded" for item in selections)
    executed = next((item for item in selections if item.final_status == "executed"), None)

    if rejections:
        reasons.add(SelectionReason.METRIC_FEASIBILITY_FAILED)
    if superseded or trace.plan_revision_count > 0:
        reasons.add(SelectionReason.METRIC_SELECTION_UNCERTAIN)
        reasons.add(SelectionReason.MODEL_DISAGREEMENT)
    if block.status != "completed":
        reasons.add(SelectionReason.METRIC_FEASIBILITY_FAILED)

    rule_id: str | None = None
    if executed is not None:
        rule_id = next(
            (item.rule_id for item in executed.evidence if item.rule_id is not None),
            None,
        )

    summary = MetricSelectionSummary(
        selected_metric=trace.final_primary_metric,
        metric_rule_id=rule_id,
        inferred_goal=executed.inferred_goal if executed is not None else None,
        rejection_reasons=[str(item) for item in rejections][:8],
        analysis_status=block.status,
    )
    return summary, reasons


def _fallback_reasons(response: GeoAgentResponse) -> set[SelectionReason]:
    """Deterministic backend fallbacks mean the model's own output was unusable."""
    reasons: set[SelectionReason] = set()
    for event in response.trace:
        if event.kind != "llm_turn":
            continue
        text = event.message.lower()
        if "fell back to deterministic seed plan" in text:
            reasons.add(SelectionReason.INVALID_STRUCTURED_OUTPUT)
            reasons.add(SelectionReason.MODEL_DISAGREEMENT)
        elif "fallback" in text:
            reasons.add(SelectionReason.MODEL_DISAGREEMENT)
    return reasons


def _task_type(user_query: str, response: GeoAgentResponse) -> TaskType:
    if response.analysis is not None and response.analysis.plan.analysis_type == "comparison":
        return TaskType.COMPARISON
    if is_multi_target_landmark_comparison(user_query):
        return TaskType.COMPARISON
    if response.overpass_query or response.geojson is not None:
        return TaskType.SPATIAL_SEARCH
    if response.passages:
        return TaskType.KNOWLEDGE_ONLY
    return TaskType.UNKNOWN


def _scope_kind(response: GeoAgentResponse) -> str:
    summary = (response.scope_summary or "").lower()
    if "around" in summary or "m of" in summary or "comparison targets" in summary:
        return "radius"
    if summary:
        return "place"
    return "none"


def extract_run_signals(user_query: str, response: GeoAgentResponse) -> RunSignals:
    """Derive bounded selection signals from one completed run."""
    reasons: set[SelectionReason] = set()
    model_codes: list[str] = []
    external_codes: list[str] = []

    for entry in list(response.errors) + list(response.warnings):
        code, message = split_code(entry)
        if not code:
            continue
        attribution = classify_error(code, message)
        if attribution is ErrorAttribution.MODEL:
            model_codes.append(code)
            reasons.update(_reasons_for_error(code, message))
        elif attribution is ErrorAttribution.EXTERNAL:
            external_codes.append(code)

    if response.live_error_code and response.live_error_code.startswith("overpass_"):
        external_codes.append(response.live_error_code)

    sequence, validations, places = _tool_events(response)
    metric_selection, metric_reasons = _metric_summary(response)
    reasons.update(metric_reasons)
    reasons.update(_fallback_reasons(response))

    rag_evidence = RagEvidenceSummary(
        passage_count=len(response.passages),
        documented_tags=list(response.validated_tags)[:16],
        document_titles=[passage.document_title for passage in response.passages][:16],
    )

    successful = (
        response.stop_reason == "final_answer"
        and not model_codes
        and not response.live_query_failed
        and (
            response.feature_count is not None
            or (response.analysis is not None and response.analysis.status == "completed")
        )
    )
    if successful and not reasons:
        reasons.add(SelectionReason.SUCCESSFUL_HIGH_VALUE_TRACE)

    feature_concept = None
    if response.analysis is not None:
        feature_concept = response.analysis.plan.feature_concept
    elif response.validated_tags:
        feature_concept = response.validated_tags[0]

    return RunSignals(
        task_type=_task_type(user_query, response),
        reasons=frozenset(reasons),
        tool_sequence=sequence,
        tool_validation_events=validations,
        place_resolution_events=places,
        rag_evidence=rag_evidence,
        metric_selection=metric_selection,
        model_error_codes=tuple(dict.fromkeys(model_codes)),
        external_error_codes=tuple(dict.fromkeys(external_codes)),
        warnings=tuple(response.warnings)[:16],
        outcome_successful=successful,
        feature_concept=feature_concept,
        scope_kind=_scope_kind(response),
    )
