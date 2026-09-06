"""Bounded planner-executor orchestration.

The model proposes tool calls; the Tool Registry is the only execution path.
Compact observations are appended to the conversation. Full GeoJSON and
documentation passages accumulate in :class:`ResultAccumulator` for the response.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from app.agent.accumulator import ResultAccumulator
from app.agent.comparison_executor import (
    MultiTargetComparisonRunner,
    should_use_multi_target_runner,
)
from app.agent.contracts import (
    GeoAgentRequest,
    GeoAgentResponse,
    ListTraceRecorder,
    StopReason,
    describe_payload,
    summarise_arguments,
)
from app.agent.energy_intent import is_energy_grid_request
from app.agent.loop import LoopBudget, LoopLimits
from app.agent.planning_diagnostics import diagnostics_log_fields, measure_planning_messages
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.spatial_trust import untrusted_spatial_scope_error
from app.analytics.report import verify_numeric_fidelity
from app.core.errors import (
    LLMError,
    LLMProtocolError,
    LLMTimeoutError,
    ToolArgumentError,
    ToolNotEligibleError,
)
from app.execution_memory.service import ExecutionMemoryService
from app.llm.contracts import ChatMessage, LLMProvider, LLMResponse, ToolCall
from app.llm.prompted_catalogue import render_prompted_tool_instructions
from app.llm.tool_protocol import FINAL_ANSWER_KEY, TOOL_CALL_KEY
from app.places.contracts import PlaceRegistry
from app.tools.analyze_features import TOOL_NAME as ANALYZE_FEATURES
from app.tools.context import AnalysisRunState, GroundingState, ToolContext
from app.tools.energy_capabilities import safe_logged_energy_ids
from app.tools.energy_tools import TOOL_NAME as ANALYZE_ENERGY_GRID
from app.tools.query_osm import TOOL_NAME as QUERY_OSM
from app.tools.registry import ToolRegistry
from app.tools.resolve_place import TOOL_NAME as RESOLVE_PLACE
from app.tools.search_osm_knowledge import TOOL_NAME as SEARCH_OSM_KNOWLEDGE
from app.tools.simbench_tools import TOOL_NAME as SIMBENCH_QUERY

_MAX_PROTOCOL_REPAIRS = 2
_MAX_IDENTICAL_ARGUMENT_FAILURES = 2
_ENERGY_TERMINAL_ERROR_CODES = frozenset(
    {
        "energy_plugin_internal_error",
        "energy_plugin_unavailable",
        "chart_normalization_error",
        "energy_analysis_failed",
    }
)
_PROTOCOL_LEAK_HINT = re.compile(
    rf"({TOOL_CALL_KEY}|{FINAL_ANSWER_KEY}|```)",
    re.IGNORECASE,
)
_ANALYTICAL_INTENT = re.compile(
    r"\b("
    r"compare|comparison|more parks|denser|density|average|median|"
    r"accessib|typical size|variability|standard deviation|coverage|"
    r"how many|which area|better access|analyze|analysis|load patterns?"
    r")\b",
    re.IGNORECASE,
)

_PROTOCOL_FAILURE_ANSWER = (
    "The language model returned an invalid structured response. No geographic query was executed."
)
_PROTOCOL_REPAIR_HINT = json.dumps(
    {
        "protocol_error": {
            "message": "Your previous response did not match the required JSON protocol.",
            "allowed_top_level_forms": ["tool_calls", "final_answer"],
            "instruction": "Return exactly one JSON object and no prose.",
            "example_tool_calls": {
                "tool_calls": [
                    {
                        "name": "query_osm",
                        "arguments": {
                            "place": "Tehran, Iran",
                            "tags": [{"key": "leisure", "value": "park"}],
                            "limit": 20,
                        },
                    }
                ]
            },
            "example_final_answer": {"final_answer": "USER-FACING ANSWER"},
            "never_both": "Never include tool_calls and final_answer in the same object.",
        }
    },
    ensure_ascii=False,
    separators=(",", ":"),
)

_TOOL_BATCH_ORDER = {
    SEARCH_OSM_KNOWLEDGE: 0,
    RESOLVE_PLACE: 1,
    QUERY_OSM: 2,
    SIMBENCH_QUERY: 2,
    ANALYZE_FEATURES: 3,
    ANALYZE_ENERGY_GRID: 3,
}

logger = logging.getLogger(__name__)


class PlannerExecutorAgent:
    """One-request planner-executor over registered OSM tools."""

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        limits: LoopLimits,
        execution_memory: ExecutionMemoryService | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._limits = limits
        self._execution_memory = execution_memory

    async def run(self, request: GeoAgentRequest) -> GeoAgentResponse:
        recorder = ListTraceRecorder()
        state = ResultAccumulator()
        tool_context = ToolContext(
            datasets=state.datasets,
            analysis=AnalysisRunState(),
            user_message=request.message,
            places=PlaceRegistry(),
            grounding=GroundingState(),
        )
        recorder.record(
            "request_received",
            f"received request ({len(request.message)} characters)",
            details={"status": "info"},
        )
        conversation_id = (request.conversation_id or "").strip() or str(uuid.uuid4())
        incremental = None
        if self._execution_memory is not None:
            try:
                incremental = await self._execution_memory.prepare_incremental(
                    conversation_id, request.message
                )
            except Exception:
                logger.warning("execution_memory_unavailable during follow-up resolution")
                state.note_warning("execution_memory_unavailable")
                incremental = None

        use_incremental = incremental is not None
        energy_intent = is_energy_grid_request(request.message)
        use_fresh_comparison = should_use_multi_target_runner(request.message) and not energy_intent
        if use_incremental or use_fresh_comparison:
            runner = MultiTargetComparisonRunner(self._llm, self._registry)
            try:
                if (
                    use_incremental
                    and self._execution_memory is not None
                    and incremental is not None
                ):
                    snapshot = await self._execution_memory.latest(conversation_id)
                    if snapshot is not None:
                        memory_trace = self._execution_memory.trace_for(
                            conversation_id=conversation_id,
                            plan=incremental,
                            snapshot=snapshot,
                        )
                        result = await runner.run_incremental(
                            user_message=request.message,
                            state=state,
                            tool_context=tool_context,
                            recorder=recorder,
                            incremental=incremental,
                            snapshot=snapshot,
                            memory_trace=memory_trace,
                        )
                        return await self._finish_with_memory(
                            request,
                            result,
                            tool_context,
                            conversation_id,
                            parent_execution_id=incremental.parent_execution_id,
                            incremental=incremental,
                        )
                if use_fresh_comparison:
                    result = await runner.run(
                        user_message=request.message,
                        state=state,
                        tool_context=tool_context,
                        recorder=recorder,
                    )
                    return await self._finish_with_memory(
                        request, result, tool_context, conversation_id
                    )
            except LLMTimeoutError as exc:
                state.note_error(exc.code, exc.message)
                recorder.record(
                    "stopped",
                    f"LLM timed out: {exc.message}",
                    error_code=exc.code,
                    details={"status": "failed"},
                )
                return GeoAgentResponse(
                    answer=(
                        "The local model timed out during request planning. "
                        "No live OSM query was executed."
                    ),
                    sources=list(state.sources),
                    trace=recorder.as_list(),
                    geojson=None,
                    feature_count=None,
                    passages=list(state.passages),
                    overpass_query=None,
                    warnings=list(state.warnings),
                    errors=list(state.errors),
                    stop_reason="llm_error",
                    model=self._llm.model_name,
                    live_query_failed=False,
                    live_error_code=None,
                    analysis=None,
                    conversation_id=conversation_id,
                )

        budget = LoopBudget(self._limits)
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=request.message),
        ]

        answer = ""
        stop_reason: StopReason = "final_answer"
        protocol_repairs = 0
        llm_round = 0
        invalid_call_counts: dict[str, int] = {}

        while True:
            llm_round += 1
            facing = _model_facing_tools(
                self._registry.definitions(),
                state=state,
                user_message=request.message,
                places=tool_context.places,
            )
            logger.info(
                "available_tools round=%s tool_count=%s tool_names=%s energy_intent=%s",
                llm_round,
                len(facing),
                [getattr(item, "name", None) for item in facing],
                is_energy_grid_request(request.message),
            )
            planning_diag = None
            if llm_round == 1:
                catalog = render_prompted_tool_instructions(facing)
                planning_diag = measure_planning_messages(
                    messages,
                    tools=facing,
                    tool_catalog_text=catalog,
                    model=self._llm.model_name,
                )
                logger.info(
                    "planning_diagnostics %s",
                    " ".join(f"{k}={v}" for k, v in diagnostics_log_fields(planning_diag).items()),
                )
            try:
                response = await self._llm.chat(
                    messages,
                    tools=facing,
                )
            except LLMTimeoutError as exc:
                state.note_error(exc.code, exc.message)
                recorder.record(
                    "stopped",
                    f"LLM timed out: {exc.message}",
                    round_index=llm_round,
                    error_code=exc.code,
                    details={"status": "failed"},
                )
                stop_reason = "llm_error"
                answer = (
                    "The local model timed out during request planning. "
                    "No live OSM query was executed."
                    if not state.datasets.refs() and not state.live_query_failed
                    else "The language model timed out before a final answer was produced."
                )
                break
            except LLMProtocolError as exc:
                state.note_error(exc.code, exc.message)
                recorder.record(
                    "stopped",
                    f"LLM protocol error: {exc.message}",
                    round_index=llm_round,
                    error_code=exc.code,
                    details={"status": "failed"},
                )
                stop_reason = "llm_error"
                answer = _PROTOCOL_FAILURE_ANSWER
                break
            except LLMError as exc:
                state.note_error(exc.code, exc.message)
                recorder.record(
                    "stopped",
                    f"LLM error: {exc.message}",
                    round_index=llm_round,
                    error_code=exc.code,
                    details={"status": "failed"},
                )
                stop_reason = "llm_error"
                answer = "The language model failed before a final answer was produced."
                break

            if response.is_protocol_error:
                protocol_repairs += 1
                detail = response.protocol_error or "invalid prompted protocol payload"
                logger.info(
                    "prompted_turn response_kind=invalid tool_call_count=0 "
                    "protocol_error=true detail=%s",
                    detail,
                )
                if protocol_repairs > _MAX_PROTOCOL_REPAIRS:
                    state.note_error("llm_protocol_error", detail)
                    stop_reason = "llm_error"
                    answer = _PROTOCOL_FAILURE_ANSWER
                    recorder.record(
                        "stopped",
                        "protocol repair budget exhausted",
                        round_index=llm_round,
                        error_code="llm_protocol_error",
                        details={"status": "failed"},
                    )
                    break
                recorder.record(
                    "protocol_repair",
                    "prompt format corrected",
                    round_index=llm_round,
                    error_code="llm_protocol_error",
                    details={
                        "status": "warning",
                        "error_code": "llm_protocol_error",
                        "repair": protocol_repairs,
                    },
                )
                state.note_warning(f"llm_protocol_error (repair {protocol_repairs}): {detail}")
                messages.append(ChatMessage(role="user", content=_PROTOCOL_REPAIR_HINT))
                continue

            logger.info(
                "prompted_turn response_kind=%s tool_call_count=%s tool_names=%s",
                "tool_calls" if response.has_tool_calls else "final_answer",
                len(response.tool_calls),
                [call.name for call in response.tool_calls],
            )
            turn_details: dict[str, Any] = {"status": "info"}
            if planning_diag is not None:
                turn_details.update(diagnostics_log_fields(planning_diag))
            recorder.record(
                "llm_turn",
                _llm_turn_summary(response),
                round_index=llm_round,
                details=turn_details,
            )

            if response.has_tool_calls:
                if not budget.can_run_tools():
                    exhausted = budget.exhausted_reason() or "max_tool_calls"
                    stop_reason = exhausted
                    answer = _budget_stop_answer(exhausted, state)
                    recorder.record(
                        "stopped",
                        f"tool budget exhausted ({exhausted})",
                        round_index=llm_round,
                        error_code=exhausted,
                        details={"status": "failed"},
                    )
                    break

                admitted = budget.admit(response.tool_calls)
                dropped = len(response.tool_calls) - len(admitted)
                if dropped:
                    state.note_warning(
                        f"Dropped {dropped} tool call(s) that exceeded the remaining call budget"
                    )
                # Dependency-safe order: ground/resolve → query → analyze.
                admitted = tuple(
                    sorted(
                        admitted,
                        key=lambda call: _TOOL_BATCH_ORDER.get(call.name, 9),
                    )
                )

                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=response.content or _assistant_tool_stub(response.tool_calls),
                        tool_calls=response.tool_calls,
                    )
                )
                stop_now = await self._execute_calls(
                    admitted,
                    messages=messages,
                    state=state,
                    recorder=recorder,
                    round_index=budget.rounds_used,
                    user_message=request.message,
                    tool_context=tool_context,
                    invalid_call_counts=invalid_call_counts,
                )
                if stop_now is not None:
                    stop_reason = "llm_error"
                    answer = stop_now
                    break

                if not budget.can_run_tools() and not admitted:
                    exhausted = budget.exhausted_reason() or "max_tool_calls"
                    stop_reason = exhausted
                    answer = _budget_stop_answer(exhausted, state)
                    recorder.record(
                        "stopped",
                        f"tool budget exhausted ({exhausted})",
                        round_index=llm_round,
                        error_code=exhausted,
                        details={"status": "failed"},
                    )
                    break
                continue

            content = (response.content or "").strip()
            if _needs_protocol_repair(content, response):
                protocol_repairs += 1
                if protocol_repairs > _MAX_PROTOCOL_REPAIRS:
                    state.note_error(
                        "llm_protocol_error",
                        "model reply was not valid tool_calls/final_answer JSON",
                    )
                    stop_reason = "llm_error"
                    answer = _PROTOCOL_FAILURE_ANSWER
                    recorder.record(
                        "stopped",
                        "protocol repair budget exhausted",
                        round_index=llm_round,
                        error_code="llm_protocol_error",
                        details={"status": "failed"},
                    )
                    break
                recorder.record(
                    "protocol_repair",
                    "prompt format corrected",
                    round_index=llm_round,
                    error_code="llm_protocol_error",
                    details={
                        "status": "warning",
                        "error_code": "llm_protocol_error",
                        "repair": protocol_repairs,
                    },
                )
                state.note_warning(
                    "llm_protocol_error (repair "
                    f"{protocol_repairs}): model reply was not valid "
                    "tool_calls/final_answer JSON"
                )
                messages.append(ChatMessage(role="user", content=_PROTOCOL_REPAIR_HINT))
                continue

            safe_answer = _safe_final_answer(content, state)
            answer = safe_answer
            stop_reason = "final_answer"
            _apply_numeric_fidelity(state, answer)
            recorder.record(
                "final_answer",
                f"final answer ({len(answer)} characters)",
                round_index=llm_round,
                details={"status": "completed"},
            )
            break

        result = GeoAgentResponse(
            answer=answer,
            sources=list(state.sources),
            trace=recorder.as_list(),
            geojson=state.geojson,
            feature_count=state.feature_count,
            passages=list(state.passages),
            overpass_query=state.overpass_query,
            warnings=list(state.warnings),
            errors=list(state.errors),
            stop_reason=stop_reason,
            model=self._llm.model_name,
            effective_limit=state.effective_limit,
            scope_summary=state.scope_summary,
            validated_tags=list(state.validated_tags),
            live_query_failed=state.live_query_failed,
            live_error_code=state.live_error_code,
            analysis=state.analysis,
            energy_analysis=state.energy_analysis,
            charts=list(state.charts),
            conversation_id=conversation_id,
        )
        return await self._finish_with_memory(request, result, tool_context, conversation_id)

    async def _finish_with_memory(
        self,
        request: GeoAgentRequest,
        response: GeoAgentResponse,
        tool_context: ToolContext,
        conversation_id: str,
        *,
        parent_execution_id: uuid.UUID | None = None,
        incremental: Any = None,
    ) -> GeoAgentResponse:
        memory_trace = response.execution_memory
        if self._execution_memory is not None:
            saved = await self._execution_memory.persist(
                request=request,
                response=response,
                tool_context=tool_context,
                conversation_id=conversation_id,
                parent_execution_id=parent_execution_id,
                provider=None,
                radius_m=incremental.reuse.radius_m if incremental is not None else None,
                feature_concept=(
                    incremental.reuse.feature_concept if incremental is not None else None
                ),
                analysis_goal=(
                    incremental.reuse.comparison_goal if incremental is not None else None
                ),
                inferred_goal=(
                    incremental.reuse.inferred_goal if incremental is not None else None
                ),
            )
            if memory_trace is None:
                memory_trace = self._execution_memory.trace_for(
                    conversation_id=conversation_id,
                    plan=incremental,
                    snapshot=saved,
                )
            elif saved is not None:
                memory_trace = memory_trace.model_copy(
                    update={"memory_execution_id": str(saved.execution_id)}
                )
        return response.model_copy(
            update={
                "conversation_id": conversation_id,
                "execution_memory": memory_trace,
            }
        )

    async def _execute_calls(
        self,
        calls: tuple[ToolCall, ...],
        *,
        messages: list[ChatMessage],
        state: ResultAccumulator,
        recorder: ListTraceRecorder,
        round_index: int,
        user_message: str,
        tool_context: ToolContext,
        invalid_call_counts: dict[str, int],
    ) -> str | None:
        """Execute tool calls. Return a stop answer if identical invalid calls repeat."""
        for call in calls:
            details = _tool_call_details(call)
            arg_keys = sorted(str(key) for key in call.arguments)
            recorder.record(
                "tool_call",
                f"requested {call.name} args={summarise_arguments(call.arguments)}",
                round_index=round_index,
                tool_name=call.name,
                details=details,
            )

            gate_error = _pre_invoke_gate(
                call,
                state=state,
                user_message=user_message,
                tool_context=tool_context,
            )
            if gate_error is not None:
                logged_ids = safe_logged_energy_ids(call.arguments)
                logger.info(
                    "tool_validation tool=%s argument_keys=%s network_id=%s "
                    "capability_id=%s validation_status=failed error_code=%s",
                    call.name,
                    arg_keys,
                    logged_ids.get("network_id", "-"),
                    logged_ids.get("capability_id", "-"),
                    gate_error.code,
                )
                recorder.record(
                    "tool_error",
                    f"Tool '{call.name}' failed ({gate_error.code}): {gate_error.message}",
                    round_index=round_index,
                    tool_name=call.name,
                    error_code=gate_error.code,
                    details={
                        "status": "failed",
                        "error_code": gate_error.code,
                        "tool_name": call.name,
                        "argument_keys": arg_keys,
                    },
                )
                if gate_error.code == "tool_not_eligible":
                    state.note_warning(f"{gate_error.code}: {gate_error.message}")
                else:
                    state.note_warning(f"{gate_error.code}: {gate_error.message}")
                signature = _invalid_call_signature(call.name, arg_keys, gate_error.code)
                invalid_call_counts[signature] = invalid_call_counts.get(signature, 0) + 1
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=(
                            f"Tool '{call.name}' failed ({gate_error.code}): {gate_error.message}"
                        ),
                        name=call.name,
                        tool_call_id=call.id,
                    )
                )
                if (
                    gate_error.code == "tool_argument_error"
                    and invalid_call_counts[signature] >= _MAX_IDENTICAL_ARGUMENT_FAILURES
                ):
                    state.note_error(
                        "tool_argument_error",
                        "repeated identical invalid tool arguments",
                    )
                    recorder.record(
                        "stopped",
                        "repeated identical invalid tool call",
                        round_index=round_index,
                        error_code="tool_argument_error",
                        details={"status": "failed"},
                    )
                    return (
                        "The model repeated the same invalid tool arguments. "
                        "No geographic query was completed."
                    )
                continue

            started = time.perf_counter()
            invocation = await self._registry.invoke(call, tool_context)
            duration_ms = int((time.perf_counter() - started) * 1000)
            energy_stop: str | None = None
            if invocation.ok:
                logged_ids = safe_logged_energy_ids(call.arguments)
                logger.info(
                    "tool_validation tool=%s argument_keys=%s network_id=%s "
                    "capability_id=%s validation_status=passed",
                    call.name,
                    arg_keys,
                    logged_ids.get("network_id", "-"),
                    logged_ids.get("capability_id", "-"),
                )
                result_details = {
                    "status": "completed",
                    "tool_name": call.name,
                    "duration_ms": duration_ms,
                    **_payload_details(invocation.payload),
                }
                recorder.record(
                    "tool_result",
                    describe_payload(invocation.payload),
                    round_index=round_index,
                    tool_name=call.name,
                    details=result_details,
                )
                state.absorb(call.name, invocation.payload)
                if call.name == QUERY_OSM:
                    tags = _summarise_tags(call.arguments.get("tags"))
                    state.validated_tags = tags
                    if tool_context.grounding.tags is None and tags:
                        tool_context.grounding.tags = list(tags)
                if call.name == SEARCH_OSM_KNOWLEDGE:
                    _apply_documentation_grounding(state, tool_context)
            else:
                code = invocation.error_code or "tool_execution_error"
                logged_ids = safe_logged_energy_ids(call.arguments)
                logger.info(
                    "tool_execution tool=%s argument_keys=%s network_id=%s "
                    "capability_id=%s validation_status=passed "
                    "execution_status=failed error_code=%s",
                    call.name,
                    arg_keys,
                    logged_ids.get("network_id", "-"),
                    logged_ids.get("capability_id", "-"),
                    code,
                )
                error_details: dict[str, Any] = {
                    "status": "failed",
                    "tool_name": call.name,
                    "error_code": code,
                    "duration_ms": duration_ms,
                    "argument_keys": arg_keys,
                    **safe_logged_energy_ids(call.arguments),
                }
                if invocation.failure_meta:
                    for key in (
                        "validated_tags",
                        "scope_type",
                        "named_place",
                        "effective_limit",
                        "upstream_status",
                        "overpass_attempts",
                        "grounding_remains_valid",
                    ):
                        value = invocation.failure_meta.get(key)
                        if value is not None:
                            error_details[key] = value
                recorder.record(
                    "tool_error",
                    _tool_error_trace_message(call.name, code, invocation),
                    round_index=round_index,
                    tool_name=call.name,
                    error_code=code,
                    details=error_details,
                )
                # Correctable argument mistakes stay as warnings until a final stop.
                if code == "tool_argument_error":
                    state.note_warning(f"{code}: {invocation.observation}")
                    signature = _invalid_call_signature(call.name, arg_keys, code)
                    invalid_call_counts[signature] = invalid_call_counts.get(signature, 0) + 1
                    if invalid_call_counts[signature] >= _MAX_IDENTICAL_ARGUMENT_FAILURES:
                        state.note_error(
                            "tool_argument_error",
                            "repeated identical invalid tool arguments",
                        )
                        recorder.record(
                            "stopped",
                            "repeated identical invalid tool call",
                            round_index=round_index,
                            error_code="tool_argument_error",
                            details={"status": "failed"},
                        )
                        messages.append(
                            ChatMessage(
                                role="tool",
                                content=invocation.observation,
                                name=call.name,
                                tool_call_id=call.id,
                            )
                        )
                        return (
                            "The model repeated the same invalid tool arguments. "
                            "No geographic query was completed."
                        )
                elif call.name == ANALYZE_ENERGY_GRID and _is_terminal_energy_error(code):
                    public = invocation.public_error or invocation.observation
                    state.note_error(code, public)
                    state.energy_workflow_stopped = True
                    state.energy_terminal_error_code = code
                    energy_stop = _energy_internal_stop_answer(code)
                else:
                    public = invocation.public_error or invocation.observation
                    state.note_error(code, public)
                    if call.name == QUERY_OSM and invocation.failure_meta:
                        state.note_live_query_failure(invocation.failure_meta)
            # Compact observation only — never the structured payload / GeoJSON.
            messages.append(
                ChatMessage(
                    role="tool",
                    content=invocation.observation,
                    name=call.name,
                    tool_call_id=call.id,
                )
            )
            if energy_stop is not None:
                recorder.record(
                    "stopped",
                    f"energy workflow stopped ({state.energy_terminal_error_code})",
                    round_index=round_index,
                    tool_name=call.name,
                    error_code=state.energy_terminal_error_code,
                    details={
                        "status": "failed",
                        "error_code": state.energy_terminal_error_code,
                        "do_not_replan": True,
                    },
                )
                return energy_stop
        return None


def _invalid_call_signature(tool_name: str, argument_keys: list[str], error_code: str) -> str:
    return f"{tool_name}|{','.join(argument_keys)}|{error_code}"


def _is_terminal_energy_error(code: str) -> bool:
    return code in _ENERGY_TERMINAL_ERROR_CODES


def _energy_internal_stop_answer(code: str) -> str:
    return (
        "The selected GeoLoadST analysis stopped because of an internal plugin or "
        f"engine failure ({code}). Another scientific method was not run, because "
        "it would not answer the same question. No charts or map overlay were invented."
    )


def _model_facing_tools(
    definitions: list[Any],
    *,
    state: ResultAccumulator,
    user_message: str,
    places: PlaceRegistry | None = None,
) -> list[Any]:
    """Filter model-facing tools by current request state.

    Tool Registry still contains every registered tool for execution authority.
    """
    has_datasets = bool(state.datasets.refs())
    analytical = bool(_ANALYTICAL_INTENT.search(user_message))
    energy = is_energy_grid_request(user_message)
    has_places = bool(places.refs()) if places is not None else False
    out: list[Any] = []
    for item in definitions:
        name = getattr(item, "name", None)
        if energy and name in {
            SEARCH_OSM_KNOWLEDGE,
            RESOLVE_PLACE,
            QUERY_OSM,
            ANALYZE_FEATURES,
        }:
            continue
        if name == ANALYZE_ENERGY_GRID and (not energy or state.energy_workflow_stopped):
            continue
        if name == ANALYZE_FEATURES and not (has_datasets and analytical):
            continue
        if (
            name == QUERY_OSM
            and analytical
            and "within" in user_message.lower()
            and places is not None
            and not has_places
            and not has_datasets
            and should_use_multi_target_runner(user_message)
        ):
            # Landmark-radius comparisons need place_ref before query_osm is useful.
            # Named city queries still keep query_osm eligible.
            continue
        out.append(item)
    return out


def _analyze_features_eligible(state: ResultAccumulator, user_message: str) -> bool:
    return bool(state.datasets.refs()) and bool(_ANALYTICAL_INTENT.search(user_message))


def _apply_documentation_grounding(state: ResultAccumulator, tool_context: ToolContext) -> None:
    """Lock documented tag selection after search_osm_knowledge when clear."""
    if tool_context.grounding.tags:
        return
    hints = state.documented_tag_hints()
    if not hints:
        return
    preferred = [hint for hint in hints if hint.lower() == "leisure=park"]
    selected = preferred or hints[:1]
    tool_context.grounding.tags = list(selected)
    state.absorb_grounding_tags(list(selected))


def _pre_invoke_gate(
    call: ToolCall,
    *,
    state: ResultAccumulator,
    user_message: str,
    tool_context: ToolContext,
) -> ToolArgumentError | ToolNotEligibleError | None:
    """Reject ineligible or dependency-incomplete calls before registry execution."""
    if call.name == ANALYZE_FEATURES:
        if not _analyze_features_eligible(state, user_message):
            if not state.datasets.refs():
                return ToolNotEligibleError(
                    "analyze_features is not eligible until query_osm has produced "
                    "dataset_ref values in this request"
                )
            return ToolNotEligibleError(
                "analyze_features is not eligible for this request (no analytical intent)"
            )
        targets = call.arguments.get("targets")
        if isinstance(targets, list):
            for target in targets:
                if not isinstance(target, dict):
                    continue
                ref = target.get("dataset_ref")
                if isinstance(ref, str) and ref not in state.datasets.refs():
                    known = ", ".join(state.datasets.refs()) or "none"
                    return ToolArgumentError(
                        f"unknown dataset_ref '{ref}'; available in this request: {known}. "
                        "Use only dataset_ref values returned by query_osm observations; "
                        "do not invent future references."
                    )
        return None

    if call.name == ANALYZE_ENERGY_GRID:
        if state.energy_workflow_stopped:
            code = state.energy_terminal_error_code or "energy_plugin_internal_error"
            return ToolNotEligibleError(
                "analyze_energy_grid is not eligible after an internal GeoLoadST "
                f"failure ({code}). Do not select a different capability_id."
            )
        network_id = call.arguments.get("network_id")
        has_loaded = bool(tool_context.analysis.energy_network_id or state.last_simbench_network_id)
        if not (isinstance(network_id, str) and network_id.strip()) and not has_loaded:
            return ToolNotEligibleError(
                "analyze_energy_grid is not eligible until simbench_query has loaded a network"
            )
        return None

    if call.name == QUERY_OSM:
        trust_error = untrusted_spatial_scope_error(
            call.arguments,
            user_message,
            places=tool_context.places,
        )
        if trust_error is not None:
            return ToolArgumentError(trust_error)
        return None
    return None


def _llm_turn_summary(response: LLMResponse) -> str:
    if response.has_tool_calls:
        names = ", ".join(call.name for call in response.tool_calls)
        return f"model requested tool(s): {names}"
    return "model returned a final answer candidate"


def _assistant_tool_stub(calls: tuple[ToolCall, ...]) -> str:
    payload = {"tool_calls": [{"name": call.name, "arguments": call.arguments} for call in calls]}
    return json.dumps(payload, ensure_ascii=False)


def _needs_protocol_repair(content: str, response: LLMResponse) -> bool:
    if response.has_tool_calls or response.is_protocol_error:
        return False
    if not content:
        return True
    # Anything that still looks like prompted protocol must not become an answer.
    if _PROTOCOL_LEAK_HINT.search(content):
        return True
    return content.lstrip().startswith("{")


def _safe_final_answer(content: str, state: ResultAccumulator) -> str:
    """Never expose prompted protocol objects as the user-facing answer."""
    if not content or _PROTOCOL_LEAK_HINT.search(content) or content.lstrip().startswith("{"):
        return (
            _fallback_answer(state)
            if state.feature_count is not None or state.passages or state.live_query_failed
            else (
                _PROTOCOL_FAILURE_ANSWER
                if not state.errors
                else "No final answer was produced because the model response was invalid."
            )
        )
    if state.live_query_failed and _looks_like_ungrounded_tag_invention(content):
        return _grounded_live_failure_answer(state)
    return content


def _tool_error_trace_message(tool_name: str, code: str, invocation: Any) -> str:
    if tool_name == QUERY_OSM and code.startswith("overpass_"):
        status = None
        attempts = None
        if invocation.failure_meta:
            status = invocation.failure_meta.get("upstream_status")
            attempts = invocation.failure_meta.get("overpass_attempts")
        parts = [f"{tool_name} failed ({code})"]
        if status is not None:
            parts.append(f"upstream_status={status}")
        if attempts is not None:
            parts.append(f"attempts={attempts}")
        return " ".join(parts)
    public = getattr(invocation, "public_error", None)
    if isinstance(public, str) and public.strip():
        return public
    return f"Tool '{tool_name}' failed ({code})"


def _looks_like_ungrounded_tag_invention(answer: str) -> bool:
    """Heuristic: reject post-timeout answers that invent unsupported tags."""
    lowered = answer.lower()
    if "public_park" in lowered or "leisure=public" in lowered:
        return True
    if "overpass ql" in lowered or "write an overpass" in lowered:
        return True
    if re.search(r"\btry\b.{0,40}\b(tag|leisure|amenity|public_park)\b", lowered):
        return True
    return bool(re.search(r"\b(lat|lon|latitude|longitude)\b\s*=\s*-?\d", lowered))


def _grounded_live_failure_answer(state: ResultAccumulator) -> str:
    tags = ", ".join(state.validated_tags) if state.validated_tags else "the validated OSM tag"
    scope = state.scope_summary or "the requested area"
    if scope.startswith("Search area: "):
        scope = scope.removeprefix("Search area: ")
    code = state.live_error_code or "overpass_error"
    if code == "overpass_timeout":
        reason = "the configured Overpass service timed out"
    elif code == "overpass_rate_limited":
        reason = "the configured Overpass service rate-limited the request"
    else:
        reason = "the configured Overpass service failed"
    docs = (
        "The OSM documentation lookup succeeded and "
        if state.passages
        else "The request was validated and "
    )
    return (
        f"{docs}public parks / the requested features were resolved to {tags}. "
        f"The live query for {scope} could not be completed because {reason}. "
        "No live features were returned. Retrying later may succeed. "
        "The tag choice remains valid; a temporary service failure is not evidence "
        "that a different OSM tag is required."
    )


def _budget_stop_answer(reason: StopReason, state: ResultAccumulator) -> str:
    parts = [
        f"Stopped because the tool budget was reached ({reason}).",
        "Do not invent additional tool results.",
    ]
    if state.feature_count is not None:
        parts.append(f"Live features collected so far: {state.feature_count}.")
    if state.passages:
        parts.append(f"Documentation passages collected: {len(state.passages)}.")
    return " ".join(parts)


def _fallback_answer(state: ResultAccumulator) -> str:
    if any(err.startswith("llm_timeout:") for err in state.errors):
        return "The local model timed out during request planning. No live OSM query was executed."
    if any(
        err.startswith("place_resolution_error:") or err.startswith("place_ambiguous:")
        for err in state.errors
    ):
        return "One or more comparison locations could not be resolved reliably."
    if state.live_query_failed:
        return _grounded_live_failure_answer(state)
    if state.errors and state.feature_count is None and not state.passages:
        return "The geographic query failed before live data could be retrieved."
    if state.feature_count == 0:
        return "The request completed, but no matching OpenStreetMap features were found."
    if state.feature_count:
        return f"Retrieved {state.feature_count} live OpenStreetMap feature(s)."
    if state.passages:
        return "Retrieved OSM documentation passages relevant to the request."
    return "No answer was produced."


def _tool_call_details(call: ToolCall) -> dict[str, Any]:
    details: dict[str, Any] = {
        "status": "info",
        "tool_name": call.name,
    }
    args = call.arguments
    if call.name == RESOLVE_PLACE:
        query = args.get("query")
        if isinstance(query, str):
            details["place_query"] = query
        return details
    if call.name == ANALYZE_ENERGY_GRID:
        details.update(safe_logged_energy_ids(args))
        return details
    if call.name == ANALYZE_FEATURES:
        details["analysis_type"] = args.get("analysis_type")
        targets = args.get("targets")
        if isinstance(targets, list):
            details["target_count"] = len(targets)
            refs = [
                t.get("dataset_ref")
                for t in targets
                if isinstance(t, dict) and isinstance(t.get("dataset_ref"), str)
            ]
            if refs:
                details["dataset_refs"] = refs
        metrics = args.get("metrics")
        if isinstance(metrics, list):
            for metric in metrics:
                if isinstance(metric, dict) and metric.get("role") == "primary":
                    details["primary_metric"] = metric.get("metric")
                    break
        return details
    if call.name != QUERY_OSM:
        return details
    if isinstance(args.get("place"), str):
        details["scope_type"] = "place"
        details["named_place"] = args["place"]
    elif isinstance(args.get("place_ref_scope"), dict):
        details["scope_type"] = "place_ref"
        ref = args["place_ref_scope"].get("place_ref")
        if isinstance(ref, str):
            details["place_ref"] = ref
        radius = args["place_ref_scope"].get("radius_m")
        if isinstance(radius, int):
            details["radius_meters"] = radius
    elif isinstance(args.get("point"), dict):
        details["scope_type"] = "point"
        radius = args["point"].get("radius_m")
        if isinstance(radius, int):
            details["radius_meters"] = radius
    elif isinstance(args.get("bbox"), dict):
        details["scope_type"] = "bbox"
    tags = args.get("tags")
    validated = _summarise_tags(tags)
    if validated:
        details["validated_tags"] = validated
    limit = args.get("limit")
    if isinstance(limit, int):
        details["effective_limit"] = limit
    return details


def _summarise_tags(tags: Any) -> list[str]:
    if not isinstance(tags, list):
        return []
    out: list[str] = []
    for item in tags:
        if isinstance(item, dict):
            key = item.get("key")
            value = item.get("value")
            if isinstance(key, str) and key:
                out.append(key if value is None else f"{key}={value}")
        elif isinstance(item, list | tuple) and len(item) == 2:
            key, value = item[0], item[1]
            if isinstance(key, str) and key:
                out.append(key if value is None else f"{key}={value}")
    return out


def _payload_details(payload: Any) -> dict[str, Any]:
    details: dict[str, Any] = {}
    place_ref = getattr(payload, "place_ref", None)
    if isinstance(place_ref, str) and getattr(payload, "status", None) == "resolved":
        details["place_ref"] = place_ref
        label = getattr(payload, "label", None)
        if isinstance(label, str):
            details["target"] = label
        source = getattr(payload, "source", None)
        if isinstance(source, str):
            details["source"] = source
        details["status"] = "completed"
    feature_count = getattr(payload, "feature_count", None)
    if isinstance(feature_count, int):
        details["feature_count"] = feature_count
    analysis_target = getattr(payload, "analysis_target", None)
    if isinstance(analysis_target, str) and analysis_target:
        details["analysis_target"] = analysis_target
    query = getattr(payload, "overpass_query", None)
    if isinstance(query, str) and query:
        details["overpass_built"] = True
    analysis_status = getattr(payload, "analysis_status", None)
    if isinstance(analysis_status, str):
        details["analysis_status"] = analysis_status
    capability = getattr(payload, "capability", None)
    if isinstance(capability, str) and capability:
        details["capability_id"] = capability
    requested_capability = getattr(payload, "requested_capability", None)
    if isinstance(requested_capability, str) and requested_capability:
        details["requested_capability"] = requested_capability
    metrics_computed = getattr(payload, "metrics_computed", None)
    if isinstance(metrics_computed, int):
        details["metrics_computed"] = metrics_computed
    decision_trace = getattr(payload, "decision_trace", None)
    if decision_trace is not None:
        revision = getattr(decision_trace, "plan_revision_count", None)
        if isinstance(revision, int):
            details["plan_revision_count"] = revision
    return details


def _apply_numeric_fidelity(state: ResultAccumulator, answer: str) -> None:
    """Guard model narrative numbers against deterministic analytics results."""
    block = state.analysis
    if block is None or block.result is None or block.comparison is None or block.report is None:
        return
    if not answer.strip():
        return
    updated = verify_numeric_fidelity(
        block.report,
        block.result,
        block.comparison,
        narratives=[answer],
    )
    if updated.numeric_fidelity == "unverified_numbers_removed":
        state.note_warning(
            "report_numeric_mismatch: unverified value(s) removed from narrative sections"
        )
        state.analysis = block.model_copy(update={"report": updated})
