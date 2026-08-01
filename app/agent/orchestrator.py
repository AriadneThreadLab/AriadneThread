"""Bounded planner-executor orchestration.

The model proposes tool calls; the Tool Registry is the only execution path.
Compact observations are appended to the conversation. Full GeoJSON and
documentation passages accumulate in :class:`ResultAccumulator` for the response.
"""

from __future__ import annotations

import json
import re

from app.agent.accumulator import ResultAccumulator
from app.agent.contracts import (
    GeoAgentRequest,
    GeoAgentResponse,
    ListTraceRecorder,
    StopReason,
    describe_payload,
    summarise_arguments,
)
from app.agent.loop import LoopBudget, LoopLimits
from app.agent.prompts import SYSTEM_PROMPT
from app.core.errors import LLMError, LLMProtocolError, LLMTimeoutError
from app.llm.contracts import ChatMessage, LLMProvider, LLMResponse, ToolCall
from app.tools.registry import ToolRegistry

_MAX_PROTOCOL_REPAIRS = 2
_BROKEN_JSON_HINT = re.compile(r"^\s*\{")


class PlannerExecutorAgent:
    """One-request planner-executor over registered OSM tools."""

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        limits: LoopLimits,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._limits = limits

    async def run(self, request: GeoAgentRequest) -> GeoAgentResponse:
        recorder = ListTraceRecorder()
        state = ResultAccumulator()
        budget = LoopBudget(self._limits)
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=request.message),
        ]
        recorder.record(
            "request_received",
            f"received request ({len(request.message)} characters)",
        )

        answer = ""
        stop_reason: StopReason = "final_answer"
        protocol_repairs = 0
        llm_round = 0

        while True:
            llm_round += 1
            try:
                response = await self._llm.chat(
                    messages,
                    tools=self._registry.definitions(),
                )
            except LLMTimeoutError as exc:
                state.note_error(exc.code, exc.message)
                recorder.record(
                    "stopped",
                    f"LLM timed out: {exc.message}",
                    round_index=llm_round,
                    error_code=exc.code,
                )
                stop_reason = "llm_error"
                answer = "The language model timed out before a final answer was produced."
                break
            except LLMProtocolError as exc:
                state.note_error(exc.code, exc.message)
                recorder.record(
                    "stopped",
                    f"LLM protocol error: {exc.message}",
                    round_index=llm_round,
                    error_code=exc.code,
                )
                stop_reason = "llm_error"
                answer = "The language model returned an unusable response."
                break
            except LLMError as exc:
                state.note_error(exc.code, exc.message)
                recorder.record(
                    "stopped",
                    f"LLM error: {exc.message}",
                    round_index=llm_round,
                    error_code=exc.code,
                )
                stop_reason = "llm_error"
                answer = "The language model failed before a final answer was produced."
                break

            recorder.record(
                "llm_turn",
                _llm_turn_summary(response),
                round_index=llm_round,
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
                    )
                    break

                admitted = budget.admit(response.tool_calls)
                dropped = len(response.tool_calls) - len(admitted)
                if dropped:
                    state.note_warning(
                        f"Dropped {dropped} tool call(s) that exceeded the remaining call budget"
                    )

                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=response.content or _assistant_tool_stub(response.tool_calls),
                    )
                )
                await self._execute_calls(
                    admitted,
                    messages=messages,
                    state=state,
                    recorder=recorder,
                    round_index=budget.rounds_used,
                )

                if not budget.can_run_tools() and not admitted:
                    exhausted = budget.exhausted_reason() or "max_tool_calls"
                    stop_reason = exhausted
                    answer = _budget_stop_answer(exhausted, state)
                    recorder.record(
                        "stopped",
                        f"tool budget exhausted ({exhausted})",
                        round_index=llm_round,
                        error_code=exhausted,
                    )
                    break
                continue

            content = (response.content or "").strip()
            if _needs_protocol_repair(content, response):
                protocol_repairs += 1
                state.note_error(
                    "llm_protocol_error",
                    "model reply was not valid tool_calls/final_answer JSON",
                )
                recorder.record(
                    "tool_error",
                    "malformed structured model reply",
                    round_index=llm_round,
                    error_code="llm_protocol_error",
                )
                if protocol_repairs > _MAX_PROTOCOL_REPAIRS:
                    stop_reason = "llm_error"
                    answer = "The language model did not follow the required response protocol."
                    recorder.record(
                        "stopped",
                        "protocol repair budget exhausted",
                        round_index=llm_round,
                        error_code="llm_protocol_error",
                    )
                    break
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "Protocol error: reply with ONLY a JSON object of either "
                            '{"tool_calls":[{"name":"...","arguments":{...}}]} '
                            'or {"final_answer":"..."}.'
                        ),
                    )
                )
                continue

            answer = content or _fallback_answer(state)
            stop_reason = "final_answer"
            recorder.record(
                "final_answer",
                f"final answer ({len(answer)} characters)",
                round_index=llm_round,
            )
            break

        return GeoAgentResponse(
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
        )

    async def _execute_calls(
        self,
        calls: tuple[ToolCall, ...],
        *,
        messages: list[ChatMessage],
        state: ResultAccumulator,
        recorder: ListTraceRecorder,
        round_index: int,
    ) -> None:
        for call in calls:
            recorder.record(
                "tool_call",
                f"requested {call.name} args={summarise_arguments(call.arguments)}",
                round_index=round_index,
                tool_name=call.name,
            )
            invocation = await self._registry.invoke(call)
            if invocation.ok:
                recorder.record(
                    "tool_result",
                    describe_payload(invocation.payload),
                    round_index=round_index,
                    tool_name=call.name,
                )
                state.absorb(call.name, invocation.payload)
            else:
                code = invocation.error_code or "tool_execution_error"
                recorder.record(
                    "tool_error",
                    invocation.observation,
                    round_index=round_index,
                    tool_name=call.name,
                    error_code=code,
                )
                state.note_error(code, invocation.observation)
            # Compact observation only — never the structured payload / GeoJSON.
            messages.append(
                ChatMessage(
                    role="tool",
                    content=invocation.observation,
                    name=call.name,
                    tool_call_id=call.id,
                )
            )


def _llm_turn_summary(response: LLMResponse) -> str:
    if response.has_tool_calls:
        names = ", ".join(call.name for call in response.tool_calls)
        return f"model requested tool(s): {names}"
    return "model returned a final answer candidate"


def _assistant_tool_stub(calls: tuple[ToolCall, ...]) -> str:
    payload = {"tool_calls": [{"name": call.name, "arguments": call.arguments} for call in calls]}
    return json.dumps(payload, ensure_ascii=False)


def _needs_protocol_repair(content: str, response: LLMResponse) -> bool:
    if response.has_tool_calls:
        return False
    if not content:
        return True
    if _BROKEN_JSON_HINT.match(content) and "final_answer" not in content:
        # Looks like structured JSON that failed to parse into a final answer.
        try:
            json.loads(content)
        except json.JSONDecodeError:
            return True
        # Valid JSON object without final_answer / tool_calls is still a protocol miss.
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return True
        if isinstance(payload, dict) and "final_answer" not in payload:
            return True
    return False


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
    if state.feature_count == 0:
        return "No live OpenStreetMap features matched the query. Do not invent results."
    if state.feature_count:
        return f"Retrieved {state.feature_count} live OpenStreetMap feature(s)."
    if state.passages:
        return "Retrieved OSM documentation passages relevant to the request."
    return "No answer was produced."
