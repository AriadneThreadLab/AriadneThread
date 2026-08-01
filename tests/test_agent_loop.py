"""Loop bounds and trace safety."""

from __future__ import annotations

import pytest
from app.agent.contracts import ListTraceRecorder, TraceEvent
from app.agent.loop import LoopBudget, LoopLimits
from app.agent.prompts import SYSTEM_PROMPT
from app.llm.contracts import ToolCall


def _calls(count: int) -> tuple[ToolCall, ...]:
    return tuple(ToolCall(id=f"c{i}", name="query_osm", arguments={}) for i in range(count))


def test_rounds_are_capped():
    budget = LoopBudget(LoopLimits(max_tool_rounds=2, max_tool_calls=10))
    assert budget.admit(_calls(1))
    assert budget.admit(_calls(1))
    assert budget.admit(_calls(1)) == ()
    assert budget.exhausted_reason() == "max_tool_rounds"


def test_calls_are_capped_across_rounds():
    budget = LoopBudget(LoopLimits(max_tool_rounds=5, max_tool_calls=3))
    assert len(budget.admit(_calls(2))) == 2
    assert len(budget.admit(_calls(2))) == 1
    assert budget.calls_used == 3
    assert budget.exhausted_reason() == "max_tool_calls"


def test_surplus_calls_in_one_round_are_dropped_not_fatal():
    budget = LoopBudget(LoopLimits(max_tool_rounds=3, max_tool_calls=2))
    admitted = budget.admit(_calls(5))
    assert len(admitted) == 2
    assert [call.id for call in admitted] == ["c0", "c1"]


def test_budget_is_available_until_a_limit_is_hit():
    budget = LoopBudget(LoopLimits(max_tool_rounds=2, max_tool_calls=2))
    assert budget.can_run_tools()
    assert budget.exhausted_reason() is None
    budget.admit(_calls(2))
    assert not budget.can_run_tools()


@pytest.mark.parametrize(("rounds", "calls"), [(0, 1), (1, 0), (-1, 5)])
def test_limits_must_be_positive(rounds, calls):
    with pytest.raises(ValueError):
        LoopLimits(max_tool_rounds=rounds, max_tool_calls=calls)


def test_trace_records_operational_events_only():
    recorder = ListTraceRecorder()
    recorder.record("request_received", "received request")
    recorder.record(
        "tool_call", "searched OSM documentation", round_index=1, tool_name="search_osm_knowledge"
    )
    recorder.record("tool_result", "received 128 feature(s)", round_index=2, tool_name="query_osm")

    events = recorder.events
    assert [event.kind for event in events] == ["request_received", "tool_call", "tool_result"]
    assert events[1].tool_name == "search_osm_knowledge"
    assert all(isinstance(event, TraceEvent) for event in events)
    assert all(event.at.tzinfo is not None for event in events)


def test_each_recorder_is_independent():
    first, second = ListTraceRecorder(), ListTraceRecorder()
    first.record("tool_call", "called Overpass")
    assert second.events == ()


def test_system_prompt_separates_documentation_from_live_data():
    assert "search_osm_knowledge" in SYSTEM_PROMPT
    assert "query_osm" in SYSTEM_PROMPT
    assert "Never present documentation as live map data." in SYSTEM_PROMPT
