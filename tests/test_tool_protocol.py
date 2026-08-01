"""Prompted tool-call protocol and reasoning removal."""

from __future__ import annotations

from app.llm.contracts import ToolDefinition
from app.llm.tool_protocol import parse_reply, render_tool_instructions, strip_reasoning


def test_reasoning_blocks_are_removed():
    raw = "<think>The user wants parks. I should search.</think>\nParks are leisure=park."
    assert strip_reasoning(raw) == "Parks are leisure=park."


def test_unclosed_reasoning_block_is_removed():
    assert strip_reasoning("Answer.\n<think>still thinking about") == "Answer."


def test_tool_call_payload_is_parsed():
    raw = (
        "<think>hidden</think>\n"
        '{"tool_calls": [{"name": "query_osm", '
        '"arguments": {"place": "Berlin", "tags": [{"key": "leisure", "value": "park"}]}}]}'
    )
    parsed = parse_reply(raw)
    assert parsed.content == ""
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "query_osm"
    assert call.arguments["place"] == "Berlin"
    assert call.id == "call_1"


def test_fenced_json_tool_call_is_parsed():
    raw = (
        '```json\n{"tool_calls": [{"name": "search_osm_knowledge", '
        '"arguments": {"query": "park tag"}}]}\n```'
    )
    parsed = parse_reply(raw)
    assert [call.name for call in parsed.tool_calls] == ["search_osm_knowledge"]


def test_tool_call_arguments_encoded_as_string_are_accepted():
    raw = '{"tool_calls": [{"name": "query_osm", "arguments": "{\\"place\\": \\"Berlin\\"}"}]}'
    parsed = parse_reply(raw)
    assert parsed.tool_calls[0].arguments == {"place": "Berlin"}


def test_multiple_tool_calls_get_unique_ids():
    raw = '{"tool_calls": [{"name": "a", "arguments": {}},{"name": "b", "arguments": {}}]}'
    parsed = parse_reply(raw)
    assert [call.id for call in parsed.tool_calls] == ["call_1", "call_2"]


def test_final_answer_payload_is_unwrapped():
    parsed = parse_reply('{"final_answer": "Berlin has many parks."}')
    assert parsed.content == "Berlin has many parks."
    assert parsed.tool_calls == ()


def test_plain_prose_is_returned_unchanged():
    parsed = parse_reply("<think>x</think>Berlin has many parks.")
    assert parsed.content == "Berlin has many parks."
    assert parsed.tool_calls == ()


def test_malformed_json_degrades_to_prose():
    parsed = parse_reply('{"tool_calls": [{"name": ')
    assert parsed.tool_calls == ()
    assert parsed.content.startswith("{")


def test_tool_entries_missing_a_name_are_ignored():
    parsed = parse_reply('{"tool_calls": [{"arguments": {}}]}')
    assert parsed.tool_calls == ()


def test_instructions_list_every_tool_and_forbid_invented_data():
    tools = [
        ToolDefinition(name="query_osm", description="live features", parameters_schema={}),
        ToolDefinition(
            name="search_osm_knowledge", description="documentation", parameters_schema={}
        ),
    ]
    instructions = render_tool_instructions(tools)
    assert "query_osm" in instructions
    assert "search_osm_knowledge" in instructions
    assert "tool_calls" in instructions
    assert "Never invent map data" in instructions
