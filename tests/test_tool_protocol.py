"""Prompted tool-call protocol and reasoning removal."""

from __future__ import annotations

from app.llm.contracts import ToolDefinition
from app.llm.tool_protocol import (
    normalize_prompted_text,
    parse_reply,
    render_tool_instructions,
    strip_reasoning,
)


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
    assert parsed.protocol_error is None
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "query_osm"
    assert call.arguments["place"] == "Berlin"
    assert call.id == "call_1"


def test_fenced_json_tool_call_is_normalized_and_parsed():
    raw = (
        '```json\n{"tool_calls": [{"name": "search_osm_knowledge", '
        '"arguments": {"query": "park tag"}}]}\n```'
    )
    assert '"tool_calls"' in normalize_prompted_text(raw)
    parsed = parse_reply(raw)
    assert parsed.protocol_error is None
    assert [call.name for call in parsed.tool_calls] == ["search_osm_knowledge"]
    assert parsed.content == ""


def test_observed_fenced_query_osm_payload_is_parsed_as_tool_calls():
    raw = """```json
{
  "tool_calls": [
    {
      "name": "query_osm",
      "arguments": {
        "tags": [["leisure", "park"]],
        "place": "Tehran, Iran",
        "limit": 20
      }
    }
  ]
}
```"""
    parsed = parse_reply(raw)
    assert parsed.protocol_error is None
    assert parsed.content == ""
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "query_osm"
    assert parsed.tool_calls[0].arguments["place"] == "Tehran, Iran"
    assert parsed.tool_calls[0].arguments["tags"] == [["leisure", "park"]]


def test_uppercase_json_fence_language_marker_is_accepted():
    raw = '```JSON\n{"final_answer": "Parks use leisure=park."}\n```'
    parsed = parse_reply(raw)
    assert parsed.content == "Parks use leisure=park."
    assert parsed.tool_calls == ()


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
    assert parsed.protocol_error is None


def test_plain_prose_is_returned_unchanged():
    parsed = parse_reply("<think>x</think>Berlin has many parks.")
    assert parsed.content == "Berlin has many parks."
    assert parsed.tool_calls == ()
    assert parsed.response_kind == "final_answer"


def test_malformed_json_returns_protocol_error_not_raw_payload():
    parsed = parse_reply('{"tool_calls": [{"name": ')
    assert parsed.tool_calls == ()
    assert parsed.content == ""
    assert parsed.protocol_error is not None
    assert parsed.response_kind == "invalid"


def test_both_tool_calls_and_final_answer_are_rejected():
    parsed = parse_reply('{"tool_calls": [{"name": "a", "arguments": {}}], "final_answer": "x"}')
    assert parsed.is_protocol_error
    assert parsed.tool_calls == ()
    assert parsed.content == ""


def test_unknown_top_level_fields_are_rejected():
    parsed = parse_reply('{"final_answer": "ok", "scratchpad": "secret"}')
    assert parsed.is_protocol_error
    assert "scratchpad" in (parsed.protocol_error or "")


def test_tool_entries_missing_a_name_are_protocol_errors():
    parsed = parse_reply('{"tool_calls": [{"arguments": {}}]}')
    assert parsed.is_protocol_error
    assert parsed.tool_calls == ()


def test_prose_before_json_fence_is_protocol_error():
    raw = (
        "I will call query_osm now.\n\n"
        '```json\n{"tool_calls":[{"name":"query_osm","arguments":{"place":"Tehran"}}]}\n```'
    )
    parsed = parse_reply(raw)
    assert parsed.is_protocol_error
    assert parsed.tool_calls == ()
    assert parsed.content == ""


def test_tool_names_as_top_level_keys_are_protocol_error():
    raw = '{"query_osm":{"place":"Tehran, Iran","max_items":20},"search_osm_knowledge":{}}'
    parsed = parse_reply(raw)
    assert parsed.is_protocol_error
    assert "tool_calls" in (parsed.protocol_error or "")


def test_args_key_instead_of_arguments_is_protocol_error():
    raw = '{"tool_calls":[{"name":"query_osm","args":{"place":"Tehran"}}]}'
    parsed = parse_reply(raw)
    assert parsed.is_protocol_error
    assert "arguments" in (parsed.protocol_error or "")


def test_over_escaped_quotes_inside_fenced_json_are_repaired_once():
    raw = (
        "```json\n"
        "{\n"
        '  "tool_calls": [\n'
        "    {\n"
        '      "name": "query_osm",\n'
        '      "arguments": {\n'
        '        "place": "Tehran, Iran",\n'
        '        "tags": [{\\"key\\": \\"leisure\\", \\"value\\": \\"park\\"}],\n'
        '        "limit": 20\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```"
    )
    parsed = parse_reply(raw)
    assert parsed.protocol_error is None
    assert parsed.tool_calls[0].name == "query_osm"
    assert parsed.tool_calls[0].arguments["tags"] == [{"key": "leisure", "value": "park"}]


def test_instructions_list_every_tool_and_forbid_invented_data():
    tools = [
        ToolDefinition(name="query_osm", description="live features", parameters_schema={}),
        ToolDefinition(
            name="search_osm_knowledge", description="documentation", parameters_schema={}
        ),
        ToolDefinition(name="analyze_features", description="analytics", parameters_schema={}),
    ]
    instructions = render_tool_instructions(tools)
    assert "query_osm" in instructions
    assert "search_osm_knowledge" in instructions
    assert "analyze_features" in instructions
    assert "tool_calls" in instructions
    assert "Never invent coordinates" in instructions
    assert '"tags":[{"key":"leisure","value":"park"}]' in instructions
    assert "Do NOT use for ordinary find/list/search" in instructions
