"""Offline coverage for Ollama-facing tool schema normalisation."""

from __future__ import annotations

import json

import pytest
from app.analytics.contracts import AnalysisPlan
from app.llm.contracts import ToolCall, ToolDefinition
from app.llm.ollama import OllamaProvider
from app.llm.tool_protocol import parse_reply, render_tool_instructions
from app.llm.tool_schema import normalize_tool_parameters_schema, schema_structure_summary
from app.osm.query_spec import OsmFeatureQuery
from app.tools.energy_tools import AnalyzeEnergyGridArgs
from app.tools.search_osm_knowledge import SearchOsmKnowledgeArgs
from app.tools.simbench_tools import SimBenchQueryArgs
from pydantic import ValidationError


def _all_tool_defs() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            "search_osm_knowledge",
            "search",
            SearchOsmKnowledgeArgs.model_json_schema(),
        ),
        ToolDefinition(
            "query_osm",
            "query",
            OsmFeatureQuery.model_json_schema(),
        ),
        ToolDefinition(
            "analyze_features",
            "analyze",
            AnalysisPlan.model_json_schema(),
        ),
        ToolDefinition(
            "simbench_query",
            "simbench",
            SimBenchQueryArgs.model_json_schema(),
        ),
        ToolDefinition(
            "analyze_energy_grid",
            "energy",
            AnalyzeEnergyGridArgs.model_json_schema(),
        ),
    ]


def test_normalize_strips_defs_refs_and_verbose_metadata():
    raw = OsmFeatureQuery.model_json_schema()
    assert "$defs" in raw
    normalised = normalize_tool_parameters_schema(raw)
    assert "$defs" not in normalised
    assert "$ref" not in json.dumps(normalised)
    assert "title" not in json.dumps(normalised)
    assert "default" not in json.dumps(normalised)
    assert normalised.get("type") == "object"
    assert "properties" in normalised


def test_each_registered_tool_schema_normalises():
    for tool in _all_tool_defs():
        normalised = normalize_tool_parameters_schema(tool.parameters_schema)
        assert normalised["type"] == "object"
        summary = schema_structure_summary(normalised)
        assert "$defs" not in summary["markers"]
        assert "$ref" not in summary["markers"]


def test_complete_registry_serialises_for_native_tools():
    encoded = [OllamaProvider._encode_tool(tool) for tool in _all_tool_defs()]
    assert len(encoded) == 5
    for item in encoded:
        assert item["type"] == "function"
        assert set(item["function"]) == {"name", "description", "parameters"}
        params = item["function"]["parameters"]
        assert params["type"] == "object"
        dumped = json.dumps(params)
        assert "$defs" not in dumped
        assert "$ref" not in dumped


def test_prompted_catalogue_is_compact_and_fits_budget():
    instructions = render_tool_instructions(_all_tool_defs())
    assert "Available tools:\n" in instructions
    catalogue = instructions.split("Available tools:\n", 1)[1]
    assert "\n  " not in catalogue
    assert len(instructions) < 8_000
    assert "$defs" not in instructions
    assert "analyze_features" in instructions
    assert "query_osm" in instructions
    assert "search_osm_knowledge" in instructions
    assert "simbench_query" in instructions
    assert "analyze_energy_grid" in instructions
    assert "topology_centrality" in instructions
    assert '"limit":20' in instructions


def test_prompted_and_native_tool_calls_share_internal_contract():
    prompted = parse_reply('{"tool_calls":[{"name":"query_osm","arguments":{"feature":"park"}}]}')
    native = OllamaProvider._decode_native_tool_calls(
        {"tool_calls": [{"function": {"name": "query_osm", "arguments": {"feature": "park"}}}]}
    )
    assert prompted.tool_calls
    assert native
    assert isinstance(prompted.tool_calls[0], ToolCall)
    assert isinstance(native[0], ToolCall)
    assert prompted.tool_calls[0].name == native[0].name == "query_osm"
    assert prompted.tool_calls[0].arguments == native[0].arguments == {"feature": "park"}


def test_pydantic_execution_validation_remains_strict():
    # Normalised advertising schemas must not weaken backend validation.
    with pytest.raises(ValidationError):
        OsmFeatureQuery.model_validate({"not_a_real_field": True})
    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate({"metrics": []})
