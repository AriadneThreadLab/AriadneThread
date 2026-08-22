"""Compact model-facing tool catalogue for prompted mode.

Authoritative validation remains on each tool's Pydantic args model. This module
only shapes what the local LLM sees in the system prompt.
"""

from __future__ import annotations

import json
from typing import Any

from app.llm.contracts import ToolDefinition
from app.llm.tool_schema import normalize_tool_parameters_schema

# Hand-authored compact shapes for registered tools. Keys must match
# the authoritative models; examples are illustrative only.
_COMPACT_SPECS: dict[str, dict[str, Any]] = {
    "search_osm_knowledge": {
        "purpose": (
            "Search local OSM Wiki documentation for tagging guidance. "
            "Returns documentation only — never live map features."
        ),
        "when_to_use": (
            "Semantic or ambiguous concepts (e.g. public parks) before query_osm. "
            "Skip when the user already gives an exact OSM tag such as leisure=park."
        ),
        "arguments": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 2, "maxLength": 500},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
        },
        "example": {"query": "public park OpenStreetMap tag", "top_k": 5},
    },
    "resolve_place": {
        "purpose": (
            "Resolve a named landmark to a trusted place_ref for place_ref_scope "
            "queries. Coordinates come from the geocoder — never invent them."
        ),
        "when_to_use": (
            "Landmarks needing a radius buffer (universities, squares). "
            "Not needed for simple city-named place queries like Tehran."
        ),
        "arguments": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 2,
                    "maxLength": 200,
                    "description": "Human-readable place only; no lat/lon.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
            },
        },
        "example": {"query": "University of Tehran, Tehran, Iran", "limit": 3},
    },
    "query_osm": {
        "purpose": (
            "Retrieve live OpenStreetMap features for ONE spatial target. "
            "Only source of live map data. Never invent coordinates or Overpass QL."
        ),
        "when_to_use": (
            "List/find/retrieve live features. Named place for cities; "
            "place_ref_scope after resolve_place for landmark radii. "
            "Call once per comparison target. Do not pass place as a list."
        ),
        "arguments": {
            "type": "object",
            "required": ["tags"],
            "properties": {
                "place": {
                    "type": "string",
                    "description": (
                        "Single named city/area string. Exactly one of "
                        "place|point|bbox|place_ref_scope."
                    ),
                },
                "place_ref_scope": {
                    "type": "object",
                    "required": ["place_ref", "radius_m"],
                    "properties": {
                        "place_ref": {"type": "string"},
                        "radius_m": {"type": "integer"},
                    },
                    "description": (
                        "Trusted resolve_place ref + user-stated radius_m (e.g. 2000)."
                    ),
                },
                "point": {
                    "type": "object",
                    "required": ["lat", "lon", "radius_m"],
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                        "radius_m": {"type": "integer"},
                    },
                    "description": "Only when the user supplied coordinates.",
                },
                "bbox": {
                    "type": "object",
                    "required": ["south", "west", "north", "east"],
                    "properties": {
                        "south": {"type": "number"},
                        "west": {"type": "number"},
                        "north": {"type": "number"},
                        "east": {"type": "number"},
                    },
                    "description": "Only when the user supplied bounds.",
                },
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "required": ["key"],
                        "properties": {
                            "key": {"type": "string"},
                            "value": {"type": "string"},
                        },
                    },
                    "description": (
                        'Tag filters as objects, e.g. [{"key":"leisure","value":"park"}].'
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "description": "Max features. Prefer 20 for exploratory city queries.",
                },
                "element_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["node", "way", "relation"]},
                },
                "include_geometry": {"type": "boolean"},
            },
        },
        "example": {
            "place": "Tehran, Iran",
            "tags": [{"key": "leisure", "value": "park"}],
            "limit": 20,
        },
    },
    "analyze_features": {
        "purpose": (
            "Deterministic statistics/comparisons over datasets already retrieved "
            "with query_osm in THIS request (dataset_ref like osm_result_1). "
            "Never invent numbers."
        ),
        "when_to_use": (
            "Only for analytical comparison questions after query_osm has returned "
            "dataset_ref values. Do NOT use for ordinary find/list/search requests."
        ),
        "arguments": {
            "type": "object",
            "required": [
                "analysis_type",
                "feature_concept",
                "comparison_goal",
                "targets",
                "metrics",
            ],
            "properties": {
                "analysis_type": {"type": "string", "enum": ["single_target", "comparison"]},
                "feature_concept": {"type": "string"},
                "comparison_goal": {"type": "string"},
                "targets": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["target_id", "label", "dataset_ref"],
                        "properties": {
                            "target_id": {"type": "string"},
                            "label": {"type": "string"},
                            "dataset_ref": {"type": "string"},
                        },
                    },
                },
                "metrics": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["metric", "inferred_goal"],
                        "properties": {
                            "metric": {"type": "string"},
                            "role": {"type": "string", "enum": ["primary", "supporting"]},
                            "inferred_goal": {"type": "string"},
                        },
                    },
                },
            },
        },
        "example": {
            "analysis_type": "comparison",
            "feature_concept": "park",
            "comparison_goal": "which area has more parks",
            "targets": [
                {
                    "target_id": "a",
                    "label": "Area A",
                    "dataset_ref": "osm_result_1",
                },
                {
                    "target_id": "b",
                    "label": "Area B",
                    "dataset_ref": "osm_result_2",
                },
            ],
            "metrics": [
                {
                    "metric": "count",
                    "role": "primary",
                    "inferred_goal": "abundance",
                }
            ],
        },
    },
}


def render_compact_tool_catalogue(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Build compact catalogue entries for the given registered tools."""
    entries: list[dict[str, Any]] = []
    for tool in tools:
        compact = _COMPACT_SPECS.get(tool.name)
        if compact is not None:
            entries.append(
                {
                    "name": tool.name,
                    "purpose": compact["purpose"],
                    "when_to_use": compact["when_to_use"],
                    "arguments": compact["arguments"],
                    "example": compact["example"],
                }
            )
            continue
        # Fallback for unexpected tools: normalised JSON Schema without examples.
        entries.append(
            {
                "name": tool.name,
                "purpose": tool.description,
                "arguments": normalize_tool_parameters_schema(tool.parameters_schema),
            }
        )
    return entries


def render_prompted_tool_instructions(tools: list[ToolDefinition]) -> str:
    """Short prompted-protocol instructions with a compact tool catalogue."""
    catalogue = render_compact_tool_catalogue(tools)
    body = json.dumps(catalogue, ensure_ascii=False, separators=(",", ":"))
    return (
        "Tool protocol (mandatory):\n"
        "Respond with exactly one JSON object and nothing else "
        "(Ollama JSON mode is enabled for this turn).\n"
        "No prose before or after the JSON. No Markdown.\n"
        'Tool call: {"tool_calls":[{"name":"<tool>","arguments":{...}}]}\n'
        'Final answer: {"final_answer":"<text>"}\n'
        "Never include both keys in one response. If more tools are needed, return "
        "ONLY tool_calls (no final_answer). Only after required tools finish may you "
        "return final_answer. Never invent tool names or coordinates. "
        "Never write Overpass QL. After an Overpass timeout, do not invent tags. "
        "Dependent tools need backend place_ref/dataset_ref values from observations.\n"
        "Available tools:\n"
        f"{body}"
    )
