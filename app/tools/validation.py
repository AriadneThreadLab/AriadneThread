"""Compact structured feedback for tool-argument validation failures."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError


def summarise_validation_error(error: ValidationError, *, max_errors: int = 6) -> str:
    """Human-readable one-line summary for logs / short messages."""
    parts: list[str] = []
    for detail in error.errors()[:max_errors]:
        location = ".".join(str(item) for item in detail["loc"]) or "<root>"
        parts.append(f"{location}: {detail['msg']}")
    return "; ".join(parts)


def validation_issue_list(error: ValidationError, *, max_errors: int = 8) -> list[dict[str, str]]:
    """Sanitized field/issue pairs (no input values)."""
    issues: list[dict[str, str]] = []
    for detail in error.errors()[:max_errors]:
        location = ".".join(str(item) for item in detail["loc"]) or "<root>"
        err_type = str(detail.get("type") or "invalid")
        issues.append({"field": location, "issue": err_type})
    return issues


def classify_validation_fields(
    error: ValidationError,
) -> tuple[list[str], list[str], list[str]]:
    """Return (missing_fields, invalid_fields, extra_fields)."""
    missing: list[str] = []
    invalid: list[str] = []
    extra: list[str] = []
    for detail in error.errors():
        location = ".".join(str(item) for item in detail["loc"]) or "<root>"
        err_type = str(detail.get("type") or "")
        if err_type == "missing":
            missing.append(location)
        elif err_type == "extra_forbidden":
            extra.append(location)
        else:
            invalid.append(location)
    return missing, invalid, extra


def format_tool_argument_observation(
    tool_name: str,
    error: ValidationError,
    *,
    argument_keys: list[str] | None = None,
) -> str:
    """Compact JSON observation the model can use to repair arguments."""
    missing, invalid, extra = classify_validation_fields(error)
    payload: dict[str, Any] = {
        "tool_error": {
            "code": "tool_argument_error",
            "tool": tool_name,
            "message": "Arguments did not match the required schema.",
            "validation": validation_issue_list(error),
            "missing_fields": missing,
            "invalid_fields": invalid,
            "extra_fields": extra,
        }
    }
    if argument_keys is not None:
        payload["tool_error"]["received_argument_keys"] = argument_keys
    if tool_name == "query_osm":
        payload["tool_error"]["hint"] = (
            "Required: tags (array of {key,value} objects) and exactly one of "
            "place | point | bbox. Use place for cities. Limit field name is "
            "'limit' (not max_items). Example: "
            '{"place":"Tehran, Iran","tags":[{"key":"leisure","value":"park"}],'
            '"limit":20}'
        )
    elif tool_name == "search_osm_knowledge":
        payload["tool_error"]["hint"] = (
            "Required: query (string). Optional: top_k. Example: "
            '{"query":"public park OSM tag","top_k":5}'
        )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
