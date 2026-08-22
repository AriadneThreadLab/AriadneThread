"""Model-facing JSON Schema normalisation for Ollama tool advertising.

Backend tools continue to validate against their authoritative Pydantic models.
Schemas sent to the model (prompted catalogue or native ``tools``) may be
simplified for interoperability and context-window size without weakening
execution validation.
"""

from __future__ import annotations

import copy
from typing import Any

#: Bump when the model-facing tool catalogue changes shape. Recorded on
#: active-learning candidates for dataset lineage.
TOOL_SCHEMA_VERSION = "ariadne-tool-schema-1"

# Metadata that inflates prompted catalogues / native tool payloads without
# helping constrained generation for local models.
_DROP_KEYS = frozenset(
    {
        "title",
        "default",
        "examples",
        "example",
        "description",
        "additionalProperties",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)


def normalize_tool_parameters_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, Ollama-oriented copy of a Pydantic JSON Schema object.

    Resolves local ``$ref`` / ``$defs`` and drops verbose metadata. Does not
    mutate ``schema``.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    resolved = _resolve_local_refs(copy.deepcopy(schema))
    cleaned = _strip_keys(resolved, _DROP_KEYS)
    if not isinstance(cleaned, dict):
        return {"type": "object", "properties": {}}
    cleaned.pop("$defs", None)
    cleaned.pop("definitions", None)
    if "type" not in cleaned:
        cleaned["type"] = "object"
    if cleaned.get("type") == "object" and "properties" not in cleaned:
        cleaned["properties"] = {}
    return cleaned


def schema_structure_summary(schema: dict[str, Any]) -> dict[str, Any]:
    """Safe structural summary for diagnostics (no property values)."""
    keys = sorted(schema.keys()) if isinstance(schema, dict) else []
    markers = (
        "$defs",
        "definitions",
        "$ref",
        "anyOf",
        "oneOf",
        "allOf",
        "enum",
        "const",
        "nullable",
    )
    present = sorted(m for m in markers if _contains_key(schema, m))
    return {"top_level_keys": keys, "markers": present}


def _resolve_local_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs") or schema.get("definitions") or {}
    if not isinstance(defs, dict):
        defs = {}

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.rsplit("/", 1)[-1]
                target = defs.get(name)
                if isinstance(target, dict):
                    merged = resolve(copy.deepcopy(target))
                    extras = {k: resolve(v) for k, v in node.items() if k != "$ref"}
                    if isinstance(merged, dict):
                        merged.update(extras)
                        return merged
                    return merged
            return {k: resolve(v) for k, v in node.items() if k not in {"$defs", "definitions"}}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    resolved = resolve(schema)
    return resolved if isinstance(resolved, dict) else {"type": "object", "properties": {}}


def _strip_keys(node: Any, drop: frozenset[str]) -> Any:
    if isinstance(node, dict):
        return {k: _strip_keys(v, drop) for k, v in node.items() if k not in drop}
    if isinstance(node, list):
        return [_strip_keys(item, drop) for item in node]
    return node


def _contains_key(node: Any, key: str) -> bool:
    if isinstance(node, dict):
        if key in node:
            return True
        return any(_contains_key(v, key) for v in node.values())
    if isinstance(node, list):
        return any(_contains_key(item, key) for item in node)
    return False
