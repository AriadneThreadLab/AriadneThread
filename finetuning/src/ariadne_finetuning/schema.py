"""Minimal JSON Schema checking for exported plan schemas.

Ariadne exports the authoritative ``AnalysisPlan`` / ``MultiTargetComparisonPlan``
JSON Schema next to every dataset, so this package never re-declares the plan
shape. What it needs in return is a dependency-free way to ask "is this
generation structurally a plan?".

Supported keywords: ``$ref``/``$defs``, ``type``, ``properties``, ``required``,
``additionalProperties``, ``items``, ``enum``, ``anyOf``. Numeric bounds, string
patterns and lengths are intentionally *not* enforced: field-level correctness
is measured by :mod:`ariadne_finetuning.metrics` against the reference plan.
"""

from __future__ import annotations

from typing import Any

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}

_MAX_DEPTH = 12


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    node: Any = root
    for part in ref.lstrip("#/").split("/"):
        if not isinstance(node, dict) or part not in node:
            return {}
        node = node[part]
    return node if isinstance(node, dict) else {}


def schema_errors(
    value: Any,
    schema: dict[str, Any],
    *,
    root: dict[str, Any] | None = None,
    path: str = "$",
    depth: int = 0,
) -> list[str]:
    """Return a list of structural violations (empty means valid)."""
    document = root if root is not None else schema
    node = _resolve(schema, document)
    if depth > _MAX_DEPTH or not node:
        return []

    if "anyOf" in node:
        branches = [item for item in node["anyOf"] if isinstance(item, dict)]
        if branches and all(
            schema_errors(value, branch, root=document, path=path, depth=depth + 1)
            for branch in branches
        ):
            return [f"{path}: does not match any allowed variant"]
        return []

    errors: list[str] = []
    expected = node.get("type")
    if isinstance(expected, str) and expected in _TYPE_MAP:
        python_type = _TYPE_MAP[expected]
        # bool is an int subclass; a boolean is never an acceptable number.
        if isinstance(value, bool) and expected in {"integer", "number"}:
            return [f"{path}: expected {expected}"]
        if not isinstance(value, python_type):
            return [f"{path}: expected {expected}"]
    if expected == "null" and value is not None:
        return [f"{path}: expected null"]

    if "enum" in node and value not in node["enum"]:
        errors.append(f"{path}: value is not one of the allowed options")

    if isinstance(value, dict):
        properties = node.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        for key in node.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required property is missing")
        if node.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}.{key}: property is not allowed")
        for key, item in value.items():
            sub = properties.get(key)
            if isinstance(sub, dict):
                errors.extend(
                    schema_errors(item, sub, root=document, path=f"{path}.{key}", depth=depth + 1)
                )
    elif isinstance(value, list):
        items = node.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                errors.extend(
                    schema_errors(
                        item, items, root=document, path=f"{path}[{index}]", depth=depth + 1
                    )
                )
    return errors


def is_valid(value: Any, schema: dict[str, Any]) -> bool:
    """Whether ``value`` structurally satisfies ``schema``."""
    return not schema_errors(value, schema)


def allowed_top_level_keys(schema: dict[str, Any]) -> frozenset[str]:
    """Property names declared by the schema root (for hallucination counting)."""
    properties = schema.get("properties")
    return frozenset(properties) if isinstance(properties, dict) else frozenset()
