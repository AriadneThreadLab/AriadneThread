"""Host view of the SolarDSS capability registry.

The LLM may only select a registered id. This module does not compute
indicators, yield, or ranking scores.
"""

from __future__ import annotations

import logging
import re
from importlib import import_module
from importlib.util import find_spec
from typing import Any

from app.core.errors import SolarUnknownCapabilityError
from app.solar.contracts import SolarDecisionDraft

logger = logging.getLogger(__name__)

_CAPABILITY_PATTERN = r"^[a-z][a-z0-9_]{2,47}$"
_SAFE_LOG_TOKEN = re.compile(r"^[A-Za-z0-9._-]{1,80}$")

#: Closed catalog ids. Live plugin ids win when the plugin is installed.
FALLBACK_CAPABILITY_IDS: tuple[str, ...] = (
    "solar_site_indicators",
    "solar_site_selection",
    "solar_candidate_ranking",
    "solar_fixed_site_analysis",
    "solar_pv_design",
    "solar_yield_estimation",
    "solar_compare_candidates",
)

_PLANNER_HINTS: dict[str, str] = {
    "solar_site_selection": (
        "Use for SITE_SELECTION region_search or aoi_search: find locations in a "
        "region or inside an AOI. Do not invent GHI or slope."
    ),
    "solar_candidate_ranking": (
        "Use for SITE_SELECTION candidate_ranking: rank supplied buildings or sites."
    ),
    "solar_fixed_site_analysis": (
        "Use for FIXED_SITE: the user already owns or specified one site."
    ),
    "solar_site_indicators": "Use only to evaluate explicit indicators at one known site.",
    "solar_pv_design": "Use for PV layout / module-count enumeration after the site is known.",
    "solar_yield_estimation": "Use for annual energy yield at a known site.",
    "solar_compare_candidates": "Use to compare already-known candidate sites.",
}

_YIELD_INTENT = re.compile(
    r"\b(annual yield|kwh(?:/kwp)?|energy yield|pvcalc)\b",
    re.IGNORECASE,
)
_DESIGN_INTENT = re.compile(
    r"\b(pv (?:layout|design)|module count|inverter sizing)\b",
    re.IGNORECASE,
)
_COMPARE_INTENT = re.compile(
    r"\b(compare (?:these |the )?(?:sites|candidates|buildings)|pareto)\b",
    re.IGNORECASE,
)
_INDICATOR_INTENT = re.compile(
    r"\b(ghi|site indicators?|solar resource at)\b",
    re.IGNORECASE,
)


def registered_capability_ids() -> tuple[str, ...]:
    """Bound catalog ids from ariadne_solar, or the closed fallback list."""
    registry = _plugin_registry()
    if registry is None:
        return FALLBACK_CAPABILITY_IDS
    ids = tuple(
        spec.capability_id for spec in registry.filter(executable_only=True) if spec.is_executable
    )
    return ids or FALLBACK_CAPABILITY_IDS


def allowed_host_capability_ids() -> tuple[str, ...]:
    return tuple(sorted(registered_capability_ids()))


def is_allowed_host_capability(capability_id: str) -> bool:
    return capability_id in allowed_host_capability_ids()


def resolve_capability_id(capability_id: str) -> str:
    requested = capability_id.strip()
    allowed = set(registered_capability_ids())
    if requested not in allowed:
        raise SolarUnknownCapabilityError(
            f"unsupported capability_id={requested!r}",
            received=requested,
            allowed=registered_capability_ids(),
        )
    return requested


def select_solar_capability(message: str, draft: SolarDecisionDraft | None = None) -> str | None:
    """Deterministic NL + slots → registered id. Never invents an unregistered id."""
    allowed = set(registered_capability_ids())

    def _pick(candidate: str) -> str | None:
        return candidate if candidate in allowed else None

    if _YIELD_INTENT.search(message):
        return _pick("solar_yield_estimation")
    if _DESIGN_INTENT.search(message):
        return _pick("solar_pv_design")
    if _COMPARE_INTENT.search(message):
        return _pick("solar_compare_candidates")
    if _INDICATOR_INTENT.search(message):
        return _pick("solar_site_indicators")
    mode = None if draft is None else draft.mode
    scope = None if draft is None else draft.site_selection_scope
    if mode == "fixed_site":
        return _pick("solar_fixed_site_analysis")
    if scope == "candidate_ranking":
        return _pick("solar_candidate_ranking")
    if mode == "site_selection":
        return _pick("solar_site_selection")
    return None


def reconcile_capability(
    requested: str | None,
    *,
    message: str,
    draft: SolarDecisionDraft,
) -> str | None:
    """Honour a registered LLM id when it matches the routed mode; else deterministic."""
    deterministic = select_solar_capability(message, draft)
    if requested is None or not requested.strip():
        return deterministic
    try:
        canonical = resolve_capability_id(requested)
    except SolarUnknownCapabilityError:
        return deterministic
    if _compatible(canonical, draft):
        return canonical
    return deterministic


def capability_id_looks_valid(value: str) -> bool:
    return re.fullmatch(_CAPABILITY_PATTERN, value) is not None


def capability_id_field_description() -> str:
    hints = [
        "Registered SolarDSS capability_id. Choose only an enum value. "
        "Do not invent method names or compute GHI, slope, or other indicators."
    ]
    for capability_id in registered_capability_ids():
        hint = _PLANNER_HINTS.get(capability_id)
        if hint is not None:
            hints.append(f"{capability_id}: {hint}")
    return " ".join(hints)


def analyze_solar_site_tool_description() -> str:
    ids = ", ".join(registered_capability_ids()) or "none"
    return (
        "Solar Decision Support routing: detect goal (existing building, site "
        "selection, candidate ranking), collect only that goal's parameters, "
        "select a registered capability_id, and return follow-up questions. "
        f"capability_id must be a registered SolarDSS id ({ids}). Do not invent "
        "a goal or calculate GHI/slope. Never ask tilt, azimuth, loss, bbox, or "
        "cell size unless the user stated them. Example: "
        '{"mode":"site_selection","site_selection_scope":"region_search",'
        '"project_type":"utility_scale","region_name":"Vienna",'
        '"capability_id":"solar_site_selection"}.'
    )


def capability_json_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "enum": list(registered_capability_ids()),
        "description": capability_id_field_description(),
    }


def unknown_capability_payload(received: object) -> dict[str, Any]:
    text = received.strip() if isinstance(received, str) else ""
    return {
        "error_code": "unknown_solar_capability",
        "received": text,
        "allowed_capabilities": list(registered_capability_ids()),
    }


def safe_logged_solar_ids(
    arguments: dict[str, Any] | None = None,
    *,
    capability_id: object = None,
    mode: object = None,
) -> dict[str, str]:
    if arguments is not None:
        capability_id = arguments.get("capability_id", capability_id)
        mode = arguments.get("mode", mode)
    out: dict[str, str] = {}
    if isinstance(capability_id, str) and _SAFE_LOG_TOKEN.fullmatch(capability_id.strip()):
        out["capability_id"] = capability_id.strip()
    if isinstance(mode, str) and _SAFE_LOG_TOKEN.fullmatch(mode.strip()):
        out["mode"] = mode.strip()
    return out


def _compatible(capability_id: str, draft: SolarDecisionDraft) -> bool:
    if draft.mode == "fixed_site":
        return capability_id in {
            "solar_fixed_site_analysis",
            "solar_site_indicators",
            "solar_pv_design",
            "solar_yield_estimation",
        }
    if draft.site_selection_scope == "candidate_ranking":
        return capability_id in {"solar_candidate_ranking", "solar_compare_candidates"}
    if draft.mode == "site_selection":
        return capability_id in {
            "solar_site_selection",
            "solar_candidate_ranking",
            "solar_compare_candidates",
        }
    return True


def _plugin_registry() -> Any | None:
    if find_spec("ariadne_solar") is None:
        return None
    try:
        module = import_module("ariadne_solar.capabilities")
        loader = getattr(module, "load_default_registry", None)
        if not callable(loader):
            return None
        return loader()
    except Exception:
        logger.exception("failed to load SolarDSS capability registry")
        return None
