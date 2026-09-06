"""Host view of the GeoLoadST capability registry.

This module does not compute scientific results. It copies registered ids from
``ariadne_geoloadst`` and applies a small, explicit alias table so the planner
can send a documented synonym and the runtime still executes the canonical id.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any

from app.core.errors import EnergyUnknownCapabilityError

logger = logging.getLogger(__name__)

_CAPABILITY_PATTERN = r"^[a-z][a-z0-9_]{2,47}$"
_SAFE_LOG_TOKEN = re.compile(r"^[A-Za-z0-9._-]{1,80}$")

#: Documented host synonyms → catalog ids. Science is unchanged.
HOST_CAPABILITY_ALIASES: dict[str, str] = {
    "moran_lisa": "lisa_instability",
    "lisa": "lisa_instability",
    "topology_analysis": "topology_centrality",
    "network_centrality": "topology_centrality",
    "centrality_analysis": "topology_centrality",
}

#: Planner hints. Not GeoLoadST formulas.
_CAPABILITY_PLANNER_HINTS: dict[str, str] = {
    "topology_centrality": (
        "Use when the user asks about network topology, structurally important "
        "buses or nodes, degree centrality, betweenness centrality, closeness "
        "centrality, or central/important nodes. Do not invent topology_analysis, "
        "network_centrality, or centrality_analysis."
    ),
    "spatial_clustering_of_instability": (
        "Use when the user asks for spatial clusters of unstable loads."
    ),
    "lisa_instability": (
        "Use for Local Moran / LISA hotspots of load instability. Host alias: moran_lisa."
    ),
    "multidim_pca_clustering": (
        "Use when the user asks for PCA, clustering of load-instability features, "
        "or multidimensional instability structure. After an internal failure of "
        "this capability, do not substitute spatial_clustering_of_instability or "
        "load_instability_rms."
    ),
}

_TOPOLOGY_INTENT = re.compile(
    r"\b("
    r"degree(?:\s+centrality)?|"
    r"betweenness(?:\s+centrality)?|"
    r"closeness(?:\s+centrality)?|"
    r"structurally\s+(?:most\s+)?important|"
    r"important\s+buses|"
    r"central(?:ity)?\s+nodes"
    r")\b",
    re.IGNORECASE,
)
_UNSTABLE_CLUSTER_INTENT = re.compile(
    r"\b("
    r"spatial\s+clusters?\s+of\s+unstable|"
    r"clusters?\s+of\s+unstable\s+loads|"
    r"spatial\s+clustering\s+of\s+instability"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CapabilityResolution:
    """Requested id plus the catalog id actually dispatched."""

    requested: str
    canonical: str
    alias_used: str | None


def registered_capability_ids() -> tuple[str, ...]:
    """Bound catalog ids from GeoLoadST. Empty when the plugin is missing."""
    registry = _plugin_registry()
    if registry is None:
        return ()
    return tuple(
        spec.capability_id for spec in registry.filter(executable_only=True) if spec.is_executable
    )


def allowed_host_capability_ids() -> tuple[str, ...]:
    """Ids the model may send: registered catalog ids plus documented aliases."""
    bound = set(registered_capability_ids())
    allowed = set(bound)
    for alias, canonical in HOST_CAPABILITY_ALIASES.items():
        if canonical in bound:
            allowed.add(alias)
    return tuple(sorted(allowed))


def is_allowed_host_capability(capability_id: str) -> bool:
    return capability_id in allowed_host_capability_ids()


def resolve_capability_id(capability_id: str) -> str:
    """Return the catalog id, or raise a typed unknown-capability error."""
    return resolve_capability(capability_id).canonical


def resolve_capability(capability_id: str) -> CapabilityResolution:
    """Normalize a host or alias id onto a bound catalog id."""
    requested = capability_id.strip()
    bound = set(registered_capability_ids())
    alias_used: str | None = None
    canonical = HOST_CAPABILITY_ALIASES.get(requested, requested)
    if canonical != requested:
        alias_used = requested
    if canonical not in bound:
        raise EnergyUnknownCapabilityError(
            f"unsupported capability_id={requested!r}",
            received=requested,
            allowed=registered_capability_ids(),
        )
    return CapabilityResolution(
        requested=requested,
        canonical=canonical,
        alias_used=alias_used,
    )


def select_energy_capability(message: str) -> str | None:
    """Deterministic NL → registered id. Never invents an unregistered id."""
    bound = set(registered_capability_ids())
    if _UNSTABLE_CLUSTER_INTENT.search(message):
        candidate = "spatial_clustering_of_instability"
        return candidate if candidate in bound else None
    if _TOPOLOGY_INTENT.search(message):
        candidate = "topology_centrality"
        return candidate if candidate in bound else None
    return None


def capability_id_field_description() -> str:
    """Model-facing description of ``capability_id``, generated from the registry."""
    hints: list[str] = [
        "Registered GeoLoadST capability_id. Choose only an enum value. Do not invent method names."
    ]
    for capability_id in registered_capability_ids():
        hint = _CAPABILITY_PLANNER_HINTS.get(capability_id)
        if hint is not None:
            hints.append(f"{capability_id}: {hint}")
    aliases = ", ".join(
        f"{alias}→{canonical}"
        for alias, canonical in sorted(HOST_CAPABILITY_ALIASES.items())
        if canonical in set(registered_capability_ids())
    )
    if aliases:
        hints.append(f"Documented aliases (normalized before execution): {aliases}.")
    return " ".join(hints)


def analyze_energy_grid_tool_description() -> str:
    ids = ", ".join(registered_capability_ids()) or "none"
    return (
        "Run GeoLoadST analysis on a SimBench network. Required string arguments: "
        "network_id and capability_id. capability_id must be a registered GeoLoadST "
        f"id ({ids}). For network topology or degree / betweenness / closeness "
        "centrality, or structurally important buses, use topology_centrality. "
        "Do not invent topology_analysis, network_centrality, or centrality_analysis. "
        'Example: {"network_id":"1-MV-urban--0-sw","capability_id":"topology_centrality"}. '
        "Not OSM. Do not pass raw Python objects."
    )


def capability_json_schema() -> dict[str, Any]:
    """JSON Schema fragment for the planner. Enum is the live registry."""
    return {
        "type": "string",
        "enum": list(registered_capability_ids()),
        "description": capability_id_field_description(),
    }


def unknown_capability_payload(received: object) -> dict[str, Any]:
    """Structured planner observation. No secrets."""
    text = received.strip() if isinstance(received, str) else ""
    return {
        "error_code": "unknown_energy_capability",
        "received": text,
        "allowed_capabilities": list(registered_capability_ids()),
    }


def safe_logged_energy_ids(
    arguments: dict[str, Any] | None = None,
    *,
    network_id: object = None,
    capability_id: object = None,
) -> dict[str, str]:
    """Copy only identifier-shaped values. Never logs credentials."""
    if arguments is not None:
        network_id = arguments.get("network_id", network_id)
        capability_id = arguments.get("capability_id", capability_id)
    out: dict[str, str] = {}
    if isinstance(network_id, str) and _SAFE_LOG_TOKEN.fullmatch(network_id.strip()):
        out["network_id"] = network_id.strip()
    if isinstance(capability_id, str) and _SAFE_LOG_TOKEN.fullmatch(capability_id.strip()):
        out["capability_id"] = capability_id.strip()
    return out


def capability_id_looks_valid(value: str) -> bool:
    return re.fullmatch(_CAPABILITY_PATTERN, value) is not None


def _plugin_registry() -> Any | None:
    if find_spec("ariadne_geoloadst") is None:
        return None
    try:
        module = import_module("ariadne_geoloadst.capabilities")
        loader = getattr(module, "load_default_registry", None)
        if not callable(loader):
            return None
        return loader()
    except Exception:
        logger.exception("failed to load GeoLoadST capability registry")
        return None
