"""Detect energy-grid / SimBench questions so OSM comparison routing stays out."""

from __future__ import annotations

import re

from app.tools.energy_capabilities import select_energy_capability

__all__ = ["is_energy_grid_request", "select_energy_capability"]

_ENERGY_GRID = re.compile(
    r"\b("
    r"simbench|geoloadst|pandapower|"
    r"energy[\s_-]?grid|energy[\s_-]?network|"
    r"power[\s_-]?grid|power[\s_-]?network|"
    r"spatial load|load patterns?|load profiles?|"
    r"load instability|grid load"
    r")\b",
    re.IGNORECASE,
)


def is_energy_grid_request(message: str) -> bool:
    """True when the question is about SimBench / energy-grid analysis, not OSM."""
    return _ENERGY_GRID.search(message) is not None
