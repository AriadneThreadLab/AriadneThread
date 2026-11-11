"""Detect solar PV decision questions so OSM comparison routing stays out."""

from __future__ import annotations

from app.solar.capabilities import select_solar_capability
from app.solar.intent import (
    detect_decision_mode,
    detect_site_selection_scope,
    is_solar_decision_request,
    is_solar_knowledge_request,
    is_solar_request,
)

__all__ = [
    "detect_decision_mode",
    "detect_site_selection_scope",
    "is_solar_decision_request",
    "is_solar_knowledge_request",
    "is_solar_request",
    "select_solar_capability",
]
