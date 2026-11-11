"""Detect solar PV decision requests. Does not compute indicators."""

from __future__ import annotations

import re

from app.solar.contracts import SolarDecisionMode, SolarSiteSelectionScope

_SOLAR = re.compile(
    r"\b("
    r"solar(?:\s+(?:farm|park|site|sites|panel|panels|pv|photovoltaic))?|"
    r"photovoltaic|"
    r"rooftop\s+pv|"
    r"ground[- ]mounted|"
    r"pv\s+(?:site|sites|farm|design|yield|system)|"
    r"\bpv\b"
    r")\b",
    re.IGNORECASE,
)
_DOCUMENTED_AOI_EXAMPLE = re.compile(
    r"best location inside (?:this )?(?:polygon|aoi|area of interest)",
    re.IGNORECASE,
)
_FIXED_SITE = re.compile(
    r"\b("
    r"i own this (?:house|home|roof|building|property)|"
    r"my (?:house|home|roof|building|property)|"
    r"this (?:house|home|roof|building|property)|"
    r"fixed site|"
    r"existing building(?: evaluation)?|"
    r"building evaluation|"
    r"evaluate (?:this |my |an? )?(?:roof|house|building)"
    r")\b",
    re.IGNORECASE,
)
_SITE_SELECTION_GOAL = re.compile(
    r"\b(site selection|find (?:a |the )?(?:site|location)s?|search for (?:a )?site)\b",
    re.IGNORECASE,
)
_CANDIDATE_RANKING = re.compile(
    r"\b("
    r"rank (?:these|the)|"
    r"these buildings|"
    r"these (?:roofs|rooftops|sites|candidates)|"
    r"candidate ranking|"
    r"rank (?:candidates|buildings|roofs|rooftops)"
    r")\b",
    re.IGNORECASE,
)
_NUMBERED_GOAL = re.compile(
    r"^\s*(?:(?:option|choice|goal)\s*)?([123])(?:[\.)\s].*)?$",
    re.IGNORECASE,
)
_AOI_SEARCH = re.compile(
    r"\b("
    r"inside this polygon|"
    r"within this polygon|"
    r"this aoi|"
    r"area of interest|"
    r"inside the (?:drawn )?area"
    r")\b",
    re.IGNORECASE,
)
_REGION_SEARCH = re.compile(
    r"\b("
    r"locations?(?:\s+in)?|"
    r"sites?(?:\s+in)?|"
    r"where (?:to|can I) (?:build|put|place)|"
    r"good solar|"
    r"find (?:good |the best )?solar|"
    r"region search|"
    r"in [A-Z]"
    r")",
    re.IGNORECASE,
)
_SOLAR_KNOWLEDGE = re.compile(
    r"("
    r"\bghi\b|"
    r"global horizontal irradiance|"
    r"\bdni\b|"
    r"\bdhi\b|"
    r"why is slope important|"
    r"why(?:'s| is) slope|"
    r"why (?:was|is) (?:this |the )?(?:site|candidate) excluded|"
    r"site excluded|"
    r"what regulation affects|"
    r"what (?:regulation|permitting|grid (?:code|rule)s?)|"
    r"permitting rules?|"
    r"grid (?:code|rules?|connection rules?)|"
    r"installation guidelines?|"
    r"panel documentation|"
    r"technical standards?|"
    r"pv methodology|"
    r"photovoltaic methodology"
    r")",
    re.IGNORECASE,
)


def is_solar_request(message: str, *, has_context: bool = False) -> bool:
    """True for a solar PV decision or a solar-PV knowledge question."""
    if is_solar_knowledge_request(message):
        return True
    return is_solar_decision_request(message, has_context=has_context)


def is_solar_decision_request(message: str, *, has_context: bool = False) -> bool:
    """True when the question is a solar PV siting or fixed-site decision."""
    if has_context:
        return True
    text = message.strip()
    if not text:
        return False
    if _SOLAR.search(text) is not None:
        return True
    return _DOCUMENTED_AOI_EXAMPLE.search(text) is not None


def is_solar_knowledge_request(message: str) -> bool:
    """True for methodology, definition, exclusion-meaning, or regulation questions."""
    text = message.strip()
    if not text:
        return False
    return _SOLAR_KNOWLEDGE.search(text) is not None


def detect_decision_mode(message: str) -> SolarDecisionMode | None:
    """SITE_SELECTION vs FIXED_SITE from wording already in the request."""
    text = message.strip()
    if not text:
        return None
    numbered = _numbered_goal(text)
    if numbered == "fixed_site":
        return "fixed_site"
    if numbered in {"site_selection", "candidate_ranking"}:
        return "site_selection"
    if _CANDIDATE_RANKING.search(text) is not None:
        return "site_selection"
    if _SITE_SELECTION_GOAL.search(text) is not None or _AOI_SEARCH.search(text) is not None:
        return "site_selection"
    if _FIXED_SITE.search(text) is not None:
        return "fixed_site"
    if _looks_like_search(text):
        return "site_selection"
    return None


def detect_site_selection_scope(message: str) -> SolarSiteSelectionScope | None:
    """REGION_SEARCH / AOI_SEARCH / CANDIDATE_RANKING. None when mode is not search."""
    text = message.strip()
    if detect_decision_mode(text) == "fixed_site":
        return None
    numbered = _numbered_goal(text)
    if numbered == "candidate_ranking" or _CANDIDATE_RANKING.search(text) is not None:
        return "candidate_ranking"
    if _AOI_SEARCH.search(text) is not None or _DOCUMENTED_AOI_EXAMPLE.search(text) is not None:
        return "aoi_search"
    if detect_decision_mode(text) == "site_selection":
        return "region_search"
    if _REGION_SEARCH.search(text) is not None and _SOLAR.search(text) is not None:
        return "region_search"
    return None


def _numbered_goal(text: str) -> str | None:
    """Map a short 1/2/3 reply onto a SolarDSS goal. Long first-turn text is ignored."""
    if len(text) > 48:
        return None
    match = _NUMBERED_GOAL.match(text)
    if match is None:
        return None
    choice = match.group(1)
    if choice == "1":
        return "fixed_site"
    if choice == "2":
        return "site_selection"
    return "candidate_ranking"


def _looks_like_search(text: str) -> bool:
    lowered = text.lower()
    if _CANDIDATE_RANKING.search(text) or _AOI_SEARCH.search(text):
        return True
    if _DOCUMENTED_AOI_EXAMPLE.search(text) or _SITE_SELECTION_GOAL.search(text):
        return True
    search_verbs = ("find ", "search ", "locate ", "where ", "best location", "good solar")
    return any(token in lowered for token in search_verbs)
