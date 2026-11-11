"""Deterministic slot extraction for solar decision routing.

The planner may also send extracted slots. This module is the safety net that
reads facts already present in the user request. It never invents GHI or other
scientific indicators.
"""

from __future__ import annotations

import re
from typing import Any

from app.solar.contracts import (
    SolarCandidateGeometry,
    SolarConstraintProfile,
    SolarDecisionDraft,
    SolarGenerationStrategy,
    SolarPlaceRef,
    SolarProjectType,
    SolarRankingProfile,
    constraint_profile_from_project_type,
)
from app.solar.intent import detect_decision_mode, detect_site_selection_scope

_REGION_IN = re.compile(
    r"\b(?:in|around|near|for|within)\s+(?!this\b|the\b|that\b)([A-Z][\w.'-]{1,40}"
    r"(?:\s+[A-Z][\w.'-]{1,40}){0,3})\b",
)
_CAPACITY = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(k(?:ilo)?w(?:att)?s?|m(?:ega)?w(?:att)?s?)\b",
    re.IGNORECASE,
)
_BUDGET = re.compile(
    r"\b(?:budget|cost cap|spend)\b[^0-9]{0,24}(\d[\d,]*(?:\.\d+)?)\s*"
    r"(million|m\b|k\b|usd|eur|€|\$)?",
    re.IGNORECASE,
)
_GRID_DISTANCE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(km|kilometers?|kilometres?|m|meters?|metres?)\s+"
    r"(?:from|to|of)\s+(?:the\s+)?grid\b",
    re.IGNORECASE,
)
_MAX_GRID = re.compile(
    r"\b(?:maximum|max(?:imum)?(?:\s+acceptable)?)\s+grid\s+distance\b[^0-9]{0,16}"
    r"(\d+(?:\.\d+)?)\s*(km|kilometers?|kilometres?|m|meters?|metres?)\b",
    re.IGNORECASE,
)
_CANDIDATE_LIST = re.compile(
    r"\b(?:candidates?|buildings?|roofs?|rooftops?|sites?)\s*:\s*([^\n.]+)",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"[\"']([^\"']{1,80})[\"']")
_LAT_LON = re.compile(
    r"\b(-?\d{1,2}(?:\.\d+)?)[,\s]+(-?\d{1,3}(?:\.\d+)?)\b",
)
_LAT_THEN_LON = re.compile(
    r"\blat(?:itude)?\s*[:=]?\s*(-?\d{1,2}(?:\.\d+)?)\b.{0,32}?"
    r"\blon(?:gitude)?\s*[:=]?\s*(-?\d{1,3}(?:\.\d+)?)\b",
    re.IGNORECASE | re.DOTALL,
)
_LON_THEN_LAT = re.compile(
    r"\blon(?:gitude)?\s*[:=]?\s*(-?\d{1,3}(?:\.\d+)?)\b.{0,32}?"
    r"\blat(?:itude)?\s*[:=]?\s*(-?\d{1,2}(?:\.\d+)?)\b",
    re.IGNORECASE | re.DOTALL,
)
_HOUSE_SITE = re.compile(
    r"\b(this house|my house|this home|my home|this roof|this building|this property)\b",
    re.IGNORECASE,
)
_BBOX = re.compile(
    r"\bbbox(?:\s*[:=]\s*|\s+)"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_CELL_SIZE = re.compile(
    r"\b(?:cell(?:[_\s-]?size)?|grid(?:\s+resolution)?)\s*[:=]?\s*"
    r"(\d+(?:\.\d+)?)\s*(m|deg|degrees?)\b",
    re.IGNORECASE,
)
_CELL_SIZE_ALT = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(m|deg|degrees?)\s+(?:cells?|grid)\b",
    re.IGNORECASE,
)
_TOP_N = re.compile(
    r"\btop\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
    re.IGNORECASE,
)
_TILT = re.compile(
    r"\btilt\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:deg(?:rees?)?)?\b",
    re.IGNORECASE,
)
_AZIMUTH = re.compile(
    r"\bazimuth\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:deg(?:rees?)?)?\b",
    re.IGNORECASE,
)
_SYSTEM_LOSS = re.compile(
    r"\bsystem\s+loss(?:es)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)
_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def extract_solar_slots(message: str) -> SolarDecisionDraft:
    """Pull every solar fact already stated. Empty slots stay empty."""
    text = message.strip()
    mode = detect_decision_mode(text)
    scope = detect_site_selection_scope(text) if mode == "site_selection" else None
    project_type = _project_type(text)
    region = _region(text) if mode == "site_selection" and scope == "region_search" else None
    aoi = _aoi(text) if mode == "site_selection" and scope == "aoi_search" else None
    location = _location(text) if mode == "fixed_site" else None
    candidates = _candidates(text) if scope == "candidate_ranking" else ()
    cell_size, cell_unit = _cell_size(text)
    return SolarDecisionDraft(
        mode=mode,
        site_selection_scope=scope,
        project_type=project_type,
        location=location,
        region=region,
        aoi=aoi,
        candidate_ids=candidates,
        target_capacity_kw=_capacity_kw(text),
        budget_amount=_budget_amount(text),
        grid_connection_required=_grid_flag(text),
        maximum_grid_distance_m=_grid_distance_m(text),
        battery_required=_battery_flag(text),
        bbox=_bbox(text),
        generation_strategy=_generation_strategy(text),
        cell_size=cell_size,
        cell_size_unit=cell_unit,
        ranking_profile=_ranking_profile(text),
        constraint_profile=_constraint_profile(text),
        mounting=_mounting(text, project_type),
        system_loss_percent=_system_loss(text),
        tilt_deg=_tilt(text),
        azimuth_deg=_azimuth(text),
        top_n=_top_n(text),
    )


def overlay_draft(base: SolarDecisionDraft, incoming: SolarDecisionDraft) -> SolarDecisionDraft:
    """Copy non-empty incoming slots onto ``base``. Incoming wins when set."""
    updates: dict[str, Any] = {}
    for field in SolarDecisionDraft.model_fields:
        new_value = getattr(incoming, field)
        if field in {"candidate_ids", "candidate_geometries"}:
            if new_value:
                updates[field] = new_value
            continue
        if new_value is not None:
            updates[field] = new_value
    merged = base.model_copy(update=updates)
    return _normalize_mode_fields(merged)


def merge_solar_drafts(
    *,
    persisted: SolarDecisionDraft | None,
    extracted: SolarDecisionDraft,
    model_slots: SolarDecisionDraft | None,
) -> SolarDecisionDraft:
    """Persisted context, then trusted model slots, then facts from the current message."""
    merged = persisted or SolarDecisionDraft()
    trusted = discard_untrusted_planner_routing(model_slots, extracted)
    if trusted is not None:
        merged = overlay_draft(merged, trusted)
    merged = overlay_draft(merged, extracted)
    return infer_engine_slots(merged)


def discard_untrusted_planner_routing(
    model_slots: SolarDecisionDraft | None,
    extracted: SolarDecisionDraft,
) -> SolarDecisionDraft | None:
    """Ignore planner-invented goal/scope when the user text did not state them."""
    if model_slots is None or extracted.mode is not None:
        return model_slots
    return model_slots.model_copy(
        update={
            "mode": None,
            "site_selection_scope": None,
            "capability_id": None,
        }
    )


def infer_engine_slots(draft: SolarDecisionDraft) -> SolarDecisionDraft:
    """Fill engine knobs that SolarDSS can infer. Never invents GHI or coordinates."""
    updates: dict[str, Any] = {}
    if draft.mode == "site_selection" and draft.site_selection_scope is None:
        updates["site_selection_scope"] = "region_search"
    if draft.constraint_profile is None:
        inferred = constraint_profile_from_project_type(draft.project_type)
        if inferred is not None:
            updates["constraint_profile"] = inferred
    if not updates:
        return _normalize_mode_fields(draft)
    return _normalize_mode_fields(draft.model_copy(update=updates))


def draft_from_tool_args(payload: dict[str, Any]) -> SolarDecisionDraft:
    """Build a draft from validated tool arguments. Unknown keys must already be gone."""
    location = _place_from_args(
        payload.get("location_name"),
        payload.get("location_longitude"),
        payload.get("location_latitude"),
    )
    region = _place_from_name(payload.get("region_name"))
    aoi = _place_from_name(payload.get("aoi_name"))
    raw_candidates = payload.get("candidate_ids") or ()
    candidates = tuple(str(item) for item in raw_candidates if str(item).strip())
    geometries = _candidate_geometries_from_args(payload.get("candidate_geometries"))
    if not candidates and geometries:
        candidates = tuple(item.candidate_id for item in geometries)
    return SolarDecisionDraft(
        mode=payload.get("mode"),
        site_selection_scope=payload.get("site_selection_scope"),
        project_type=payload.get("project_type"),
        location=location,
        region=region,
        aoi=aoi,
        candidate_ids=candidates,
        candidate_geometries=geometries,
        target_capacity_kw=payload.get("target_capacity_kw"),
        budget_amount=payload.get("budget_amount"),
        budget_currency=payload.get("budget_currency"),
        objective=payload.get("objective"),
        grid_connection_required=payload.get("grid_connection_required"),
        maximum_grid_distance_m=payload.get("maximum_grid_distance_m"),
        battery_required=payload.get("battery_required"),
        capability_id=payload.get("capability_id"),
        bbox=_bbox_from_args(payload.get("bbox")),
        generation_strategy=payload.get("generation_strategy"),
        cell_size=payload.get("cell_size"),
        cell_size_unit=payload.get("cell_size_unit"),
        ranking_profile=payload.get("ranking_profile"),
        constraint_profile=payload.get("constraint_profile"),
        mounting=payload.get("mounting"),
        tracking=payload.get("tracking"),
        yield_engine=payload.get("yield_engine"),
        system_loss_percent=payload.get("system_loss_percent"),
        tilt_deg=payload.get("tilt_deg"),
        azimuth_deg=payload.get("azimuth_deg"),
        top_n=payload.get("top_n"),
    )


def mentioned_optional_fields(message: str) -> frozenset[str]:
    """Optional SolarDSS fields the user brought up but may not have valued."""
    lowered = message.lower()
    mentioned: set[str] = set()
    if re.search(r"\b(capacity|kwp|kilowatt|energy demand)\b", lowered):
        mentioned.add("target_capacity_kw")
    if re.search(r"\b(budget|cost cap|spend)\b", lowered):
        mentioned.add("budget")
    if re.search(r"\b(objective|minimi[sz]e|maximi[sz]e)\b", lowered):
        mentioned.add("objective")
    if re.search(r"\bbatter(?:y|ies)|storage\b", lowered):
        mentioned.add("battery_required")
    if re.search(r"\bgrid (?:connection|distance|interconnect|requirements?)", lowered):
        mentioned.add("grid_connection_required")
        mentioned.add("maximum_grid_distance")
    if re.search(r"\btilt\b", lowered):
        mentioned.add("tilt_deg")
    if re.search(r"\bazimuth\b", lowered):
        mentioned.add("azimuth_deg")
    if re.search(r"\bsystem loss", lowered):
        mentioned.add("system_loss_percent")
    if "bbox" in lowered:
        mentioned.add("bbox")
    if re.search(r"\b(regular|hexagonal)\s+grid\b", lowered):
        mentioned.add("generation_strategy")
    if re.search(r"\bcell(?:[_\s-]?size)?\b", lowered):
        mentioned.add("cell_size")
    return frozenset(mentioned)


def _normalize_mode_fields(draft: SolarDecisionDraft) -> SolarDecisionDraft:
    updates: dict[str, Any] = {}
    if draft.mode == "fixed_site":
        updates["site_selection_scope"] = None
        updates["region"] = None
        updates["aoi"] = None
        updates["candidate_ids"] = ()
        updates["candidate_geometries"] = ()
        updates["bbox"] = None
        updates["generation_strategy"] = None
        updates["cell_size"] = None
        updates["cell_size_unit"] = None
        updates["ranking_profile"] = None
    elif draft.mode == "site_selection":
        updates["location"] = None
        scope = draft.site_selection_scope
        if scope == "region_search":
            updates["aoi"] = None
            updates["candidate_ids"] = ()
            updates["candidate_geometries"] = ()
        elif scope == "aoi_search":
            updates["region"] = None
            updates["candidate_ids"] = ()
            updates["candidate_geometries"] = ()
        elif scope == "candidate_ranking":
            updates["region"] = None
            updates["aoi"] = None
            updates["bbox"] = None
            updates["generation_strategy"] = None
            updates["cell_size"] = None
            updates["cell_size_unit"] = None
    if not updates:
        return draft
    return draft.model_copy(update=updates)


def _project_type(text: str) -> SolarProjectType | None:
    lowered = text.lower()
    if re.search(r"\b(house|home|residential)\b", lowered) and re.search(
        r"\b(roof|rooftop|panel)", lowered
    ):
        return "residential_rooftop"
    if re.search(r"\bcommercial (?:roof|rooftop)\b", lowered):
        return "commercial_rooftop"
    if re.search(r"\b(rooftop|roof(?:top)? pv)\b", lowered):
        return "rooftop"
    if re.search(r"\b(solar farm|utility[- ]scale|solar park)\b", lowered):
        return "utility_scale"
    if re.search(r"\bground[- ]mounted\b", lowered):
        return "ground_mounted"
    return None


def _region(text: str) -> SolarPlaceRef | None:
    match = _REGION_IN.search(text)
    if match is None:
        return None
    name = match.group(1).strip().rstrip(".,;:")
    if name.lower() in {"polygon", "aoi", "area"}:
        return None
    return SolarPlaceRef(name=name)


def _aoi(text: str) -> SolarPlaceRef | None:
    if re.search(r"\b(this polygon|this aoi|area of interest)\b", text, re.IGNORECASE):
        return SolarPlaceRef(name="supplied AOI")
    return None


def _location(text: str) -> SolarPlaceRef | None:
    house = _HOUSE_SITE.search(text)
    coords = _explicit_point(text)
    if coords is not None:
        lon, lat = coords
        name = house.group(1) if house is not None else None
        return SolarPlaceRef(name=name, longitude=lon, latitude=lat)
    if house is not None:
        return SolarPlaceRef(name=house.group(1))
    return None


def _explicit_point(text: str) -> tuple[float, float] | None:
    named = _LAT_THEN_LON.search(text)
    if named is not None:
        return (float(named.group(2)), float(named.group(1)))
    named_lon = _LON_THEN_LAT.search(text)
    if named_lon is not None:
        return (float(named_lon.group(1)), float(named_lon.group(2)))
    if not re.search(r"\b(lat|lon|coord)", text, re.IGNORECASE):
        return None
    match = _LAT_LON.search(text)
    if match is None:
        return None
    first = float(match.group(1))
    second = float(match.group(2))
    # lat, lon when the first value is a plausible latitude.
    if abs(first) <= 90.0:
        return (second, first)
    return (first, second)


def _candidates(text: str) -> tuple[str, ...]:
    listed = _CANDIDATE_LIST.search(text)
    if listed is not None:
        parts = [item.strip() for item in re.split(r"[,;/]| and ", listed.group(1))]
        return tuple(item for item in parts if item and item.lower() not in {"these", "the"})
    quoted = tuple(item.strip() for item in _QUOTED.findall(text) if item.strip())
    return quoted


def _capacity_kw(text: str) -> float | None:
    match = _CAPACITY.search(text)
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("m"):
        return value * 1000.0
    return value


def _budget_amount(text: str) -> float | None:
    match = _BUDGET.search(text)
    if match is None:
        return None
    raw = float(match.group(1).replace(",", ""))
    suffix = (match.group(2) or "").lower()
    if suffix in {"million", "m"}:
        return raw * 1_000_000.0
    if suffix == "k":
        return raw * 1_000.0
    return raw


def _battery_flag(text: str) -> bool | None:
    lowered = text.lower()
    if re.search(r"\b(no battery|without (?:a )?battery|battery not required)\b", lowered):
        return False
    if re.search(r"\b(battery storage|with (?:a )?battery|battery required)\b", lowered):
        return True
    return None


def _grid_flag(text: str) -> bool | None:
    lowered = text.lower()
    if re.search(
        r"\b(off[- ]grid|no grid connection|no grid (?:connection |distance )?requirements?)\b",
        lowered,
    ):
        return False
    if re.search(
        r"\b("
        r"grid connection required|"
        r"grid (?:connection |distance )?requirements?|"
        r"must (?:be )?connect(?:ed)? to (?:the )?grid"
        r")\b",
        lowered,
    ):
        return True
    return None


def _grid_distance_m(text: str) -> float | None:
    match = _GRID_DISTANCE.search(text) or _MAX_GRID.search(text)
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("k"):
        return value * 1000.0
    return value


def _place_from_name(name: object) -> SolarPlaceRef | None:
    if not isinstance(name, str) or not name.strip():
        return None
    return SolarPlaceRef(name=name.strip())


def _place_from_args(
    name: object,
    longitude: object,
    latitude: object,
) -> SolarPlaceRef | None:
    label = name.strip() if isinstance(name, str) and name.strip() else None
    lon: float | None = None
    lat: float | None = None
    if isinstance(longitude, int | float) and not isinstance(longitude, bool):
        lon = float(longitude)
    if isinstance(latitude, int | float) and not isinstance(latitude, bool):
        lat = float(latitude)
    if (lon is None) != (lat is None):
        lon = None
        lat = None
    if label is None and lon is None:
        return None
    return SolarPlaceRef(
        name=label,
        longitude=float(lon) if lon is not None else None,
        latitude=float(lat) if lat is not None else None,
    )


def candidates_have_geometry(draft: SolarDecisionDraft) -> bool:
    """True when every listed candidate already has a caller point or polygon."""
    geometries = {item.candidate_id: item for item in draft.candidate_geometries}
    if not geometries:
        return False
    identifiers = draft.candidate_ids or tuple(geometries)
    return all(ident in geometries for ident in identifiers)


def _candidate_geometries_from_args(raw: object) -> tuple[SolarCandidateGeometry, ...]:
    if not isinstance(raw, list | tuple):
        return ()
    parsed: list[SolarCandidateGeometry] = []
    seen: set[str] = set()
    for item in raw:
        record: SolarCandidateGeometry
        if isinstance(item, SolarCandidateGeometry):
            record = item
        elif isinstance(item, dict):
            record = SolarCandidateGeometry.model_validate(item)
        else:
            continue
        if record.candidate_id in seen:
            continue
        seen.add(record.candidate_id)
        parsed.append(record)
    return tuple(parsed)


def ranking_requested(message: str) -> bool:
    """True when the user asked to rank, not merely to search."""
    return re.search(r"\brank(?:ing|ed)?\b|prioriti[sz]e", message, re.IGNORECASE) is not None


def yield_requested(message: str) -> bool:
    """True when the user asked for generation/yield numbers."""
    return (
        re.search(
            r"\b(annual generation|monthly (?:and annual )?energy|energy generation|"
            r"estimate (?:annual )?(?:generation|yield)|annual energy)\b",
            message,
            re.IGNORECASE,
        )
        is not None
    )


def _bbox(text: str) -> tuple[float, float, float, float] | None:
    match = _BBOX.search(text)
    if match is None:
        return None
    west, south, east, north = (float(match.group(index)) for index in range(1, 5))
    if east < west or north < south:
        return None
    return (west, south, east, north)


def _bbox_from_args(raw: object) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        return None
    numbers: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float):
            return None
        numbers.append(float(item))
    west, south, east, north = numbers
    if east < west or north < south:
        return None
    return (west, south, east, north)


def _generation_strategy(text: str) -> SolarGenerationStrategy | None:
    lowered = text.lower()
    if re.search(r"\bhexagonal(?:\s+grid)?\b", lowered):
        return "hexagonal_grid"
    if re.search(r"\bregular(?:\s+grid)?\b|\bgrid cells?\b", lowered):
        return "regular_grid"
    return None


def _cell_size(text: str) -> tuple[float | None, str | None]:
    match = _CELL_SIZE.search(text) or _CELL_SIZE_ALT.search(text)
    if match is None:
        return None, None
    unit = match.group(2).lower()
    return float(match.group(1)), "deg" if unit.startswith("deg") else "m"


def _ranking_profile(text: str) -> SolarRankingProfile | None:
    lowered = text.lower()
    if re.search(
        r"prioriti[sz]e (?:annual )?(?:energy|production|generation)|"
        r"energy[- ]focused|"
        r"energy yield|"
        r"maximi[sz]e annual (?:energy|production|generation)",
        lowered,
    ):
        return "energy_focused"
    if re.search(
        r"infrastructure[- ]focused|"
        r"infrastructure access|"
        r"prioriti[sz]e (?:grid|infrastructure|roads?)",
        lowered,
    ):
        return "infrastructure_focused"
    if re.search(
        r"environment[- ]focused|"
        r"environmental (?:limits?|constraints?|criteria)|"
        r"land constraints?|"
        r"prioriti[sz]e (?:environment|protected)",
        lowered,
    ):
        return "environment_focused"
    if re.search(r"\bbalanced(?:\s+(?:profile|ranking|weights|mix))?\b", lowered):
        return "balanced"
    return None


def _constraint_profile(text: str) -> SolarConstraintProfile | None:
    lowered = text.lower()
    if "utility_scale_default_example" in lowered:
        return "utility_scale_default_example"
    if "rooftop_default_example" in lowered:
        return "rooftop_default_example"
    return None


def _mounting(text: str, project_type: SolarProjectType | None) -> str | None:
    lowered = text.lower()
    if re.search(r"\bground[- ]mount", lowered):
        return "ground"
    if re.search(r"\broof(?:top)?[- ]mount", lowered):
        return "rooftop"
    if project_type in {"utility_scale", "ground_mounted"}:
        return "ground"
    if project_type in {"rooftop", "commercial_rooftop", "residential_rooftop"}:
        return "rooftop"
    return None


def _tilt(text: str) -> float | None:
    match = _TILT.search(text)
    if match is None:
        return None
    return float(match.group(1))


def _azimuth(text: str) -> float | None:
    match = _AZIMUTH.search(text)
    if match is None:
        return None
    return float(match.group(1))


def _system_loss(text: str) -> float | None:
    match = _SYSTEM_LOSS.search(text)
    if match is None:
        return None
    return float(match.group(1))


def _top_n(text: str) -> int | None:
    match = _TOP_N.search(text)
    if match is None:
        return None
    token = match.group(1).lower()
    value = int(token) if token.isdigit() else _WORD_NUMBERS.get(token)
    if value is None or value < 1:
        return None
    return value
