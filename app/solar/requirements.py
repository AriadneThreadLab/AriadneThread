"""Missing-requirement adapter: SolarDSS is the authority when importable.

The host never invents GHI, slope, elevation, or other provider-filled layers.
When SolarDSS is not installed, a field-level mirror of the same slot rules is
used so routing tests stay offline.
"""

from __future__ import annotations

import re
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Literal

from app.solar.contracts import SolarDecisionDraft, SolarMissingRequirement
from app.solar.slots import candidates_have_geometry, ranking_requested, yield_requested

_OPTIONAL_CAPACITY_REASON = "needed only for capacity-specific sizing"
_OPTIONAL_BUDGET_REASON = "needed only for cost-constrained selection or design"
_OPTIONAL_OBJECTIVE_REASON = "needed to select a registered objective; none is inferred"
_OPTIONAL_GRID_FLAG_REASON = "needed only when interconnection policy is in scope"
_OPTIONAL_GRID_DISTANCE_REASON = "needed when grid_connection_required is true"
_OPTIONAL_ENV_REASON = "needed to weight environmental indicators; weights are not inferred"
_OPTIONAL_BATTERY_REASON = "needed only if storage is in scope"
_OPTIONAL_MODULE_REASON = "needed only for technology-specific design"


def missing_requirements_for(
    draft: SolarDecisionDraft,
    message: str = "",
) -> tuple[tuple[SolarMissingRequirement, ...], Literal["solardss", "host_mirror"]]:
    """Return (records, source). Source is ``solardss`` or ``host_mirror``."""
    structural = _structural_missing(draft)
    execution = _execution_missing(draft, message)
    if not structural:
        records = _solardss_missing(draft)
        if records is not None:
            return records + execution, "solardss"
    return structural + _host_missing(draft) + execution, "host_mirror"


def _structural_missing(draft: SolarDecisionDraft) -> tuple[SolarMissingRequirement, ...]:
    missing: list[SolarMissingRequirement] = []
    if draft.mode is None:
        missing.append(
            SolarMissingRequirement(
                field="mode",
                required=True,
                reason="existing building, site selection, or candidate ranking is not yet known",
            )
        )
        return tuple(missing)
    if draft.project_type is None and draft.site_selection_scope != "candidate_ranking":
        missing.append(
            SolarMissingRequirement(
                field="project_type",
                required=True,
                reason="building type (rooftop versus ground-mounted) is not yet known",
            )
        )
    if draft.mode == "site_selection" and draft.site_selection_scope is None:
        missing.append(
            SolarMissingRequirement(
                field="site_selection_scope",
                required=True,
                reason="region_search, aoi_search, or candidate_ranking is not yet known",
            )
        )
    return tuple(missing)


def _solardss_missing(draft: SolarDecisionDraft) -> tuple[SolarMissingRequirement, ...] | None:
    if find_spec("solardss") is None:
        return None
    try:
        problem_mod = import_module("solardss.decision.problem")
        requirements_mod = import_module("solardss.decision.requirements")
        problem_cls = getattr(problem_mod, "SolarDecisionProblem", None)
        getter = getattr(requirements_mod, "get_missing_requirements", None)
        if problem_cls is None or not callable(getter):
            return None
        payload = _solardss_constructor_kwargs(draft)
        problem = problem_cls(**payload)
        records = getter(problem)
    except Exception:
        return None
    out: list[SolarMissingRequirement] = []
    for item in records:
        field = str(getattr(item, "field", "") or "")
        reason = str(getattr(item, "reason", "") or "missing")
        required = bool(getattr(item, "required", False))
        if not field:
            continue
        out.append(SolarMissingRequirement(field=field, required=required, reason=reason))
    return tuple(out)


def _solardss_constructor_kwargs(draft: SolarDecisionDraft) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": draft.mode,
        "project_type": draft.project_type,
    }
    if draft.mode == "site_selection":
        payload["site_selection_scope"] = draft.site_selection_scope
        bbox = _bounding_box(draft)
        if draft.site_selection_scope == "region_search" and draft.region is not None:
            payload["region"] = _geographic_ref(draft.region.model_dump(), bbox=bbox)
        elif draft.site_selection_scope == "aoi_search" and draft.aoi is not None:
            payload["aoi"] = _geographic_ref(draft.aoi.model_dump(), bbox=bbox)
        elif draft.site_selection_scope == "candidate_ranking" and draft.candidate_ids:
            payload["candidates"] = _candidate_set(draft.candidate_ids)
    elif draft.location is not None:
        payload["location"] = _geographic_ref(draft.location.model_dump())
    if draft.target_capacity_kw is not None:
        payload["target_capacity_kw"] = draft.target_capacity_kw
    if draft.budget_amount is not None:
        payload["budget"] = _quantity(draft.budget_amount, draft.budget_currency or "currency")
    if draft.objective is not None:
        payload["objective"] = draft.objective
    if draft.grid_connection_required is not None:
        payload["grid_connection_required"] = draft.grid_connection_required
    if draft.maximum_grid_distance_m is not None:
        payload["maximum_grid_distance"] = _quantity(draft.maximum_grid_distance_m, "m")
    if draft.battery_required is not None:
        payload["battery_required"] = draft.battery_required
    return {key: value for key, value in payload.items() if value is not None}


def _geographic_ref(data: dict[str, Any], bbox: Any | None = None) -> Any:
    geo_mod = import_module("solardss.data.geo")
    payload = {key: value for key, value in data.items() if value is not None}
    if bbox is not None:
        payload["bbox"] = bbox
    return geo_mod.GeographicRef(**payload)


def _bounding_box(draft: SolarDecisionDraft) -> Any | None:
    if draft.bbox is None:
        return None
    geo_mod = import_module("solardss.data.geo")
    west, south, east, north = draft.bbox
    return geo_mod.BoundingBox(min_lon=west, min_lat=south, max_lon=east, max_lat=north)


def _candidate_set(candidate_ids: tuple[str, ...]) -> Any:
    contracts = import_module("solardss.decision.contracts")
    return contracts.CandidateSet(candidate_ids=candidate_ids)


def _quantity(value: float, unit: str) -> Any:
    quantity_mod = import_module("solardss.core.quantity")
    return quantity_mod.Quantity(value=value, unit=unit)


def _host_missing(draft: SolarDecisionDraft) -> tuple[SolarMissingRequirement, ...]:
    return _host_spatial(draft) + _host_optional(draft)


def _host_spatial(draft: SolarDecisionDraft) -> tuple[SolarMissingRequirement, ...]:
    missing: list[SolarMissingRequirement] = []
    if draft.mode == "fixed_site":
        if draft.location is None:
            missing.append(
                SolarMissingRequirement(
                    field="location",
                    required=True,
                    reason="fixed-site evaluation requires location as given context",
                )
            )
        return tuple(missing)
    scope = draft.site_selection_scope
    if scope == "region_search" and draft.region is None:
        missing.append(
            SolarMissingRequirement(
                field="region",
                required=True,
                reason="region_search requires region",
            )
        )
    elif scope == "aoi_search" and draft.aoi is None:
        missing.append(
            SolarMissingRequirement(
                field="aoi",
                required=True,
                reason="aoi_search requires aoi",
            )
        )
    elif scope == "candidate_ranking" and not draft.candidate_ids:
        missing.append(
            SolarMissingRequirement(
                field="candidates",
                required=True,
                reason="candidate_ranking requires candidates",
            )
        )
    return tuple(missing)


def _host_optional(draft: SolarDecisionDraft) -> tuple[SolarMissingRequirement, ...]:
    missing: list[SolarMissingRequirement] = []
    if draft.target_capacity_kw is None:
        missing.append(
            SolarMissingRequirement(
                field="target_capacity_kw",
                required=False,
                reason=_OPTIONAL_CAPACITY_REASON,
            )
        )
    if draft.budget_amount is None:
        missing.append(
            SolarMissingRequirement(
                field="budget",
                required=False,
                reason=_OPTIONAL_BUDGET_REASON,
            )
        )
    if draft.objective is None:
        missing.append(
            SolarMissingRequirement(
                field="objective",
                required=False,
                reason=_OPTIONAL_OBJECTIVE_REASON,
            )
        )
    if draft.grid_connection_required is None:
        missing.append(
            SolarMissingRequirement(
                field="grid_connection_required",
                required=False,
                reason=_OPTIONAL_GRID_FLAG_REASON,
            )
        )
    elif draft.grid_connection_required is True and draft.maximum_grid_distance_m is None:
        missing.append(
            SolarMissingRequirement(
                field="maximum_grid_distance",
                required=True,
                reason=_OPTIONAL_GRID_DISTANCE_REASON,
            )
        )
    if draft.mode == "site_selection":
        missing.append(
            SolarMissingRequirement(
                field="environmental_preferences",
                required=False,
                reason=_OPTIONAL_ENV_REASON,
            )
        )
    if draft.battery_required is None:
        missing.append(
            SolarMissingRequirement(
                field="battery_required",
                required=False,
                reason=_OPTIONAL_BATTERY_REASON,
            )
        )
    missing.append(
        SolarMissingRequirement(
            field="module_preferences",
            required=False,
            reason=_OPTIONAL_MODULE_REASON,
        )
    )
    return tuple(missing)


def _execution_missing(
    draft: SolarDecisionDraft, message: str
) -> tuple[SolarMissingRequirement, ...]:
    """Plugin execution gaps. Never asks for GHI, slope, or other provider layers."""
    if draft.mode is None:
        return ()
    if draft.mode == "fixed_site":
        return _fixed_site_execution(draft, message)
    scope = draft.site_selection_scope
    if scope in {"region_search", "aoi_search"}:
        return _grid_search_execution(draft, message)
    if scope == "candidate_ranking":
        return _ranking_execution(draft, message)
    return ()


def _fixed_site_execution(
    draft: SolarDecisionDraft, message: str
) -> tuple[SolarMissingRequirement, ...]:
    missing: list[SolarMissingRequirement] = []
    if draft.location is None:
        missing.append(
            SolarMissingRequirement(
                field="location",
                required=True,
                reason="existing-building evaluation needs a location as given context",
            )
        )
    if yield_requested(message):
        missing.extend(_yield_execution(draft, ranking=False))
    return tuple(missing)


def _grid_search_execution(
    draft: SolarDecisionDraft, message: str
) -> tuple[SolarMissingRequirement, ...]:
    spatial_ok = (draft.site_selection_scope == "region_search" and draft.region is not None) or (
        draft.site_selection_scope == "aoi_search" and draft.aoi is not None
    )
    if not spatial_ok:
        return ()
    missing: list[SolarMissingRequirement] = []
    lowered = message.lower()
    if "bbox" in lowered and draft.bbox is None:
        missing.append(
            SolarMissingRequirement(
                field="bbox",
                required=True,
                reason="the user stated a bounding box but west,south,east,north are missing",
            )
        )
    if re.search(r"\b(regular|hexagonal)\s+grid\b", lowered) and draft.generation_strategy is None:
        missing.append(
            SolarMissingRequirement(
                field="generation_strategy",
                required=True,
                reason="the user stated a grid strategy that is not yet parsed",
            )
        )
    if (
        draft.generation_strategy not in {None, "features"}
        and (draft.cell_size is None or draft.cell_size_unit is None)
        and re.search(r"\bcell(?:[_\s-]?size)?\b", lowered)
    ):
        missing.append(
            SolarMissingRequirement(
                field="cell_size",
                required=True,
                reason="the user stated a grid that still needs cell_size",
            )
        )
    if ranking_requested(message) and draft.ranking_profile is None:
        missing.append(
            SolarMissingRequirement(
                field="ranking_profile",
                required=True,
                reason="ranking criteria are still missing",
            )
        )
    if yield_requested(message):
        missing.extend(
            _yield_execution(
                draft,
                ranking=ranking_requested(message) or draft.ranking_profile is not None,
            )
        )
    return tuple(missing)


def _ranking_execution(
    draft: SolarDecisionDraft, message: str
) -> tuple[SolarMissingRequirement, ...]:
    if not draft.candidate_ids:
        return ()
    missing: list[SolarMissingRequirement] = []
    if not candidates_have_geometry(draft):
        missing.append(
            SolarMissingRequirement(
                field="candidate_geometries",
                required=True,
                reason="candidate_ranking copies caller polygons; it does not invent locations",
            )
        )
    if draft.ranking_profile is None:
        missing.append(
            SolarMissingRequirement(
                field="ranking_profile",
                required=True,
                reason="ranking_profile is required and is not inferred",
            )
        )
    if yield_requested(message):
        missing.extend(_yield_execution(draft, ranking=True))
    return tuple(missing)


def _yield_execution(
    draft: SolarDecisionDraft, *, ranking: bool
) -> tuple[SolarMissingRequirement, ...]:
    """Capacity is a user decision. Tilt, azimuth, and loss are never asked here."""
    del ranking
    missing: list[SolarMissingRequirement] = []
    if draft.target_capacity_kw is None:
        missing.append(
            SolarMissingRequirement(
                field="target_capacity_kw",
                required=True,
                reason="yield needs installed capacity; it is not inferred",
            )
        )
    return tuple(missing)
