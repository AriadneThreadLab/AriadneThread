"""Host-side solar decision slots.

These models map onto SolarDSS ``SolarDecisionProblem``. They never hold
computed indicators (GHI, slope, elevation, and similar provider-filled layers).
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SolarDecisionMode = Literal["site_selection", "fixed_site"]
SolarSiteSelectionScope = Literal["region_search", "aoi_search", "candidate_ranking"]
SolarProjectType = Literal[
    "utility_scale",
    "ground_mounted",
    "rooftop",
    "commercial_rooftop",
    "residential_rooftop",
]
SolarRoutingStatus = Literal["needs_input", "routed"]
SolarGenerationStrategy = Literal["regular_grid", "hexagonal_grid", "features"]
SolarRankingProfile = Literal[
    "energy_focused",
    "infrastructure_focused",
    "environment_focused",
    "balanced",
]
SolarConstraintProfile = Literal[
    "utility_scale_default_example",
    "rooftop_default_example",
]
SolarMounting = Literal["ground", "rooftop"]
SolarYieldEngine = Literal["pvgis", "pvlib"]
SolarTracking = Literal["fixed", "single_axis", "dual_axis"]
SolarCellSizeUnit = Literal["m", "deg"]

_ROOFTOP_PROJECTS: frozenset[str] = frozenset(
    {"rooftop", "commercial_rooftop", "residential_rooftop"}
)
_GROUND_PROJECTS: frozenset[str] = frozenset({"utility_scale", "ground_mounted"})


def _mounting_from_project_type(project_type: SolarProjectType | None) -> SolarMounting | None:
    """Map a user-stated project class to mounting. Never invents tilt or loss."""
    if project_type is None:
        return None
    if project_type in _ROOFTOP_PROJECTS:
        return "rooftop"
    if project_type in _GROUND_PROJECTS:
        return "ground"
    return None


PROVIDER_FILLED_FIELDS: frozenset[str] = frozenset(
    {
        "ghi",
        "ghi_annual",
        "dni",
        "dhi",
        "slope",
        "slope_deg",
        "elevation",
        "elevation_m",
        "aspect",
        "aspect_deg",
        "shading",
        "shading_impact",
        "irradiation",
        "distance_to_roads",
        "distance_to_grid",
        "land_cover",
        "temperature",
        "air_temperature",
    }
)

#: Yield-engine knobs that SolarDSS may skip. Ask only if the user already stated them.
ENGINE_OPTIONAL_FIELDS: frozenset[str] = frozenset(
    {
        "tilt_deg",
        "azimuth_deg",
        "system_loss_percent",
        "mounting",
        "top_n",
    }
)

#: Engine method knobs. Never asked as raw API fields unless the user stated them.
ENGINE_INTERNAL_FIELDS: frozenset[str] = ENGINE_OPTIONAL_FIELDS | frozenset(
    {
        "bbox",
        "generation_strategy",
        "cell_size",
        "constraint_profile",
        "site_point",
        "module_preferences",
    }
)


def constraint_profile_from_project_type(
    project_type: SolarProjectType | None,
) -> SolarConstraintProfile | None:
    """Map building/project class to a shipped ConstraintProfile. Never invents science."""
    if project_type is None:
        return None
    if project_type in _ROOFTOP_PROJECTS:
        return "rooftop_default_example"
    if project_type in _GROUND_PROJECTS:
        return "utility_scale_default_example"
    return None


class SolarCandidateGeometry(BaseModel):
    """Caller-supplied candidate location. Coordinates are never invented."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1, max_length=80)
    name: str | None = Field(default=None, max_length=120)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    polygon: tuple[tuple[tuple[float, float], ...], ...] | None = None

    @model_validator(mode="after")
    def _point_or_polygon(self) -> SolarCandidateGeometry:
        if (self.longitude is None) != (self.latitude is None):
            raise ValueError("candidate longitude and latitude must be supplied together")
        if self.longitude is None and self.polygon is None:
            raise ValueError("candidate needs a point or polygon; locations are not invented")
        return self


class SolarPlaceRef(BaseModel):
    """Named or coordinate place handle. The host does not geocode here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str | None = Field(default=None, max_length=120)
    place_id: str | None = Field(default=None, max_length=80)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)

    @model_validator(mode="after")
    def _pair_and_identify(self) -> SolarPlaceRef:
        if (self.longitude is None) != (self.latitude is None):
            raise ValueError("longitude and latitude must be supplied together")
        if not any((self.name, self.place_id, self.longitude is not None)):
            raise ValueError("place requires a name, place_id, or coordinates")
        return self


class SolarMissingRequirement(BaseModel):
    """One gap copied from SolarDSS ``get_missing_requirements`` (or the host mirror)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str = Field(min_length=1, max_length=64)
    required: bool
    reason: str = Field(min_length=1, max_length=240)


class SolarFollowUpQuestion(BaseModel):
    """One user-facing question. Never a request for a provider-filled indicator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str = Field(min_length=1, max_length=64)
    prompt: str = Field(min_length=4, max_length=200)
    required: bool = True


class SolarDecisionDraft(BaseModel):
    """Partial SolarDecisionProblem slots extracted from conversation, not science."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: SolarDecisionMode | None = None
    site_selection_scope: SolarSiteSelectionScope | None = None
    project_type: SolarProjectType | None = None
    location: SolarPlaceRef | None = None
    region: SolarPlaceRef | None = None
    aoi: SolarPlaceRef | None = None
    candidate_ids: tuple[str, ...] = ()
    candidate_geometries: tuple[SolarCandidateGeometry, ...] = ()
    target_capacity_kw: float | None = Field(default=None, ge=0.0)
    budget_amount: float | None = Field(default=None, ge=0.0)
    budget_currency: str | None = Field(default=None, max_length=8)
    objective: str | None = Field(default=None, max_length=80)
    grid_connection_required: bool | None = None
    maximum_grid_distance_m: float | None = Field(default=None, ge=0.0)
    battery_required: bool | None = None
    capability_id: str | None = Field(default=None, max_length=48)
    bbox: tuple[float, float, float, float] | None = None
    generation_strategy: SolarGenerationStrategy | None = None
    cell_size: float | None = Field(default=None, gt=0.0)
    cell_size_unit: SolarCellSizeUnit | None = None
    ranking_profile: SolarRankingProfile | None = None
    constraint_profile: SolarConstraintProfile | None = None
    mounting: SolarMounting | None = None
    tracking: SolarTracking | None = None
    yield_engine: SolarYieldEngine | None = None
    system_loss_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    tilt_deg: float | None = Field(default=None, ge=0.0, le=90.0)
    azimuth_deg: float | None = Field(default=None, ge=0.0, le=360.0)
    top_n: int | None = Field(default=None, ge=1, le=20)

    @field_validator("candidate_ids")
    @classmethod
    def _unique_candidates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value if item.strip())
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("candidate_ids must be unique")
        return cleaned

    @model_validator(mode="after")
    def _bbox_order(self) -> SolarDecisionDraft:
        if self.bbox is not None:
            west, south, east, north = self.bbox
            if east < west or north < south:
                raise ValueError("bbox maxima must be >= minima")
        return self

    @model_validator(mode="after")
    def _mode_scope(self) -> SolarDecisionDraft:
        updates: dict[str, object] = {}
        if self.mode == "fixed_site":
            if self.site_selection_scope is not None:
                updates["site_selection_scope"] = None
            if self.region is not None:
                updates["region"] = None
            if self.aoi is not None:
                updates["aoi"] = None
            if self.candidate_ids:
                updates["candidate_ids"] = ()
            if self.candidate_geometries:
                updates["candidate_geometries"] = ()
        elif self.mode == "site_selection" and self.location is not None:
            updates["location"] = None
        if not updates:
            return self
        return self.model_copy(update=updates)

    def filled_fields(self) -> tuple[str, ...]:
        filled: list[str] = []
        for name in (
            "mode",
            "site_selection_scope",
            "project_type",
            "location",
            "region",
            "aoi",
            "target_capacity_kw",
            "budget_amount",
            "objective",
            "grid_connection_required",
            "maximum_grid_distance_m",
            "battery_required",
            "capability_id",
            "bbox",
            "generation_strategy",
            "cell_size",
            "cell_size_unit",
            "ranking_profile",
            "constraint_profile",
            "mounting",
            "tracking",
            "yield_engine",
            "system_loss_percent",
            "tilt_deg",
            "azimuth_deg",
            "top_n",
        ):
            if getattr(self, name) is not None:
                filled.append(name)
        if self.candidate_ids:
            filled.append("candidates")
        if self.candidate_geometries:
            filled.append("candidate_geometries")
        return tuple(filled)

    def to_problem_payload(self) -> dict[str, Any]:
        """JSON-shaped SolarDecisionProblem fields. No computed indicators."""
        payload: dict[str, Any] = {
            "mode": self.mode,
            "project_type": self.project_type,
            "site_selection_scope": self.site_selection_scope,
        }
        if self.location is not None:
            payload["location"] = self.location.model_dump()
        if self.region is not None:
            payload["region"] = self.region.model_dump()
        if self.aoi is not None:
            payload["aoi"] = self.aoi.model_dump()
        if self.candidate_ids:
            payload["candidates"] = {"candidate_ids": list(self.candidate_ids)}
        if self.target_capacity_kw is not None:
            payload["target_capacity_kw"] = self.target_capacity_kw
        if self.budget_amount is not None:
            payload["budget"] = {
                "value": self.budget_amount,
                "unit": self.budget_currency or "currency",
            }
        if self.objective is not None:
            payload["objective"] = self.objective
        if self.grid_connection_required is not None:
            payload["grid_connection_required"] = self.grid_connection_required
        if self.maximum_grid_distance_m is not None:
            payload["maximum_grid_distance"] = {
                "value": self.maximum_grid_distance_m,
                "unit": "m",
            }
        if self.battery_required is not None:
            payload["battery_required"] = self.battery_required
        return payload

    def execution_parameters(self) -> dict[str, Any]:
        """Plugin parameters. Never includes computed GHI or other indicators."""
        payload: dict[str, Any] = {}
        if self.project_type is not None:
            payload["project_type"] = self.project_type
        if self.site_selection_scope is not None:
            payload["site_selection_scope"] = self.site_selection_scope
        if self.generation_strategy is not None:
            payload["generation_strategy"] = self.generation_strategy
        if self.cell_size is not None:
            payload["cell_size"] = self.cell_size
        if self.cell_size_unit is not None:
            payload["cell_size_unit"] = self.cell_size_unit
        if self.ranking_profile is not None:
            payload["ranking_profile"] = self.ranking_profile
        if self.constraint_profile is not None:
            payload["constraint_profile"] = self.constraint_profile
        mounting = self.mounting or _mounting_from_project_type(self.project_type)
        if mounting is not None:
            payload["mounting"] = mounting
        if self.tracking is not None:
            payload["tracking"] = self.tracking
        if self.yield_engine is not None:
            payload["yield_engine"] = self.yield_engine
        if self.system_loss_percent is not None:
            payload["system_loss_percent"] = self.system_loss_percent
        if self.tilt_deg is not None:
            payload["tilt_deg"] = self.tilt_deg
        if self.azimuth_deg is not None:
            payload["azimuth_deg"] = self.azimuth_deg
        if self.top_n is not None:
            payload["top_n"] = self.top_n
        if self.target_capacity_kw is not None:
            payload["installed_capacity_kw"] = self.target_capacity_kw
        return payload


class SolarDecisionReport(BaseModel):
    """Routing result: structured slots, capability id, and adaptive questions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: SolarRoutingStatus
    mode: SolarDecisionMode | None = None
    site_selection_scope: SolarSiteSelectionScope | None = None
    project_type: SolarProjectType | None = None
    capability_id: str | None = None
    problem: dict[str, Any] = Field(default_factory=dict)
    missing_requirements: tuple[SolarMissingRequirement, ...] = ()
    questions: tuple[SolarFollowUpQuestion, ...] = ()
    filled_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    requirements_source: Literal["solardss", "host_mirror"] = "host_mirror"


class SolarDecisionStore(Protocol):
    """Persists structured solar slots across turns of one conversation."""

    def load(self, conversation_id: str) -> SolarDecisionDraft | None: ...

    def save(self, conversation_id: str, draft: SolarDecisionDraft) -> None: ...
