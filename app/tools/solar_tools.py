"""``analyze_solar_site`` — host contract for solar decision routing.

The LLM extracts intent, slots, and a registered capability_id. This tool
merges those slots with the user message and persisted conversation context,
calls SolarDSS ``get_missing_requirements()``, and returns only the follow-up
questions the user must still answer. It does not compute scientific indicators.
"""

from __future__ import annotations

import json
import logging
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.analytics.charts import AnalysisChart
from app.analytics.plugin_result import (
    AgentQuestion,
    ScientificAnalysisResult,
    agent_questions_from_solar,
    scientific_result_from_solar,
)
from app.core.errors import (
    SolarAnalysisFailedError,
    SolarAnalysisInfeasibleError,
    SolarPluginInternalError,
    SolarPluginUnavailableError,
    ToolError,
)
from app.osm.contracts import GeoJsonFeatureCollection
from app.solar.capabilities import (
    allowed_host_capability_ids,
    analyze_solar_site_tool_description,
    capability_id_field_description,
    capability_id_looks_valid,
    capability_json_schema,
    reconcile_capability,
)
from app.solar.contracts import (
    SolarCandidateGeometry,
    SolarDecisionDraft,
    SolarDecisionReport,
    SolarFollowUpQuestion,
    SolarMissingRequirement,
    SolarProjectType,
)
from app.solar.display import SolarDecisionSupportReport, copy_solar_outputs
from app.solar.questions import questions_for_missing
from app.solar.requirements import missing_requirements_for
from app.solar.slots import (
    draft_from_tool_args,
    extract_solar_slots,
    mentioned_optional_fields,
    merge_solar_drafts,
)
from app.solar.store import InMemorySolarDecisionStore
from app.tools.context import ToolContext
from app.tools.contracts import ToolOutcome

logger = logging.getLogger(__name__)

TOOL_NAME = "analyze_solar_site"
TOOL_DESCRIPTION = analyze_solar_site_tool_description()
SOLAR_DECISION_ATTRIBUTION = (
    "Solar PV decision routing via SolarDSS. Indicators are not computed in Ariadne."
)
SOLAR_MAP_ATTRIBUTION = (
    "SolarDSS map geometry and indicators copied from the SolarDSS engine. Not OpenStreetMap."
)


class AnalyzeSolarSiteArgs(BaseModel):
    """Closed slot contract. The model must not send computed indicator values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_id: str | None = Field(
        default=None,
        description=capability_id_field_description(),
    )
    mode: Literal["site_selection", "fixed_site"] | None = Field(
        default=None,
        description="SITE_SELECTION or FIXED_SITE. Omit when unknown.",
    )
    site_selection_scope: Literal["region_search", "aoi_search", "candidate_ranking"] | None = (
        Field(
            default=None,
            description="Required for SITE_SELECTION. Omit for FIXED_SITE.",
        )
    )
    project_type: SolarProjectType | None = Field(
        default=None,
        description="Rooftop versus ground-mounted class, if the user stated it.",
    )
    region_name: str | None = Field(default=None, max_length=120)
    aoi_name: str | None = Field(default=None, max_length=120)
    location_name: str | None = Field(default=None, max_length=120)
    location_longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    location_latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    candidate_ids: tuple[str, ...] = ()
    candidate_geometries: tuple[SolarCandidateGeometry, ...] = ()
    target_capacity_kw: float | None = Field(default=None, ge=0.0)
    budget_amount: float | None = Field(default=None, ge=0.0)
    budget_currency: str | None = Field(default=None, max_length=8)
    objective: str | None = Field(default=None, max_length=80)
    grid_connection_required: bool | None = None
    maximum_grid_distance_m: float | None = Field(default=None, ge=0.0)
    battery_required: bool | None = None
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Search bbox west,south,east,north. Not geocoded from a place name.",
    )
    generation_strategy: Literal["regular_grid", "hexagonal_grid", "features"] | None = Field(
        default=None,
        description="How candidates are obtained. Ranking never invents a grid.",
    )
    cell_size: float | None = Field(default=None, gt=0.0)
    cell_size_unit: Literal["m", "deg"] | None = None
    ranking_profile: (
        Literal[
            "energy_focused",
            "infrastructure_focused",
            "environment_focused",
            "balanced",
        ]
        | None
    ) = Field(
        default=None,
        description="Shipped example DecisionProfile id. Weights are never inferred.",
    )
    constraint_profile: (
        Literal["utility_scale_default_example", "rooftop_default_example"] | None
    ) = Field(
        default=None,
        description="Shipped example ConstraintProfile id. Not a universal standard.",
    )
    mounting: Literal["ground", "rooftop"] | None = None
    tracking: Literal["fixed", "single_axis", "dual_axis"] | None = None
    yield_engine: Literal["pvgis", "pvlib"] | None = None
    system_loss_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    tilt_deg: float | None = Field(default=None, ge=0.0, le=90.0)
    azimuth_deg: float | None = Field(default=None, ge=0.0, le=360.0)
    top_n: int | None = Field(default=None, ge=1, le=20)

    @field_validator("capability_id")
    @classmethod
    def _normalize_capability(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("capability_id must be a string")
        text = value.strip()
        if not text:
            return None
        if not capability_id_looks_valid(text):
            raise ValueError("capability_id must be a closed catalog identifier")
        if text not in allowed_host_capability_ids():
            raise ValueError(
                "unsupported capability_id; "
                f"registered: {', '.join(capability_json_schema()['enum'])}"
            )
        return text

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        schema = super().model_json_schema(*args, **kwargs)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            properties["capability_id"] = capability_json_schema()
        return schema


class AnalyzeSolarSiteResult(BaseModel):
    """Routing payload plus optional copied SolarDSS map/indicator outputs.

    ``execute`` still returns routing only. GeoJSON, charts, and ``support`` are
    pass-through copies from the integration layer. Ariadne never fills them.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["needs_input", "routed"]
    capability: str | None = None
    mode: str | None = None
    site_selection_scope: str | None = None
    project_type: str | None = None
    problem: dict[str, Any] = Field(default_factory=dict)
    filled_fields: list[str] = Field(default_factory=list)
    missing_requirements: list[dict[str, Any]] = Field(default_factory=list)
    questions: list[SolarFollowUpQuestion] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_title: str = "SolarDSS decision routing"
    detail: str = ""
    report: SolarDecisionReport | None = None
    geojson: GeoJsonFeatureCollection | None = Field(
        default=None,
        description="SolarDSS FeatureCollection copied from the plugin. Never invented.",
    )
    charts: list[AnalysisChart] = Field(
        default_factory=list,
        description="Chart specs copied from SolarDSS. Series are never model-authored.",
    )
    support: SolarDecisionSupportReport | None = Field(
        default=None,
        description="Copied SolarDSS indicators for the UI. Absent when the engine returned none.",
    )
    analysis_result: ScientificAnalysisResult | None = Field(
        default=None,
        description="Unified plugin envelope. Present when SolarDSS copied outputs exist.",
    )
    agent_questions: list[AgentQuestion] = Field(
        default_factory=list,
        description="Clarification prompts for the Agent Questions panel.",
    )
    feature_count: int | None = Field(default=None, ge=0)


class SolarAnalysisBackend(Protocol):
    """Adapter port. Receives a registered capability id plus a validated request."""

    def analyze(self, *, capability_id: str, request: dict[str, Any]) -> dict[str, Any]: ...


class AnalyzeSolarSiteTool:
    """Merge slots, persist context, ask only SolarDSS-required user fields."""

    def __init__(
        self,
        store: InMemorySolarDecisionStore | None = None,
        backend: SolarAnalysisBackend | None = None,
    ) -> None:
        self._store = store if store is not None else InMemorySolarDecisionStore()
        self._backend = backend if backend is not None else PluginSolarAnalysisBackend()

    @property
    def name(self) -> str:
        return TOOL_NAME

    @property
    def description(self) -> str:
        return analyze_solar_site_tool_description()

    @property
    def args_model(self) -> type[AnalyzeSolarSiteArgs]:
        return AnalyzeSolarSiteArgs

    async def execute(
        self, args: AnalyzeSolarSiteArgs, context: ToolContext
    ) -> ToolOutcome[AnalyzeSolarSiteResult]:
        store = context.solar_store if context.solar_store is not None else self._store
        conversation_id = (context.conversation_id or "").strip()
        persisted = store.load(conversation_id) if conversation_id else None
        extracted = extract_solar_slots(context.user_message)
        model_slots = draft_from_tool_args(args.model_dump())
        draft = merge_solar_drafts(
            persisted=persisted,
            extracted=extracted,
            model_slots=model_slots,
        )
        capability = reconcile_capability(
            args.capability_id or draft.capability_id,
            message=context.user_message,
            draft=draft,
        )
        draft = draft.model_copy(update={"capability_id": capability})
        missing, source = missing_requirements_for(draft, context.user_message)
        questions = questions_for_missing(
            missing,
            draft=draft,
            mentioned=mentioned_optional_fields(context.user_message),
        )
        if conversation_id:
            store.save(conversation_id, draft)
        context.analysis.solar_capability_requested = args.capability_id
        context.analysis.solar_capability_canonical = capability
        context.analysis.solar_draft = draft
        status: Literal["needs_input", "routed"] = "needs_input" if questions else "routed"
        warnings: list[str] = []
        support = None
        geojson = None
        charts: list[AnalysisChart] = []
        feature_count = None
        science = "not_computed"
        plugin_detail = ""
        if status == "routed" and capability:
            try:
                payload = self._backend.analyze(
                    capability_id=capability,
                    request=_execution_request(draft, capability),
                )
            except SolarPluginUnavailableError as exc:
                warnings.append(str(exc.message))
            except ToolError:
                raise
            except Exception as exc:
                logger.exception(
                    "analyze_solar_site plugin failed capability_id=%s",
                    capability,
                )
                raise SolarPluginInternalError(
                    f"SolarDSS plugin raised {type(exc).__name__} for capability_id={capability!r}",
                    capability_id=capability,
                ) from exc
            else:
                host_status = str(payload.get("status", ""))
                plugin_detail = str(payload.get("detail", "") or "")
                if host_status in {"success", "completed"}:
                    support, geojson, charts = copy_solar_outputs(
                        payload,
                        routing_mode=draft.mode,
                        routing_project_type=draft.project_type,
                        routing_capability=capability,
                    )
                    if geojson is not None:
                        feature_count = len(geojson.get("features", []))
                    science = "copied_from_solardss"
                    extra_warnings = payload.get("warnings")
                    if isinstance(extra_warnings, list | tuple):
                        warnings.extend(str(item) for item in extra_warnings)
                elif host_status == "unavailable":
                    warnings.append(plugin_detail or "SolarDSS plugin unavailable")
                elif host_status in {"rejected", "infeasible", "not_bound"}:
                    raise SolarAnalysisInfeasibleError(
                        plugin_detail or host_status,
                        capability_id=capability,
                    )
                else:
                    raise SolarAnalysisFailedError(
                        plugin_detail or host_status,
                        capability_id=capability,
                    )
        report = SolarDecisionReport(
            status=status,
            mode=draft.mode,
            site_selection_scope=draft.site_selection_scope,
            project_type=draft.project_type,
            capability_id=capability,
            problem=draft.to_problem_payload(),
            missing_requirements=missing,
            questions=questions,
            filled_fields=draft.filled_fields(),
            warnings=tuple(warnings),
            requirements_source=source,
        )
        result = AnalyzeSolarSiteResult(
            status=status,
            capability=capability,
            mode=draft.mode,
            site_selection_scope=draft.site_selection_scope,
            project_type=draft.project_type,
            problem=report.problem,
            filled_fields=list(report.filled_fields),
            missing_requirements=[item.model_dump() for item in missing],
            questions=list(questions),
            warnings=warnings,
            detail=_detail(status, draft, questions, source, science=science),
            report=report,
            geojson=geojson,
            charts=charts,
            support=support,
            analysis_result=scientific_result_from_solar(
                support,
                geojson=geojson,
                charts=charts,
                capability_id=capability,
            ),
            agent_questions=agent_questions_from_solar(questions),
            feature_count=feature_count,
        )
        observation = _observation(result, missing, science=science)
        logger.info(
            "analyze_solar_site status=%s mode=%s scope=%s capability=%s questions=%s "
            "source=%s science=%s",
            status,
            draft.mode or "-",
            draft.site_selection_scope or "-",
            capability or "-",
            [item.field for item in questions],
            source,
            science,
        )
        return ToolOutcome(observation=observation, payload=result)


def _detail(
    status: str,
    draft: SolarDecisionDraft,
    questions: tuple[SolarFollowUpQuestion, ...],
    source: str,
    *,
    science: str,
) -> str:
    if status == "needs_input":
        fields = ", ".join(item.field for item in questions) or "none"
        return (
            f"Structured solar slots are incomplete. Ask only: {fields}. "
            f"Requirements from {source}. Do not compute indicators."
        )
    if science == "copied_from_solardss":
        return (
            f"SolarDSS computed mode={draft.mode} "
            f"scope={draft.site_selection_scope} capability={draft.capability_id}. "
            "Numbers are copied from SolarDSS, not invented."
        )
    return (
        f"Solar decision routed as mode={draft.mode} "
        f"scope={draft.site_selection_scope} capability={draft.capability_id}. "
        f"Requirements from {source}. Indicators are not computed in this tool."
    )


def _observation(
    result: AnalyzeSolarSiteResult,
    missing: tuple[SolarMissingRequirement, ...],
    *,
    science: str,
) -> str:
    question_payload = [item.model_dump() for item in result.questions]
    missing_payload = [item.model_dump() for item in missing]
    instruction = (
        "Do not calculate GHI, slope, elevation, distance to roads, distance "
        "to grid, land cover, or temperature. Ask only the listed Decision "
        "Support questions. If the goal is missing, ask only existing building "
        "evaluation vs site selection vs candidate ranking. Do not ask tilt, "
        "azimuth, losses, bbox, or cell size unless listed. If questions is "
        "empty, summarize the structured decision context and do not ask "
        "redundant questions."
    )
    if science == "copied_from_solardss":
        instruction = (
            "Scientific numbers, geometries, and charts were copied from SolarDSS. "
            "Do not invent replacements. Partition DETERMINISTIC ANALYTICAL EVIDENCE "
            "from KNOWLEDGE / DOCUMENT EVIDENCE."
        )
    payload = {
        "status": result.status,
        "source": "solar_decision_routing",
        "mode": result.mode,
        "site_selection_scope": result.site_selection_scope,
        "project_type": result.project_type,
        "capability_id": result.capability,
        "filled_fields": result.filled_fields,
        "problem": result.problem,
        "missing_requirements": missing_payload,
        "questions": question_payload,
        "science": science,
        "feature_count": result.feature_count,
        "instruction": instruction,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _execution_request(draft: SolarDecisionDraft, capability_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "capability_id": capability_id,
        "parameters": draft.execution_parameters(),
    }
    place = draft.location or draft.aoi or draft.region
    if place is not None:
        location: dict[str, Any] = {}
        if place.name:
            location["name"] = place.name
        if place.place_id:
            location["place_id"] = place.place_id
        if place.longitude is not None and place.latitude is not None:
            location["longitude"] = place.longitude
            location["latitude"] = place.latitude
        if location:
            payload["location"] = location
    if draft.bbox is not None:
        payload["bbox"] = list(draft.bbox)
    if draft.candidate_geometries:
        payload["candidates"] = [
            _candidate_execution_record(item) for item in draft.candidate_geometries
        ]
    return payload


def _candidate_execution_record(item: SolarCandidateGeometry) -> dict[str, Any]:
    record: dict[str, Any] = {"candidate_id": item.candidate_id}
    if item.name:
        record["name"] = item.name
    if item.longitude is not None and item.latitude is not None:
        record["longitude"] = item.longitude
        record["latitude"] = item.latitude
    if item.polygon is not None:
        record["polygon"] = item.polygon
    return record


class PluginSolarAnalysisBackend:
    """Dispatch a validated request to ``ariadne_solar.analyze``. No science here."""

    def analyze(self, *, capability_id: str, request: dict[str, Any]) -> dict[str, Any]:
        module = _require_solar_plugin()
        analyze_fn = getattr(module, "analyze", None)
        if not callable(analyze_fn):
            raise SolarPluginUnavailableError(
                "installed ariadne_solar package is missing analyze()",
                capability_id=capability_id,
            )
        try:
            payload = analyze_fn(capability_id, request)
        except ToolError:
            raise
        except Exception as exc:
            logger.exception(
                "SolarDSS plugin analyze raised capability_id=%s",
                capability_id,
            )
            raise SolarPluginInternalError(
                f"SolarDSS plugin raised {type(exc).__name__} for capability_id={capability_id!r}",
                capability_id=capability_id,
            ) from exc
        if not isinstance(payload, dict):
            raise SolarPluginInternalError(
                f"SolarDSS plugin returned a non-object result for capability_id={capability_id!r}",
                capability_id=capability_id,
            )
        return payload


def _require_solar_plugin() -> Any:
    if find_spec("ariadne_solar") is None:
        raise SolarPluginUnavailableError(
            "ariadne_solar is not installed. OSM and GeoLoadST remain available. "
            "Install the SolarDSS plugin to execute solar analysis."
        )
    return import_module("ariadne_solar")


def build_analyze_solar_site_tool(
    store: InMemorySolarDecisionStore | None = None,
    backend: SolarAnalysisBackend | None = None,
) -> AnalyzeSolarSiteTool:
    """Composition helper. Does not import SolarDSS at call time."""
    return AnalyzeSolarSiteTool(store=store, backend=backend)
