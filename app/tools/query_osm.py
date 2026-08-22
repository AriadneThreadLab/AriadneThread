"""``query_osm`` - retrieve real OpenStreetMap features through Overpass.

The model never writes Overpass QL. It fills in :class:`OsmFeatureQuery`, which
this tool renders with the deterministic builder and hands to the transport.
The generated query is returned as provenance alongside the features.

The compact observation intentionally excludes GeoJSON so large feature
collections are never injected into the LLM context.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.geojson_combine import annotate_features_for_target
from app.core.errors import (
    OverpassError,
    OverpassQueryBuildError,
    ToolArgumentError,
    ToolExecutionError,
)
from app.osm.contracts import (
    OSM_ATTRIBUTION,
    GeoJsonFeatureCollection,
    OsmGeoJsonEncoder,
    OverpassClient,
)
from app.osm.query_builder import build_overpass_query
from app.osm.query_spec import OsmFeatureQuery, PointRadius
from app.tools.context import ToolContext
from app.tools.contracts import ToolOutcome
from app.tools.overpass_failures import describe_scope as _scope_label
from app.tools.overpass_failures import describe_tags as _tag_labels
from app.tools.overpass_failures import (
    format_overpass_failure_observation,
    public_overpass_error_message,
)

TOOL_NAME = "query_osm"
TOOL_DESCRIPTION = (
    "Retrieve live OpenStreetMap features. Required: tags as "
    '[{"key":"leisure","value":"park"}] and exactly one scope — place (cities), '
    "point (user-supplied coords), bbox (user-supplied bounds), or "
    "place_ref_scope (trusted resolve_place ref + user radius_m). Optional limit "
    "(integer, not max_items). Never invent coordinates or Overpass QL. "
    "Only source of live map data. One spatial target per call."
)

#: Arguments are the validated query spec itself; no parallel schema exists.
QueryOsmArgs = OsmFeatureQuery


class OsmQuerySource(BaseModel):
    """Where the returned features came from."""

    model_config = ConfigDict(frozen=True)

    endpoint: str
    attribution: str = OSM_ATTRIBUTION
    retrieved_at: datetime
    source_type: str = "live_osm"


class QueryOsmResult(BaseModel):
    """Structured result kept for the API/UI (not the LLM observation)."""

    model_config = ConfigDict(frozen=True)

    feature_count: int = Field(ge=0)
    geojson: GeoJsonFeatureCollection
    overpass_query: str
    source: OsmQuerySource
    warnings: list[str] = Field(default_factory=list)
    truncated: bool = False
    effective_limit: int = Field(ge=1)
    scope_summary: str
    analysis_target: str | None = None
    place_ref: str | None = None


def _describe_scope(query: OsmFeatureQuery, *, target_label: str | None = None) -> str:
    if query.place is not None:
        return f"in {query.place}"
    if query.place_ref_scope is not None:
        label = target_label or query.place_ref_scope.place_ref
        return f"within {query.place_ref_scope.radius_m} m of {label}"
    if query.point is not None:
        return f"within {query.point.radius_m} m of {query.point.lat:g},{query.point.lon:g}"
    box = query.bbox
    assert box is not None
    return f"in bbox {box.south:g},{box.west:g},{box.north:g},{box.east:g}"


def _scope_summary(
    query: OsmFeatureQuery,
    *,
    target_label: str | None = None,
) -> str:
    if query.place is not None:
        return f"Search area: {query.place}"
    if query.place_ref_scope is not None:
        label = target_label or query.place_ref_scope.place_ref
        return (
            f"Analysis area: {query.place_ref_scope.radius_m} m around "
            f"{label} (trusted place_ref={query.place_ref_scope.place_ref})"
        )
    if query.point is not None:
        return (
            f"Analysis area: {query.point.radius_m} m around "
            f"{query.point.lat:g}, {query.point.lon:g}"
        )
    box = query.bbox
    assert box is not None
    return (
        f"Bounding box: south={box.south:g}, west={box.west:g}, "
        f"north={box.north:g}, east={box.east:g}"
    )


def _resolve_to_executable(
    args: OsmFeatureQuery,
    context: ToolContext,
) -> tuple[OsmFeatureQuery, str | None, str | None]:
    """Translate place_ref_scope into a point query using trusted registry coords."""
    if args.place_ref_scope is None:
        return args, None, None
    place_ref = args.place_ref_scope.place_ref
    record = context.places.get(place_ref)
    if record is None:
        known = ", ".join(context.places.refs()) or "none"
        raise ToolArgumentError(
            f"unknown place_ref '{place_ref}'; available in this request: {known}"
        )
    point_query = args.model_copy(
        update={
            "place_ref_scope": None,
            "point": PointRadius(
                lat=record.latitude,
                lon=record.longitude,
                radius_m=args.place_ref_scope.radius_m,
            ),
        }
    )
    return point_query, record.label, place_ref


def _enforce_grounded_tags(args: OsmFeatureQuery, context: ToolContext) -> None:
    grounded = context.grounding.tags
    if not grounded:
        return
    requested = [tag.key if tag.value is None else f"{tag.key}={tag.value}" for tag in args.tags]
    if sorted(tag.lower() for tag in requested) == sorted(tag.lower() for tag in grounded):
        return
    raise ToolArgumentError(
        "query_osm tags must reuse the grounded tag selection "
        f"({', '.join(grounded)}); silent substitution is not allowed. "
        "Call search_osm_knowledge again if a different tag is required."
    )


def _describe_tags(query: OsmFeatureQuery) -> str:
    return ", ".join(
        tag.key if tag.value is None else f"{tag.key}={tag.value}" for tag in query.tags
    )


class QueryOsmToolError(ToolExecutionError):
    """query_osm failed after a validated OsmFeatureQuery was built.

    ``observation`` is the compact JSON the model should see. ``failure_meta``
    carries safe provenance for the accumulator / execution trace.
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        observation: str,
        failure_meta: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.code = code
        self.observation = observation
        self.failure_meta = failure_meta


class QueryOsmTool:
    """Tool implementation over an :class:`OverpassClient`."""

    def __init__(
        self,
        client: OverpassClient,
        encoder: OsmGeoJsonEncoder,
        *,
        timeout_seconds: int,
        max_results: int,
    ) -> None:
        self._client = client
        self._encoder = encoder
        self._timeout_seconds = timeout_seconds
        self._max_results = max_results

    @property
    def name(self) -> str:
        return TOOL_NAME

    @property
    def description(self) -> str:
        return TOOL_DESCRIPTION

    @property
    def args_model(self) -> type[OsmFeatureQuery]:
        return QueryOsmArgs

    async def aclose(self) -> None:
        await self._client.aclose()

    async def execute(
        self,
        args: OsmFeatureQuery,
        context: ToolContext,
    ) -> ToolOutcome[QueryOsmResult]:
        _enforce_grounded_tags(args, context)
        executable, target_label, place_ref = _resolve_to_executable(args, context)
        effective = self._apply_server_limit(executable)
        # Preserve original place_ref_scope metadata for summaries / provenance.
        summary_query = args.model_copy(update={"limit": effective.limit})
        try:
            query = build_overpass_query(effective, timeout_seconds=self._timeout_seconds)
        except OverpassQueryBuildError as exc:
            raise ToolExecutionError(f"could not build Overpass query: {exc.message}") from exc

        try:
            response = await self._client.run(query)
        except OverpassError as exc:
            raise self._as_tool_error(
                summary_query if args.place_ref_scope is not None else effective,
                query,
                exc,
                target_label=target_label,
            ) from exc

        conversion = self._encoder.encode(response.elements)
        raw_features = conversion.feature_collection.get("features", [])
        features: list[Any] = list(raw_features) if isinstance(raw_features, list) else []
        warnings = list(response.warnings) + list(conversion.warnings)
        truncated = response.truncated
        # Hard safety cap: never return more features than the validated limit,
        # even if Overpass ignores the ``out … N`` bound. Hitting the limit
        # exactly is also treated as incomplete — Overpass may have stopped at N.
        if len(features) > effective.limit:
            features = features[: effective.limit]
            truncated = True
            warnings.append(f"Feature list truncated to the configured limit of {effective.limit}")
        elif len(features) == effective.limit:
            truncated = True
            warnings.append(
                f"Feature count equals the configured limit of {effective.limit}; "
                "treat this count as a lower bound, not a complete inventory."
            )

        geojson: GeoJsonFeatureCollection = {
            "type": "FeatureCollection",
            "features": features,
        }
        if target_label is not None:
            geojson = annotate_features_for_target(
                geojson,
                analysis_target=target_label,
                place_ref=place_ref,
            )
        result = QueryOsmResult(
            feature_count=len(geojson.get("features", [])),
            geojson=geojson,
            overpass_query=query,
            source=OsmQuerySource(
                endpoint=response.endpoint,
                retrieved_at=response.retrieved_at,
            ),
            warnings=warnings,
            truncated=truncated,
            effective_limit=effective.limit,
            scope_summary=_scope_summary(summary_query, target_label=target_label),
            analysis_target=target_label,
            place_ref=place_ref,
        )
        # Register the executable point query so density/area analytics work.
        dataset_ref = context.datasets.register(result, effective)
        if context.grounding.tags is None:
            context.grounding.tags = [
                tag.key if tag.value is None else f"{tag.key}={tag.value}" for tag in args.tags
            ]
        return ToolOutcome(
            observation=self._observation(
                summary_query,
                result,
                dataset_ref=dataset_ref,
                target_label=target_label,
            ),
            payload=result,
        )

    def _as_tool_error(
        self,
        query: OsmFeatureQuery,
        overpass_ql: str,
        error: OverpassError,
        *,
        target_label: str | None = None,
    ) -> QueryOsmToolError:
        if query.place is not None:
            scope_type = "place"
        elif query.place_ref_scope is not None:
            scope_type = "place_ref"
        elif query.point is not None:
            scope_type = "point"
        else:
            scope_type = "bbox"
        return QueryOsmToolError(
            code=error.code,
            message=public_overpass_error_message(error),
            observation=format_overpass_failure_observation(query, error),
            failure_meta={
                "overpass_query": overpass_ql,
                "effective_limit": query.limit,
                "scope_summary": _scope_summary(query, target_label=target_label),
                "validated_tags": _tag_labels(query),
                "named_place": query.place,
                "place_ref": (
                    query.place_ref_scope.place_ref if query.place_ref_scope is not None else None
                ),
                "scope_type": scope_type,
                "scope": _scope_label(query),
                "upstream_status": error.upstream_status,
                "overpass_attempts": error.attempts,
                "error_code": error.code,
                "grounding_remains_valid": True,
            },
        )

    def _apply_server_limit(self, args: OsmFeatureQuery) -> OsmFeatureQuery:
        """Clamp the model-requested limit to the configured server maximum."""
        if args.limit <= self._max_results:
            return args
        return args.model_copy(update={"limit": self._max_results})

    def _observation(
        self,
        query: OsmFeatureQuery,
        result: QueryOsmResult,
        *,
        dataset_ref: str,
        target_label: str | None = None,
    ) -> str:
        """Compact agent-facing summary. Must not include GeoJSON."""
        status = "empty" if result.feature_count == 0 else "ok"
        if result.feature_count > 0 and result.warnings:
            status = "ok_with_warnings"
        scope = _describe_scope(query, target_label=target_label)
        target_bit = f" analysis_target={target_label}" if target_label else ""
        parts = [
            (
                f"status={status} source=live_osm "
                f"feature_count={result.feature_count} "
                f"dataset_ref={dataset_ref}{target_bit} "
                f"warnings={len(result.warnings)} "
                f"query={_describe_tags(query)} {scope}"
            ),
            (
                f"Queried live OpenStreetMap via Overpass for {_describe_tags(query)} "
                f"{scope}: {result.feature_count} feature(s) returned "
                f"(dataset_ref={dataset_ref})."
            ),
            "These are live map features, not OSM documentation.",
        ]
        if result.feature_count == 0:
            parts.append(
                "No features matched. Do not invent results, coordinates, names "
                "or counts; consider different documented tags or a wider area."
            )
        else:
            parts.append("Do not invent additional features beyond this result.")
        if result.truncated:
            parts.append(f"Results were truncated at the {query.limit} feature limit.")
        if result.warnings:
            parts.append("warnings: " + "; ".join(result.warnings[:5]))
        observation = " ".join(parts)
        if "FeatureCollection" in observation or '"coordinates"' in observation:
            raise ToolExecutionError("internal error: GeoJSON leaked into observation")
        return observation
