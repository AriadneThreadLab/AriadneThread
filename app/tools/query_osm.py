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

from app.core.errors import (
    OverpassError,
    OverpassQueryBuildError,
    ToolExecutionError,
    ToolTimeoutError,
)
from app.osm.contracts import (
    OSM_ATTRIBUTION,
    GeoJsonFeatureCollection,
    OsmGeoJsonEncoder,
    OverpassClient,
)
from app.osm.query_builder import build_overpass_query
from app.osm.query_spec import OsmFeatureQuery
from app.tools.contracts import ToolOutcome

TOOL_NAME = "query_osm"
TOOL_DESCRIPTION = (
    "Retrieve real OpenStreetMap features through the Overpass API. Provide a "
    "spatial scope (place name, point with radius, or bounding box) and the OSM "
    "tags to match. This is the only source of live map data. Returns live OSM "
    "features, NOT documentation."
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


def _describe_scope(query: OsmFeatureQuery) -> str:
    if query.place is not None:
        return f"in {query.place}"
    if query.point is not None:
        return f"within {query.point.radius_m} m of {query.point.lat:g},{query.point.lon:g}"
    box = query.bbox
    assert box is not None
    return f"in bbox {box.south:g},{box.west:g},{box.north:g},{box.east:g}"


def _describe_tags(query: OsmFeatureQuery) -> str:
    return ", ".join(
        tag.key if tag.value is None else f"{tag.key}={tag.value}" for tag in query.tags
    )


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

    async def execute(self, args: OsmFeatureQuery) -> ToolOutcome[QueryOsmResult]:
        effective = self._apply_server_limit(args)
        try:
            query = build_overpass_query(effective, timeout_seconds=self._timeout_seconds)
        except OverpassQueryBuildError as exc:
            raise ToolExecutionError(f"could not build Overpass query: {exc.message}") from exc

        try:
            response = await self._client.run(query)
        except ToolTimeoutError as exc:
            raise ToolExecutionError(exc.message) from exc
        except OverpassError as exc:
            raise ToolExecutionError(exc.message) from exc

        conversion = self._encoder.encode(response.elements)
        raw_features = conversion.feature_collection.get("features", [])
        features: list[Any] = list(raw_features) if isinstance(raw_features, list) else []
        warnings = list(response.warnings) + list(conversion.warnings)
        truncated = response.truncated
        if len(features) > effective.limit:
            features = features[: effective.limit]
            truncated = True
            warnings.append(f"Feature list truncated to the configured limit of {effective.limit}")

        geojson: GeoJsonFeatureCollection = {
            "type": "FeatureCollection",
            "features": features,
        }
        result = QueryOsmResult(
            feature_count=len(features),
            geojson=geojson,
            overpass_query=query,
            source=OsmQuerySource(
                endpoint=response.endpoint,
                retrieved_at=response.retrieved_at,
            ),
            warnings=warnings,
            truncated=truncated,
        )
        return ToolOutcome(observation=self._observation(effective, result), payload=result)

    def _apply_server_limit(self, args: OsmFeatureQuery) -> OsmFeatureQuery:
        """Clamp the model-requested limit to the configured server maximum."""
        if args.limit <= self._max_results:
            return args
        return args.model_copy(update={"limit": self._max_results})

    def _observation(self, query: OsmFeatureQuery, result: QueryOsmResult) -> str:
        """Compact agent-facing summary. Must not include GeoJSON."""
        status = "empty" if result.feature_count == 0 else "ok"
        if result.feature_count > 0 and result.warnings:
            status = "ok_with_warnings"
        parts = [
            (
                f"status={status} source=live_osm "
                f"feature_count={result.feature_count} "
                f"warnings={len(result.warnings)} "
                f"query={_describe_tags(query)} {_describe_scope(query)}"
            ),
            (
                f"Queried live OpenStreetMap via Overpass for {_describe_tags(query)} "
                f"{_describe_scope(query)}: {result.feature_count} feature(s) returned."
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
