"""``resolve_place`` — trusted landmark geocoding for point-radius OSM scopes.

The model supplies only a human-readable place query. Coordinates come from a
configured geocoder behind Tool Registry — never from LLM memory.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.core.errors import PlaceAmbiguousError, PlaceResolutionError, ToolExecutionError
from app.places.contracts import PlaceResolver
from app.places.selection import select_trusted_hit, short_label
from app.tools.context import ToolContext
from app.tools.contracts import ToolOutcome

TOOL_NAME = "resolve_place"
TOOL_DESCRIPTION = (
    "Resolve a named landmark to a trusted reference location for later "
    "query_osm place_ref_scope calls. Arguments: required query (string), "
    "optional limit (1-5). Returns place_ref — never invent coordinates."
)


class PlaceResolutionRequest(BaseModel):
    """Model-facing place-resolution arguments (no coordinates)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(
        min_length=2,
        max_length=200,
        description="Human-readable place, e.g. 'University of Tehran, Tehran, Iran'.",
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Max geocoder candidates to consider (provider ranking).",
    )
    label: str | None = Field(
        default=None,
        min_length=2,
        max_length=120,
        description="Optional display label; defaults to the query head.",
    )


class PlaceResolutionResult(BaseModel):
    """Compact trusted place result for API/UI (not a full geocoder dump)."""

    model_config = ConfigDict(frozen=True)

    place_ref: str
    label: str
    display_name: str
    latitude: float
    longitude: float
    source: str
    source_id: str
    status: str = "resolved"
    query: str


class ResolvePlaceTool:
    """Tool implementation over a :class:`PlaceResolver`."""

    def __init__(self, resolver: PlaceResolver) -> None:
        self._resolver = resolver

    @property
    def name(self) -> str:
        return TOOL_NAME

    @property
    def description(self) -> str:
        return TOOL_DESCRIPTION

    @property
    def args_model(self) -> type[PlaceResolutionRequest]:
        return PlaceResolutionRequest

    async def aclose(self) -> None:
        await self._resolver.aclose()

    async def execute(
        self,
        args: PlaceResolutionRequest,
        context: ToolContext,
    ) -> ToolOutcome[PlaceResolutionResult]:
        try:
            hits = await self._resolver.search(args.query, limit=args.limit)
            hit = select_trusted_hit(args.query, hits)
        except PlaceAmbiguousError:
            raise
        except PlaceResolutionError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ToolExecutionError(f"place resolution failed: {exc}") from exc

        label = (
            args.label.strip()
            if isinstance(args.label, str) and args.label.strip()
            else short_label(args.query, hit.display_name)
        )
        try:
            record = context.places.register(
                query=args.query.strip(),
                label=label,
                display_name=hit.display_name,
                latitude=hit.latitude,
                longitude=hit.longitude,
                source=self._resolver.source_name,
                source_id=hit.source_id,
            )
        except ValueError as exc:
            raise PlaceResolutionError(str(exc)) from exc

        result = PlaceResolutionResult(
            place_ref=record.place_ref,
            label=record.label,
            display_name=record.display_name,
            latitude=record.latitude,
            longitude=record.longitude,
            source=record.source,
            source_id=record.source_id,
            query=record.query,
        )
        observation = (
            f"status=resolved source={result.source} place_ref={result.place_ref} "
            f"label={result.label} "
            f"display_name={result.display_name} "
            f"latitude={result.latitude:.6f} longitude={result.longitude:.6f}. "
            "Use place_ref_scope with this place_ref and the user-requested "
            "radius_m in query_osm. Do not invent different coordinates."
        )
        return ToolOutcome(observation=observation, payload=result)
