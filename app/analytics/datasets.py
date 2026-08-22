"""Request-scoped dataset references for analytics (no cross-request leakage)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.analytics.contracts import GeoPoint
from app.analytics.geometry import bbox_area_m2, spherical_cap_area_m2
from app.core.errors import ToolArgumentError
from app.osm.contracts import GeoJsonFeatureCollection

if TYPE_CHECKING:
    from app.osm.query_spec import OsmFeatureQuery
    from app.tools.query_osm import QueryOsmResult


class UnknownDatasetError(ToolArgumentError):
    """Raised when a dataset_ref is not registered in this request."""

    code = "tool_argument_error"


class DatasetScope(BaseModel):
    """Derived spatial scope metadata — never model-supplied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope_kind: Literal["place", "point", "bbox"]
    summary: str
    place: str | None = None
    center: GeoPoint | None = None
    radius_m: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    area_km2: float | None = None


class DatasetRecord(BaseModel):
    """One request-scoped live OSM result available to analyze_features."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_ref: str
    feature_collection: GeoJsonFeatureCollection
    feature_count: int = Field(ge=0)
    resolved_tags: tuple[str, ...]
    scope: DatasetScope
    effective_limit: int = Field(ge=1)
    truncated: bool = False
    retrieved_at: datetime
    endpoint: str


class DatasetRegistry:
    """One instance per agent run. Not shared, not persisted, not global."""

    MAX_DATASETS = 8

    def __init__(self) -> None:
        self._records: dict[str, DatasetRecord] = {}
        self._order: list[str] = []

    def register(
        self,
        result: QueryOsmResult,
        query: OsmFeatureQuery,
    ) -> str:
        """Register a successful query_osm payload; return its dataset_ref."""
        if len(self._order) >= self.MAX_DATASETS:
            raise ToolArgumentError(
                f"at most {self.MAX_DATASETS} datasets may be registered per request"
            )
        dataset_ref = f"osm_result_{len(self._order) + 1}"
        resolved_tags = tuple(
            tag.key if tag.value is None else f"{tag.key}={tag.value}" for tag in query.tags
        )
        scope = _scope_from_query(query, result.scope_summary)
        record = DatasetRecord(
            dataset_ref=dataset_ref,
            feature_collection=result.geojson,
            feature_count=result.feature_count,
            resolved_tags=resolved_tags,
            scope=scope,
            effective_limit=result.effective_limit,
            truncated=result.truncated,
            retrieved_at=result.source.retrieved_at,
            endpoint=result.source.endpoint,
        )
        self._records[dataset_ref] = record
        self._order.append(dataset_ref)
        return dataset_ref

    def get(self, dataset_ref: str) -> DatasetRecord:
        try:
            return self._records[dataset_ref]
        except KeyError as exc:
            known = ", ".join(self._order) or "none"
            raise UnknownDatasetError(
                f"unknown dataset_ref '{dataset_ref}'; available in this request: {known}"
            ) from exc

    def restore(
        self,
        *,
        feature_collection: GeoJsonFeatureCollection,
        feature_count: int,
        resolved_tags: tuple[str, ...],
        scope: DatasetScope,
        effective_limit: int,
        truncated: bool,
        retrieved_at: datetime,
        endpoint: str,
    ) -> str:
        """Register a persisted execution dataset as a new request-scoped ref.

        Never reuses a previous ``osm_result_N`` string as identity.
        """
        if len(self._order) >= self.MAX_DATASETS:
            raise ToolArgumentError(
                f"at most {self.MAX_DATASETS} datasets may be registered per request"
            )
        dataset_ref = f"osm_result_{len(self._order) + 1}"
        record = DatasetRecord(
            dataset_ref=dataset_ref,
            feature_collection=feature_collection,
            feature_count=feature_count,
            resolved_tags=resolved_tags,
            scope=scope,
            effective_limit=effective_limit,
            truncated=truncated,
            retrieved_at=retrieved_at,
            endpoint=endpoint,
        )
        self._records[dataset_ref] = record
        self._order.append(dataset_ref)
        return dataset_ref

    def refs(self) -> tuple[str, ...]:
        return tuple(self._order)

    def describe(self) -> str:
        if not self._order:
            return "datasets=none"
        parts = []
        for ref in self._order:
            record = self._records[ref]
            tags = ",".join(record.resolved_tags)
            parts.append(f"{ref}({record.feature_count} features; {tags})")
        return "datasets=" + "; ".join(parts)


def _scope_from_query(query: OsmFeatureQuery, summary: str) -> DatasetScope:
    if query.place is not None:
        return DatasetScope(
            scope_kind="place",
            summary=summary,
            place=query.place,
            area_km2=None,
        )
    if query.point is not None:
        area_m2 = spherical_cap_area_m2(float(query.point.radius_m))
        return DatasetScope(
            scope_kind="point",
            summary=summary,
            center=GeoPoint(lat=query.point.lat, lon=query.point.lon),
            radius_m=query.point.radius_m,
            area_km2=area_m2 / 1_000_000.0,
        )
    assert query.bbox is not None
    area_m2 = bbox_area_m2(
        query.bbox.south,
        query.bbox.west,
        query.bbox.north,
        query.bbox.east,
    )
    return DatasetScope(
        scope_kind="bbox",
        summary=summary,
        bbox=(
            query.bbox.south,
            query.bbox.west,
            query.bbox.north,
            query.bbox.east,
        ),
        area_km2=area_m2 / 1_000_000.0,
    )
