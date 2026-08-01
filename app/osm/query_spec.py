"""Validated description of an OSM feature query.

This model is the security boundary between the language model and the Overpass
API: raw model-authored Overpass QL is never executed. The model may only fill
in these fields, which a deterministic builder turns into a query.

The supported subset is intentionally small (the MVP needs area-, radius- and
bbox-scoped tag lookups); it is not a model of the Overpass language.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ElementType = Literal["node", "way", "relation"]

#: Characters that would break out of an Overpass QL string literal. Rejected
#: rather than escaped, so that no crafted value can alter query structure.
FORBIDDEN_LITERAL_CHARS = frozenset('"\\\n\r\t')

MAX_RADIUS_METERS = 50_000
DEFAULT_LIMIT = 200
MAX_LIMIT = 1000

ALL_ELEMENT_TYPES: tuple[ElementType, ...] = ("node", "way", "relation")


def _reject_unsafe(value: str, *, field: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    if FORBIDDEN_LITERAL_CHARS.intersection(text):
        raise ValueError(f"{field} contains characters that are not allowed in an OSM query")
    return text


class TagFilter(BaseModel):
    """One OSM tag condition.

    A missing ``value`` means "the key is present with any value", which is how
    broad categories such as ``amenity`` are expressed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z0-9_:-]*$",
        description="OSM tag key, e.g. 'leisure'.",
    )
    value: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="OSM tag value, e.g. 'park'. Omit to match any value.",
    )

    @model_validator(mode="after")
    def _validate_value(self) -> TagFilter:
        if self.value is not None:
            _reject_unsafe(self.value, field="tag value")
        return self


class PointRadius(BaseModel):
    """A circular search area in metres around a WGS84 point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_m: int = Field(gt=0, le=MAX_RADIUS_METERS)


class BoundingBox(BaseModel):
    """A WGS84 bounding box."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    south: float = Field(ge=-90, le=90)
    west: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)

    @model_validator(mode="after")
    def _validate_ordering(self) -> BoundingBox:
        if self.south >= self.north:
            raise ValueError("south must be less than north")
        if self.west >= self.east:
            raise ValueError("west must be less than east")
        return self


class OsmFeatureQuery(BaseModel):
    """A fully validated OSM feature request.

    Exactly one spatial scope must be given: a named ``place``, a ``point``
    radius, or a ``bbox``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    place: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="Name of an administrative area, e.g. 'Berlin'.",
    )
    point: PointRadius | None = Field(
        default=None,
        description="Circular search around a coordinate.",
    )
    bbox: BoundingBox | None = Field(
        default=None,
        description="Bounding box search.",
    )
    tags: list[TagFilter] = Field(
        min_length=1,
        max_length=8,
        description="Tag conditions combined with AND, e.g. leisure=park.",
    )
    element_types: list[ElementType] = Field(
        default_factory=lambda: list(ALL_ELEMENT_TYPES),
        min_length=1,
        max_length=3,
        description="Which OSM element types to search.",
    )
    include_geometry: bool = Field(
        default=True,
        description="Return full geometry instead of centre points only.",
    )
    limit: int = Field(
        default=DEFAULT_LIMIT,
        gt=0,
        le=MAX_LIMIT,
        description="Maximum number of features to return.",
    )

    @model_validator(mode="after")
    def _validate_scope(self) -> OsmFeatureQuery:
        scopes = [self.place is not None, self.point is not None, self.bbox is not None]
        if sum(scopes) != 1:
            raise ValueError("exactly one of 'place', 'point' or 'bbox' must be provided")
        if self.place is not None:
            _reject_unsafe(self.place, field="place")
        return self

    @property
    def scope_kind(self) -> Literal["place", "point", "bbox"]:
        if self.place is not None:
            return "place"
        if self.point is not None:
            return "point"
        return "bbox"

    @property
    def ordered_element_types(self) -> tuple[ElementType, ...]:
        """Element types in a stable order, so built queries are reproducible."""
        selected = set(self.element_types)
        return tuple(kind for kind in ALL_ELEMENT_TYPES if kind in selected)
