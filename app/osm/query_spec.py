"""Validated description of an OSM feature query.

This model is the security boundary between the language model and the Overpass
API: raw model-authored Overpass QL is never executed. The model may only fill
in these fields, which a deterministic builder turns into a query.

The supported subset is intentionally small (the MVP needs area-, radius- and
bbox-scoped tag lookups); it is not a model of the Overpass language.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ElementType = Literal["node", "way", "relation"]

#: Characters that would break out of an Overpass QL string literal. Rejected
#: rather than escaped, so that no crafted value can alter query structure.
FORBIDDEN_LITERAL_CHARS = frozenset('"\\\n\r\t')

MAX_RADIUS_METERS = 50_000
DEFAULT_LIMIT = 200
#: Hard ceiling for model-requested feature counts. Must stay aligned with
#: ``OVERPASS_MAX_RESULTS`` (application-controlled). Values above this are
#: rejected by validation; the tool also clamps to the configured server max.
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


class PlaceRefScope(BaseModel):
    """Circular scope around a trusted ``resolve_place`` result.

    The model never supplies latitude/longitude here — only a backend
    ``place_ref`` and the user-requested radius in metres.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    place_ref: str = Field(
        min_length=1,
        max_length=32,
        pattern=r"^place_[1-9][0-9]*$",
        description="Request-scoped place_ref from resolve_place (e.g. place_1).",
    )
    radius_m: int = Field(gt=0, le=MAX_RADIUS_METERS)


class OsmFeatureQuery(BaseModel):
    """A fully validated OSM feature request.

    Exactly one spatial scope must be given: a named ``place``, a ``point``
    radius, a ``bbox``, or a trusted ``place_ref_scope``.
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
    place_ref_scope: PlaceRefScope | None = Field(
        default=None,
        description=(
            "Circular search around a trusted resolve_place result. "
            "Use for landmarks; never invent coordinates."
        ),
    )
    tags: list[TagFilter] = Field(
        min_length=1,
        max_length=8,
        description=(
            "Tag conditions. Combined with AND when tag_match=all (default), or "
            "as a union when tag_match=any."
        ),
    )
    tag_match: Literal["all", "any"] = Field(
        default="all",
        description=(
            "How multiple tags combine. 'all' (default) requires every tag on the "
            "same element (AND). 'any' unions alternative tags (OR), used for "
            "green-space categories such as leisure=park or landuse=grass."
        ),
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

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tag_pairs(cls, value: Any) -> Any:
        """Accept ``[["leisure","park"]]`` as well as ``[{key,value}]`` objects."""
        if not isinstance(value, list):
            return value
        coerced: list[Any] = []
        for item in value:
            if isinstance(item, TagFilter):
                coerced.append(item)
                continue
            if isinstance(item, dict):
                coerced.append(item)
                continue
            if isinstance(item, list | tuple) and len(item) == 2:
                key, tag_value = item[0], item[1]
                coerced.append({"key": key, "value": tag_value})
                continue
            raise ValueError(
                "each tag must be an object {key,value} or a two-item [key,value] pair"
            )
        return coerced

    @field_validator("place", mode="before")
    @classmethod
    def _reject_place_list(cls, value: Any) -> Any:
        if isinstance(value, list):
            raise ValueError(
                "place must be a single named area string; for multi-target "
                "comparison call query_osm once per target with place_ref_scope"
            )
        return value

    @model_validator(mode="after")
    def _validate_scope(self) -> OsmFeatureQuery:
        scopes = [
            self.place is not None,
            self.point is not None,
            self.bbox is not None,
            self.place_ref_scope is not None,
        ]
        if sum(scopes) != 1:
            raise ValueError(
                "exactly one of 'place', 'point', 'bbox' or 'place_ref_scope' must be provided"
            )
        if self.place is not None:
            _reject_unsafe(self.place, field="place")
        return self

    @property
    def scope_kind(self) -> Literal["place", "point", "bbox", "place_ref"]:
        if self.place is not None:
            return "place"
        if self.point is not None:
            return "point"
        if self.place_ref_scope is not None:
            return "place_ref"
        return "bbox"

    @property
    def ordered_element_types(self) -> tuple[ElementType, ...]:
        """Element types in a stable order, so built queries are reproducible."""
        selected = set(self.element_types)
        return tuple(kind for kind in ALL_ELEMENT_TYPES if kind in selected)
