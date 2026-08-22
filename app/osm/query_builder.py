"""Deterministic Overpass QL builder.

Pure functions only: the same :class:`OsmFeatureQuery` always produces the same
query string, which makes the generated query reviewable, testable and safe to
show to the user as provenance.
"""

from __future__ import annotations

from app.core.errors import OverpassQueryBuildError
from app.osm.query_spec import (
    FORBIDDEN_LITERAL_CHARS,
    ElementType,
    OsmFeatureQuery,
    TagFilter,
)

_AREA_SET = "searchArea"


def _literal(value: str) -> str:
    """Quote a value that has already been validated as literal-safe."""
    if FORBIDDEN_LITERAL_CHARS.intersection(value):
        raise OverpassQueryBuildError("value contains characters that cannot be quoted safely")
    return f'"{value}"'


def _tag_selector(tag: TagFilter) -> str:
    if tag.value is None:
        return f"[{_literal(tag.key)}]"
    return f"[{_literal(tag.key)}={_literal(tag.value)}]"


def place_area_name(place: str) -> str:
    """Primary toponym used for Overpass area lookup.

    Named places often arrive as ``City, Country``. OSM ``name`` / ``name:en``
    for the administrative area is typically the city alone; using the full
    ``City, Country`` string commonly misses the area and forces expensive
    fallbacks that time out. Scope summaries still keep the full user place.
    """
    primary = place.split(",", 1)[0].strip()
    return primary or place.strip()


def _area_lookup(place: str) -> str:
    """Match ``name`` or ``name:en`` for the primary toponym into ``searchArea``."""
    primary = place_area_name(place)
    quoted = _literal(primary)
    return f'(\n  area["name"={quoted}];\n  area["name:en"={quoted}];\n)->.{_AREA_SET};'


def _spatial_filter(query: OsmFeatureQuery) -> str:
    if query.point is not None:
        point = query.point
        return f"(around:{point.radius_m},{point.lat:g},{point.lon:g})"
    if query.bbox is not None:
        box = query.bbox
        return f"({box.south:g},{box.west:g},{box.north:g},{box.east:g})"
    return f"(area.{_AREA_SET})"


def _statement(kind: ElementType, selectors: str, spatial: str) -> str:
    return f"  {kind}{selectors}{spatial};"


def build_overpass_query(query: OsmFeatureQuery, *, timeout_seconds: int) -> str:
    """Render a validated feature request as Overpass QL.

    ``out geom`` is used when geometry is requested and ``out center``
    otherwise, so way and relation results always carry a usable position.
    The ``out … N`` clause asks Overpass to bound the result set; the tool
    also caps features during GeoJSON normalization as a hard safety limit.
    """
    if timeout_seconds <= 0:
        raise OverpassQueryBuildError("timeout_seconds must be positive")

    spatial = _spatial_filter(query)
    if query.tag_match == "any":
        statements: list[str] = []
        for tag in query.tags:
            selectors = _tag_selector(tag)
            statements.extend(
                _statement(kind, selectors, spatial) for kind in query.ordered_element_types
            )
        body = "\n".join(statements)
    else:
        selectors = "".join(_tag_selector(tag) for tag in query.tags)
        body = "\n".join(
            _statement(kind, selectors, spatial) for kind in query.ordered_element_types
        )

    lines = [f"[out:json][timeout:{timeout_seconds}];"]
    if query.place is not None:
        lines.append(_area_lookup(query.place))
    lines.append("(")
    lines.append(body)
    lines.append(");")
    lines.append(f"out {'geom' if query.include_geometry else 'center'} {query.limit};")
    return "\n".join(lines)
