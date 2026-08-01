"""Convert Overpass JSON elements into a GeoJSON FeatureCollection.

Supported reliably:

* nodes with lat/lon → Point
* ways with ``geometry`` → LineString or closed Polygon
* ways/relations with ``center`` → Point

Unsupported relation member geometries are skipped with an explicit warning.
No geometry is invented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.osm.contracts import (
    OSM_ATTRIBUTION,
    GeoJsonConversionResult,
    OverpassElement,
)


def empty_feature_collection() -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": []}


class OverpassGeoJsonEncoder:
    """Deterministic Overpass → GeoJSON converter for the MVP element subset."""

    def encode(self, elements: Sequence[OverpassElement]) -> GeoJsonConversionResult:
        features: list[dict[str, Any]] = []
        warnings: list[str] = []

        for element in elements:
            kind = str(element.get("type") or "")
            osm_id = element.get("id")
            if kind == "node":
                feature = self._node_feature(element)
                if feature is None:
                    warnings.append(f"Skipped node {osm_id!r}: missing coordinates")
                else:
                    features.append(feature)
            elif kind == "way":
                feature, warning = self._way_feature(element)
                if feature is None:
                    warnings.append(warning or f"Skipped way {osm_id!r}: missing geometry")
                else:
                    features.append(feature)
                    if warning:
                        warnings.append(warning)
            elif kind == "relation":
                feature, warning = self._relation_feature(element)
                if feature is None:
                    warnings.append(
                        warning
                        or (f"Skipped relation {osm_id!r}: full relation geometry is not supported")
                    )
                else:
                    features.append(feature)
                    if warning:
                        warnings.append(warning)
            else:
                warnings.append(f"Skipped unsupported Overpass element type {kind!r}")

        return GeoJsonConversionResult(
            feature_collection={
                "type": "FeatureCollection",
                "features": features,
            },
            warnings=tuple(warnings),
        )

    def _properties(self, element: OverpassElement) -> dict[str, Any]:
        tags = element.get("tags")
        if not isinstance(tags, Mapping):
            tags = {}
        return {
            "osm_type": element.get("type"),
            "osm_id": element.get("id"),
            "tags": dict(tags),
            "attribution": OSM_ATTRIBUTION,
        }

    def _node_feature(self, element: OverpassElement) -> dict[str, Any] | None:
        lon = element.get("lon")
        lat = element.get("lat")
        if not isinstance(lon, int | float) or not isinstance(lat, int | float):
            return None
        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "properties": self._properties(element),
        }

    def _way_feature(self, element: OverpassElement) -> tuple[dict[str, Any] | None, str | None]:
        geometry = self._geometry_from_points(element.get("geometry"))
        if geometry is not None:
            return (
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": self._properties(element),
                },
                None,
            )
        center = self._center_point(element.get("center"))
        if center is not None:
            return (
                {
                    "type": "Feature",
                    "geometry": center,
                    "properties": self._properties(element),
                },
                None,
            )
        return None, f"Skipped way {element.get('id')!r}: missing geometry/center"

    def _relation_feature(
        self, element: OverpassElement
    ) -> tuple[dict[str, Any] | None, str | None]:
        # Full multipolygon member expansion is out of MVP scope.
        geometry = self._geometry_from_points(element.get("geometry"))
        if geometry is not None:
            return (
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": self._properties(element),
                },
                f"Relation {element.get('id')!r}: used provided geometry points only; "
                "member relations are not expanded",
            )
        center = self._center_point(element.get("center"))
        if center is not None:
            return (
                {
                    "type": "Feature",
                    "geometry": center,
                    "properties": self._properties(element),
                },
                f"Relation {element.get('id')!r}: only centre point is available; "
                "full relation geometry is not supported",
            )
        return None, (
            f"Skipped relation {element.get('id')!r}: full relation geometry is not supported"
        )

    def _center_point(self, center: Any) -> dict[str, Any] | None:
        if not isinstance(center, Mapping):
            return None
        lon = center.get("lon")
        lat = center.get("lat")
        if not isinstance(lon, int | float) or not isinstance(lat, int | float):
            return None
        return {"type": "Point", "coordinates": [float(lon), float(lat)]}

    def _geometry_from_points(self, points: Any) -> dict[str, Any] | None:
        if not isinstance(points, list) or len(points) < 2:
            return None
        coordinates: list[list[float]] = []
        for point in points:
            if not isinstance(point, Mapping):
                return None
            lon = point.get("lon")
            lat = point.get("lat")
            if not isinstance(lon, int | float) or not isinstance(lat, int | float):
                return None
            coordinates.append([float(lon), float(lat)])
        if len(coordinates) >= 4 and coordinates[0] == coordinates[-1]:
            return {"type": "Polygon", "coordinates": [coordinates]}
        return {"type": "LineString", "coordinates": coordinates}
