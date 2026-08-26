"""Spherical geodesy helpers — no degree-unit area/distance maths."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

# IUGG mean Earth radius (metres), WGS84.
EARTH_RADIUS_M = 6_371_008.8

GEOMETRY_STRATEGY_HAVERSINE = "haversine_sphere_r6371008.8"
GEOMETRY_STRATEGY_SPHERICAL_AREA = "spherical_excess_area"
GEOMETRY_STRATEGY_CAP = "spherical_cap_area"
GEOMETRY_STRATEGY_BBOX = "spherical_bbox_area"
GEOMETRY_STRATEGY_CENTROID = "polygon_ring_centroid"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def spherical_cap_area_m2(radius_m: float) -> float:
    """Area of a spherical cap of geodesic radius ``radius_m``."""
    if radius_m <= 0:
        return 0.0
    angular = radius_m / EARTH_RADIUS_M
    return 2.0 * math.pi * EARTH_RADIUS_M**2 * (1.0 - math.cos(angular))


def bbox_area_m2(south: float, west: float, north: float, east: float) -> float:
    """Area of a WGS84 bounding box on the mean sphere (m²)."""
    if south >= north or west >= east:
        raise ValueError("invalid bounding box ordering")
    phi_n = math.radians(north)
    phi_s = math.radians(south)
    d_lambda = math.radians(east - west)
    return EARTH_RADIUS_M**2 * abs(d_lambda) * abs(math.sin(phi_n) - math.sin(phi_s))


def _normalize_delta_lon(delta: float) -> float:
    """Normalise a longitude difference into [-π, π]."""
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta < -math.pi:
        delta += 2 * math.pi
    return delta


def spherical_polygon_area_m2(ring: list[list[float]]) -> float | None:
    """Spherical excess area of a closed outer ring (lon, lat) in m².

    Returns ``None`` for degenerate or antimeridian-spanning rings.
    """
    if len(ring) < 4:
        return None
    if ring[0] != ring[-1]:
        return None

    lons = [p[0] for p in ring[:-1]]
    if max(lons) - min(lons) > 180.0:
        return None

    total = 0.0
    n = len(ring) - 1
    for i in range(n):
        lon1, lat1 = ring[i]
        lon2, lat2 = ring[(i + 1) % n]
        lambda1 = math.radians(lon1)
        lambda2 = math.radians(lon2)
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_lambda = _normalize_delta_lon(lambda2 - lambda1)
        total += d_lambda * (math.sin(phi1) + math.sin(phi2))
    return abs(EARTH_RADIUS_M**2 * total / 2.0)


def representative_point(geometry: dict[str, Any]) -> tuple[float, float] | None:
    """Return (lat, lon) representative point, or None if unsupported."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    if kind == "Point":
        if (
            isinstance(coords, list)
            and len(coords) >= 2
            and isinstance(coords[0], int | float)
            and isinstance(coords[1], int | float)
        ):
            return float(coords[1]), float(coords[0])
        return None
    if kind == "LineString":
        if not isinstance(coords, list) or len(coords) < 2:
            return None
        lons: list[float] = []
        lats: list[float] = []
        for point in coords:
            if (
                not isinstance(point, list)
                or len(point) < 2
                or not isinstance(point[0], int | float)
                or not isinstance(point[1], int | float)
            ):
                return None
            lons.append(float(point[0]))
            lats.append(float(point[1]))
        return sum(lats) / len(lats), sum(lons) / len(lons)
    if kind == "Polygon":
        if not isinstance(coords, list) or not coords:
            return None
        ring = coords[0]
        if not isinstance(ring, list) or len(ring) < 4:
            return None
        return _ring_centroid(ring)
    return None


def _ring_centroid(ring: list[Any]) -> tuple[float, float] | None:
    """Area-weighted centroid in a local equirectangular frame; fallback mean."""
    points: list[tuple[float, float]] = []
    for point in ring:
        if (
            not isinstance(point, list)
            or len(point) < 2
            or not isinstance(point[0], int | float)
            or not isinstance(point[1], int | float)
        ):
            return None
        points.append((float(point[0]), float(point[1])))
    if points[0] != points[-1]:
        return None
    verts = points[:-1]
    if len(verts) < 3:
        return None
    mean_lat = sum(lat for _, lat in verts) / len(verts)
    mean_lon = sum(lon for lon, _ in verts) / len(verts)
    cos_lat = math.cos(math.radians(mean_lat)) or 1e-12
    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(verts)):
        x0 = (verts[i][0] - mean_lon) * cos_lat
        y0 = verts[i][1] - mean_lat
        x1 = (verts[(i + 1) % len(verts)][0] - mean_lon) * cos_lat
        y1 = verts[(i + 1) % len(verts)][1] - mean_lat
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area) < 1e-18:
        return mean_lat, mean_lon
    cx /= 3.0 * area
    cy /= 3.0 * area
    lon = mean_lon + cx / cos_lat
    lat = mean_lat + cy
    return lat, lon


def polygon_area_from_geometry(geometry: dict[str, Any]) -> float | None:
    """Area in m² for a Polygon geometry, or None if invalid.

    Outer-ring only, matching the original Metric Catalog extractor. Indicator
    calculations that need holes or MultiPolygon use :func:`geometry_area_m2`.
    """
    if geometry.get("type") != "Polygon":
        return None
    rings = _polygon_rings(geometry.get("coordinates"))
    if rings is None:
        return None
    return spherical_polygon_area_m2(rings[0])


GEOMETRY_STRATEGY_GEODESIC_LENGTH = "haversine_polyline_r6371008.8"
GEOMETRY_STRATEGY_NEAREST_GEODESIC = "haversine_nearest_on_geometry_r6371008.8"
GEOMETRY_STRATEGY_SHANNON = "shannon_entropy_category_counts"
GEOMETRY_STRATEGY_NETWORK = "degree_ge3_snapped_vertices_1m"

_CLIP_EPS_M = 1e-6


def _lonlat_point(raw: Any) -> list[float] | None:
    if (
        not isinstance(raw, list)
        or len(raw) < 2
        or not isinstance(raw[0], int | float)
        or not isinstance(raw[1], int | float)
    ):
        return None
    return [float(raw[0]), float(raw[1])]


def _polyline_coords(raw: Any) -> list[list[float]] | None:
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    coords: list[list[float]] = []
    for item in raw:
        point = _lonlat_point(item)
        if point is None:
            return None
        coords.append(point)
    return coords


def _polygon_rings(raw: Any) -> list[list[list[float]]] | None:
    if not isinstance(raw, list) or not raw:
        return None
    rings: list[list[list[float]]] = []
    for ring_raw in raw:
        ring = _polyline_coords(ring_raw)
        if ring is None or len(ring) < 4:
            return None
        rings.append(ring)
    return rings


def iter_polylines(geometry: dict[str, Any]) -> list[list[list[float]]]:
    """Return vertex polylines for line or polygon-ring geometries."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    if kind == "LineString":
        line = _polyline_coords(coords)
        return [line] if line is not None else []
    if kind == "MultiLineString" and isinstance(coords, list):
        lines: list[list[list[float]]] = []
        for part in coords:
            line = _polyline_coords(part)
            if line is not None:
                lines.append(line)
        return lines
    if kind == "Polygon":
        rings = _polygon_rings(coords)
        return [rings[0]] if rings is not None else []
    if kind == "MultiPolygon" and isinstance(coords, list):
        lines = []
        for polygon in coords:
            rings = _polygon_rings(polygon)
            if rings is not None:
                lines.append(rings[0])
        return lines
    return []


def polyline_length_m(coords: list[list[float]]) -> float | None:
    """Great-circle length of a polyline (lon, lat) in metres."""
    if len(coords) < 2:
        return None
    total = 0.0
    for start, end in pairwise(coords):
        total += haversine_m(start[1], start[0], end[1], end[0])
    return total


def geometry_length_m(geometry: dict[str, Any]) -> float | None:
    """Geodesic length in metres for LineString, MultiLineString, or Polygon ring."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    if kind == "LineString":
        line = _polyline_coords(coords)
        return None if line is None else polyline_length_m(line)
    if kind == "MultiLineString":
        if not isinstance(coords, list) or not coords:
            return None
        total = 0.0
        any_valid = False
        for part in coords:
            line = _polyline_coords(part)
            if line is None:
                continue
            length = polyline_length_m(line)
            if length is None:
                continue
            total += length
            any_valid = True
        return total if any_valid else None
    if kind == "Polygon":
        rings = _polygon_rings(coords)
        if rings is None:
            return None
        return polyline_length_m(rings[0])
    if kind == "MultiPolygon":
        if not isinstance(coords, list) or not coords:
            return None
        total = 0.0
        any_valid = False
        for polygon in coords:
            rings = _polygon_rings(polygon)
            if rings is None:
                continue
            length = polyline_length_m(rings[0])
            if length is None:
                continue
            total += length
            any_valid = True
        return total if any_valid else None
    return None


def geometry_area_m2(geometry: dict[str, Any]) -> float | None:
    """Spherical area in m² for Polygon or MultiPolygon, holes subtracted."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    if kind == "Polygon":
        return _polygon_area_with_holes(coords)
    if kind == "MultiPolygon":
        if not isinstance(coords, list) or not coords:
            return None
        total = 0.0
        any_valid = False
        for polygon in coords:
            area = _polygon_area_with_holes(polygon)
            if area is None:
                continue
            total += area
            any_valid = True
        return total if any_valid else None
    return None


def _polygon_area_with_holes(raw: Any) -> float | None:
    rings = _polygon_rings(raw)
    if rings is None:
        return None
    outer = spherical_polygon_area_m2(rings[0])
    if outer is None:
        return None
    holes = 0.0
    for ring in rings[1:]:
        hole = spherical_polygon_area_m2(ring)
        if hole is None:
            return None
        holes += hole
    return max(0.0, outer - holes)


def nearest_distance_m(lat: float, lon: float, geometry: dict[str, Any]) -> float | None:
    """Geodesic distance from a WGS84 point to the nearest location on ``geometry``."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    if kind == "Point":
        point = _lonlat_point(coords)
        if point is None:
            return None
        return haversine_m(lat, lon, point[1], point[0])
    if kind == "MultiPoint":
        if not isinstance(coords, list) or not coords:
            return None
        distances = []
        for item in coords:
            point = _lonlat_point(item)
            if point is None:
                continue
            distances.append(haversine_m(lat, lon, point[1], point[0]))
        return min(distances) if distances else None
    if kind == "LineString":
        line = _polyline_coords(coords)
        return None if line is None else _nearest_to_polyline_m(lat, lon, line)
    if kind == "MultiLineString":
        if not isinstance(coords, list) or not coords:
            return None
        distances = []
        for part in coords:
            line = _polyline_coords(part)
            if line is None:
                continue
            distance = _nearest_to_polyline_m(lat, lon, line)
            if distance is not None:
                distances.append(distance)
        return min(distances) if distances else None
    if kind == "Polygon":
        rings = _polygon_rings(coords)
        if rings is None:
            return None
        return _nearest_to_polyline_m(lat, lon, rings[0])
    if kind == "MultiPolygon":
        if not isinstance(coords, list) or not coords:
            return None
        distances = []
        for polygon in coords:
            rings = _polygon_rings(polygon)
            if rings is None:
                continue
            distance = _nearest_to_polyline_m(lat, lon, rings[0])
            if distance is not None:
                distances.append(distance)
        return min(distances) if distances else None
    return None


def _nearest_to_polyline_m(lat: float, lon: float, coords: list[list[float]]) -> float | None:
    if len(coords) == 1:
        return haversine_m(lat, lon, coords[0][1], coords[0][0])
    if len(coords) < 1:
        return None
    best: float | None = None
    for start, end in pairwise(coords):
        distance = _nearest_to_segment_m(lat, lon, start, end)
        if best is None or distance < best:
            best = distance
    return best


def _nearest_to_segment_m(lat: float, lon: float, start: list[float], end: list[float]) -> float:
    """Nearest distance to a segment via local equirectangular metres, then haversine."""
    lat0 = math.radians(lat)
    cos_lat = math.cos(lat0) or 1e-12
    scale = EARTH_RADIUS_M

    def to_xy(point: list[float]) -> tuple[float, float]:
        x = math.radians(point[0] - lon) * cos_lat * scale
        y = math.radians(point[1] - lat) * scale
        return x, y

    ax, ay = to_xy(start)
    bx, by = to_xy(end)
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= _CLIP_EPS_M:
        return haversine_m(lat, lon, start[1], start[0])
    t = max(0.0, min(1.0, (-ax * dx - ay * dy) / denom))
    nearest_lon = start[0] + t * (end[0] - start[0])
    nearest_lat = start[1] + t * (end[1] - start[1])
    return haversine_m(lat, lon, nearest_lat, nearest_lon)


def clip_polyline_to_cap(
    coords: list[list[float]],
    center_lat: float,
    center_lon: float,
    radius_m: float,
) -> list[list[list[float]]]:
    """Clip a polyline to a geodesic radius, returning zero or more inside parts."""
    if len(coords) < 2 or radius_m <= 0:
        return []
    parts: list[list[list[float]]] = []
    current: list[list[float]] = []
    for start, end in pairwise(coords):
        clipped = _clip_segment_to_cap(start, end, center_lat, center_lon, radius_m)
        if not clipped:
            if len(current) >= 2:
                parts.append(current)
            current = []
            continue
        if not current:
            current = [clipped[0], clipped[1]]
            continue
        if current[-1] != clipped[0]:
            if len(current) >= 2:
                parts.append(current)
            current = [clipped[0], clipped[1]]
        else:
            current.append(clipped[1])
    if len(current) >= 2:
        parts.append(current)
    return parts


def _clip_segment_to_cap(
    start: list[float],
    end: list[float],
    center_lat: float,
    center_lon: float,
    radius_m: float,
) -> list[list[float]] | None:
    lat0 = math.radians(center_lat)
    cos_lat = math.cos(lat0) or 1e-12
    scale = EARTH_RADIUS_M

    def to_xy(point: list[float]) -> tuple[float, float]:
        x = math.radians(point[0] - center_lon) * cos_lat * scale
        y = math.radians(point[1] - center_lat) * scale
        return x, y

    def from_xy(x: float, y: float) -> list[float]:
        lon = center_lon + math.degrees(x / (cos_lat * scale))
        lat = center_lat + math.degrees(y / scale)
        return [lon, lat]

    ax, ay = to_xy(start)
    bx, by = to_xy(end)
    d0 = math.hypot(ax, ay)
    d1 = math.hypot(bx, by)
    inside0 = d0 <= radius_m + _CLIP_EPS_M
    inside1 = d1 <= radius_m + _CLIP_EPS_M
    if inside0 and inside1:
        return [start, end]

    dx = bx - ax
    dy = by - ay
    qa = dx * dx + dy * dy
    qb = 2.0 * (ax * dx + ay * dy)
    qc = ax * ax + ay * ay - radius_m * radius_m
    roots: list[float] = []
    if qa > _CLIP_EPS_M:
        disc = qb * qb - 4.0 * qa * qc
        if disc >= 0.0:
            sqrt_disc = math.sqrt(disc)
            for raw in ((-qb - sqrt_disc) / (2.0 * qa), (-qb + sqrt_disc) / (2.0 * qa)):
                if 0.0 - 1e-9 <= raw <= 1.0 + 1e-9:
                    roots.append(min(1.0, max(0.0, raw)))
    roots = sorted({round(item, 9) for item in roots})
    hits = [from_xy(ax + t * dx, ay + t * dy) for t in roots]
    if inside0 and not inside1:
        return [start, hits[-1]] if hits else None
    if inside1 and not inside0:
        return [hits[0], end] if hits else None
    if len(hits) >= 2:
        return [hits[0], hits[-1]]
    return None


def clip_polyline_to_bbox(
    coords: list[list[float]],
    south: float,
    west: float,
    north: float,
    east: float,
) -> list[list[list[float]]]:
    """Clip a polyline to a WGS84 box; remaining length is still geodesic."""
    if len(coords) < 2 or south >= north or west >= east:
        return []
    parts: list[list[list[float]]] = []
    current: list[list[float]] = []
    for start, end in pairwise(coords):
        clipped = _clip_segment_to_bbox(start, end, south, west, north, east)
        if not clipped:
            if len(current) >= 2:
                parts.append(current)
            current = []
            continue
        if not current:
            current = [clipped[0], clipped[1]]
            continue
        if current[-1] != clipped[0]:
            if len(current) >= 2:
                parts.append(current)
            current = [clipped[0], clipped[1]]
        else:
            current.append(clipped[1])
    if len(current) >= 2:
        parts.append(current)
    return parts


def _inside_bbox(point: list[float], south: float, west: float, north: float, east: float) -> bool:
    lon, lat = point
    return west <= lon <= east and south <= lat <= north


def _clip_segment_to_bbox(
    start: list[float],
    end: list[float],
    south: float,
    west: float,
    north: float,
    east: float,
) -> list[list[float]] | None:
    inside0 = _inside_bbox(start, south, west, north, east)
    inside1 = _inside_bbox(end, south, west, north, east)
    if inside0 and inside1:
        return [start, end]
    hits: list[list[float]] = []
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) > 1e-18:
        for edge_lon in (west, east):
            t = (edge_lon - start[0]) / dx
            if 0.0 <= t <= 1.0:
                lat = start[1] + t * dy
                if south <= lat <= north:
                    hits.append([edge_lon, lat])
    if abs(dy) > 1e-18:
        for edge_lat in (south, north):
            t = (edge_lat - start[1]) / dy
            if 0.0 <= t <= 1.0:
                lon = start[0] + t * dx
                if west <= lon <= east:
                    hits.append([lon, edge_lat])
    if inside0 and hits:
        return [start, hits[0]]
    if inside1 and hits:
        return [hits[0], end]
    if len(hits) >= 2:
        hits.sort(key=lambda point: (point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2)
        return [hits[0], hits[-1]]
    return None


def clipped_geometry_length_m(
    geometry: dict[str, Any],
    *,
    center: tuple[float, float] | None = None,
    radius_m: float | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> tuple[float | None, bool]:
    """Geodesic length after optional cap/bbox clip. ``clipped`` is True when applied."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    lines: list[list[list[float]]] = []
    if kind == "LineString":
        line = _polyline_coords(coords)
        if line is not None:
            lines.append(line)
    elif kind == "MultiLineString" and isinstance(coords, list):
        for part in coords:
            line = _polyline_coords(part)
            if line is not None:
                lines.append(line)
    elif kind == "Polygon":
        rings = _polygon_rings(coords)
        if rings is not None:
            lines.append(rings[0])
    elif kind == "MultiPolygon" and isinstance(coords, list):
        for polygon in coords:
            rings = _polygon_rings(polygon)
            if rings is not None:
                lines.append(rings[0])
    else:
        return None, False

    clipped = False
    parts = lines
    if center is not None and radius_m is not None:
        clipped = True
        parts = []
        for line in lines:
            parts.extend(clip_polyline_to_cap(line, center[0], center[1], radius_m))
    elif bbox is not None:
        clipped = True
        parts = []
        south, west, north, east = bbox
        for line in lines:
            parts.extend(clip_polyline_to_bbox(line, south, west, north, east))

    if not parts:
        return (0.0, True) if clipped else (None, False)
    total = 0.0
    for part in parts:
        length = polyline_length_m(part)
        if length is None:
            continue
        total += length
    return total, clipped
