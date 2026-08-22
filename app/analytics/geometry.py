"""Spherical geodesy helpers — no degree-unit area/distance maths."""

from __future__ import annotations

import math
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
    """Area in m² for a Polygon geometry, or None if invalid."""
    if geometry.get("type") != "Polygon":
        return None
    coords = geometry.get("coordinates")
    if not isinstance(coords, list) or not coords:
        return None
    ring = coords[0]
    if not isinstance(ring, list):
        return None
    normalised: list[list[float]] = []
    for point in ring:
        if (
            not isinstance(point, list)
            or len(point) < 2
            or not isinstance(point[0], int | float)
            or not isinstance(point[1], int | float)
        ):
            return None
        normalised.append([float(point[0]), float(point[1])])
    return spherical_polygon_area_m2(normalised)
