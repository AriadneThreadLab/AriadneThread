"""Spherical geometry helpers."""

from __future__ import annotations

import math

from app.analytics.geometry import (
    EARTH_RADIUS_M,
    bbox_area_m2,
    haversine_m,
    representative_point,
    spherical_cap_area_m2,
    spherical_polygon_area_m2,
)


def test_haversine_city_scale() -> None:
    # Rough Tehran Azadi to Valiasr (~6-8 km order of magnitude)
    d = haversine_m(35.6997, 51.3380, 35.7219, 51.4050)
    assert 5_000 < d < 12_000


def test_cap_area_matches_small_circle_approximation() -> None:
    r = 1000.0
    area = spherical_cap_area_m2(r)
    approx = math.pi * r * r
    assert abs(area - approx) / approx < 0.001


def test_bbox_area_positive() -> None:
    area = bbox_area_m2(35.6, 51.3, 35.8, 51.5)
    assert area > 0
    assert area / 1_000_000 < 500  # km² sanity


def test_unit_square_near_equator_area() -> None:
    ring = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]
    area = spherical_polygon_area_m2(ring)
    assert area is not None
    # Analytic spherical quad for 1°x1° at equator ≈ R² * Δλ * (sinφn - sinφs)
    expected = bbox_area_m2(0.0, 0.0, 1.0, 1.0)
    assert abs(area - expected) / expected < 0.02


def test_antimeridian_polygon_rejected() -> None:
    ring = [
        [170.0, 0.0],
        [-170.0, 0.0],
        [-170.0, 1.0],
        [170.0, 1.0],
        [170.0, 0.0],
    ]
    assert spherical_polygon_area_m2(ring) is None


def test_representative_point_point_and_polygon() -> None:
    assert representative_point({"type": "Point", "coordinates": [51.4, 35.7]}) == (
        35.7,
        51.4,
    )
    ring = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]
    lat, lon = representative_point({"type": "Polygon", "coordinates": [ring]})  # type: ignore[misc]
    assert abs(lat - 1.0) < 0.2
    assert abs(lon - 1.0) < 0.2


def test_earth_radius_constant() -> None:
    assert EARTH_RADIUS_M == 6_371_008.8
