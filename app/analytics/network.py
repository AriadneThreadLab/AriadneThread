"""Road-network topology for intersection density.

Intersections are graph vertices of degree ≥ 3. Polyline bends (degree 2) and
dead-ends (degree 1) are not intersections. Coincident coordinates are snapped
to a 1 m cell when OSM node ids are unavailable.

Geometric crossings without a shared vertex are not counted: in OSM they are
not topological junctions (bridges, tunnels, or unsplit mapping).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from app.analytics.datasets import DatasetScope
from app.analytics.geometry import (
    EARTH_RADIUS_M,
    clip_polyline_to_bbox,
    clip_polyline_to_cap,
    iter_polylines,
)

SNAP_CELL_M = 1.0
MIN_INTERSECTION_DEGREE = 3

VertexKey = tuple[int, int] | int


@dataclass(frozen=True, slots=True)
class IntersectionCensus:
    """Topological intersection count for clipped highway geometries."""

    intersection_count: int
    vertex_count: int
    way_count: int
    snap_cell_m: float
    used_osm_node_ids: bool
    notes: tuple[str, ...]


def count_network_intersections(
    features: list[dict[str, Any]],
    scope: DatasetScope,
) -> IntersectionCensus:
    """Count degree ≥ 3 vertices in the snapped, clipped road graph."""
    will_clip = _scope_clips(scope)
    prepared: list[tuple[list[list[list[float]]], list[list[int]] | None]] = []
    all_have_nodes = True
    way_count = 0

    for feature in features:
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        polylines = iter_polylines(geometry)
        if not polylines:
            continue
        way_count += 1
        node_ids = None if will_clip else _node_ids_per_line(feature, polylines)
        if node_ids is None:
            all_have_nodes = False
        prepared.append((_clip_polylines(polylines, scope), node_ids))

    if way_count == 0:
        return IntersectionCensus(
            intersection_count=0,
            vertex_count=0,
            way_count=0,
            snap_cell_m=SNAP_CELL_M,
            used_osm_node_ids=False,
            notes=("no highway geometries to build a network from",),
        )

    use_nodes = (not will_clip) and all_have_nodes
    origin = _origin(scope, features)
    adjacency: dict[VertexKey, set[VertexKey]] = {}
    for clipped, node_ids in prepared:
        for index, line in enumerate(clipped):
            ids = node_ids[index] if use_nodes and node_ids is not None else None
            keys = _vertex_keys(line, origin, ids)
            for start, end in pairwise(keys):
                if start == end:
                    continue
                adjacency.setdefault(start, set()).add(end)
                adjacency.setdefault(end, set()).add(start)

    intersections = sum(
        1 for neighbours in adjacency.values() if len(neighbours) >= MIN_INTERSECTION_DEGREE
    )
    if use_nodes:
        notes = (f"intersections are OSM nodes with degree ≥ {MIN_INTERSECTION_DEGREE}",)
    else:
        notes = (
            f"intersections are vertices with degree ≥ {MIN_INTERSECTION_DEGREE}; "
            f"coordinates snapped to {SNAP_CELL_M:g} m cells",
        )
    return IntersectionCensus(
        intersection_count=intersections,
        vertex_count=len(adjacency),
        way_count=way_count,
        snap_cell_m=SNAP_CELL_M,
        used_osm_node_ids=use_nodes,
        notes=notes,
    )


def _scope_clips(scope: DatasetScope) -> bool:
    if scope.scope_kind == "point" and scope.center is not None and scope.radius_m is not None:
        return True
    return scope.scope_kind == "bbox" and scope.bbox is not None


def _origin(scope: DatasetScope, features: list[dict[str, Any]]) -> tuple[float, float]:
    if scope.center is not None:
        return scope.center.lat, scope.center.lon
    if scope.bbox is not None:
        south, west, north, east = scope.bbox
        return (south + north) / 2.0, (west + east) / 2.0
    for feature in features:
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        for line in iter_polylines(geometry):
            if line:
                return line[0][1], line[0][0]
    return 0.0, 0.0


def _clip_polylines(
    polylines: list[list[list[float]]], scope: DatasetScope
) -> list[list[list[float]]]:
    if scope.scope_kind == "point" and scope.center is not None and scope.radius_m is not None:
        clipped: list[list[list[float]]] = []
        for line in polylines:
            clipped.extend(
                clip_polyline_to_cap(
                    line, scope.center.lat, scope.center.lon, float(scope.radius_m)
                )
            )
        return clipped
    if scope.scope_kind == "bbox" and scope.bbox is not None:
        south, west, north, east = scope.bbox
        clipped = []
        for line in polylines:
            clipped.extend(clip_polyline_to_bbox(line, south, west, north, east))
        return clipped
    return polylines


def _node_ids_per_line(
    feature: dict[str, Any], polylines: list[list[list[float]]]
) -> list[list[int]] | None:
    props = feature.get("properties")
    if not isinstance(props, dict):
        return None
    raw = props.get("nodes")
    if not isinstance(raw, list) or not raw:
        return None
    ids: list[int] = []
    for item in raw:
        if not isinstance(item, int):
            return None
        ids.append(item)
    expected = sum(len(line) for line in polylines)
    if len(ids) != expected:
        return None
    sliced: list[list[int]] = []
    offset = 0
    for line in polylines:
        sliced.append(ids[offset : offset + len(line)])
        offset += len(line)
    return sliced


def _vertex_keys(
    line: list[list[float]],
    origin: tuple[float, float],
    node_ids: list[int] | None,
) -> list[VertexKey]:
    if node_ids is not None and len(node_ids) == len(line):
        return list(node_ids)
    lat0, lon0 = origin
    cos_lat = math.cos(math.radians(lat0)) or 1e-12
    keys: list[VertexKey] = []
    for point in line:
        lon, lat = point[0], point[1]
        x = math.radians(lon - lon0) * cos_lat * EARTH_RADIUS_M
        y = math.radians(lat - lat0) * EARTH_RADIUS_M
        keys.append((round(x / SNAP_CELL_M), round(y / SNAP_CELL_M)))
    return keys
