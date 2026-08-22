"""Combine multi-target OSM FeatureCollections for map / download.

Target-level analyses remain independent (separate ``dataset_ref`` values).
The combined collection is for map/download only: each target's features are
kept as their own copies so MapLibre can colour by ``target_index``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.osm.contracts import GeoJsonFeatureCollection


def annotate_features_for_target(
    collection: GeoJsonFeatureCollection,
    *,
    analysis_target: str,
    place_ref: str | None = None,
    target_id: str | None = None,
    target_index: int | None = None,
) -> GeoJsonFeatureCollection:
    """Copy features and attach application-level target metadata (not OSM tags).

    Top-level primitive properties are duplicated so MapLibre can style by
    target without reading nested objects.
    """
    features_in = collection.get("features")
    if not isinstance(features_in, list):
        return {"type": "FeatureCollection", "features": []}
    out: list[Any] = []
    for raw in features_in:
        if not isinstance(raw, dict):
            continue
        feature = deepcopy(raw)
        props = feature.get("properties")
        if not isinstance(props, dict):
            props = {}
            feature["properties"] = props
        # Keep OSM ``tags`` untouched; store Ariadne metadata separately.
        meta = props.get("ariadne")
        if not isinstance(meta, dict):
            meta = {}
        meta = dict(meta)
        meta["analysis_target"] = analysis_target
        targets = meta.get("analysis_targets")
        if isinstance(targets, list):
            merged = [str(item) for item in targets if isinstance(item, str)]
        else:
            merged = []
        if analysis_target not in merged:
            merged.append(analysis_target)
        meta["analysis_targets"] = merged
        if place_ref is not None:
            meta["place_ref"] = place_ref
        if target_id is not None:
            meta["target_id"] = target_id
        if target_index is not None:
            meta["target_index"] = target_index
        props["ariadne"] = meta
        props["analysis_target"] = analysis_target
        props["analysis_target_label"] = analysis_target
        props["target_label"] = analysis_target
        if target_id is not None:
            props["target_id"] = target_id
        if target_index is not None:
            props["target_index"] = target_index
        out.append(feature)
    return {"type": "FeatureCollection", "features": out}


def combine_target_feature_collections(
    collections: list[tuple[str, GeoJsonFeatureCollection]],
) -> GeoJsonFeatureCollection:
    """Concatenate per-target FeatureCollections for map display.

    A feature that falls in more than one radius is duplicated, once per
    target, so each copy keeps a single ``analysis_target`` / ``target_index``
    for map colouring. Analytics still use the separate request-scoped
    datasets, not this combined collection.
    """
    features: list[dict[str, Any]] = []
    multi = len(collections) > 1
    for index, (target, collection) in enumerate(collections):
        annotated = annotate_features_for_target(
            collection,
            analysis_target=target,
            target_id=f"t{index + 1}" if multi else None,
            target_index=index if multi else None,
        )
        for feature in annotated.get("features", []):
            if isinstance(feature, dict):
                features.append(feature)
    return {"type": "FeatureCollection", "features": features}
