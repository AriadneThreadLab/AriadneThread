"""Assign OSM features to catalog TagCategory partitions."""

from __future__ import annotations

from typing import Any

from app.indicators.contracts import FeatureSpecification, TagCategory
from app.osm.query_spec import TagFilter


def feature_tags(feature: dict[str, Any]) -> dict[str, Any] | None:
    """Return the OSM tag map, or None when tags are missing."""
    props = feature.get("properties")
    if not isinstance(props, dict):
        return None
    tags = props.get("tags")
    if not isinstance(tags, dict):
        return None
    if not tags:
        return None
    return tags


def assign_category(
    tags: dict[str, Any],
    categories: tuple[TagCategory, ...],
) -> tuple[str | None, tuple[str, ...]]:
    """Return (first matching category id, all matching ids).

    Unknown tag sets yield ``(None, ())``. Multiple matches keep catalog order.
    """
    matched: list[str] = []
    for category in categories:
        if _category_matches(tags, category):
            matched.append(category.id)
    if not matched:
        return None, ()
    return matched[0], tuple(matched)


def categorize_features(
    features: list[dict[str, Any]],
    spec: FeatureSpecification,
) -> tuple[dict[str, int], int, int, int]:
    """Count features per category.

    Returns ``(counts, missing, unknown, multi_match)``. Missing means no tag
    map; unknown means tags that match no catalog category.
    """
    counts: dict[str, int] = {category.id: 0 for category in spec.categories}
    missing = 0
    unknown = 0
    multi_match = 0
    for feature in features:
        tags = feature_tags(feature)
        if tags is None:
            missing += 1
            continue
        primary, matched = assign_category(tags, spec.categories)
        if primary is None:
            unknown += 1
            continue
        if len(matched) > 1:
            multi_match += 1
        counts[primary] += 1
    return counts, missing, unknown, multi_match


def _category_matches(tags: dict[str, Any], category: TagCategory) -> bool:
    filters = category.tag_filters()
    if not filters:
        return False
    hits = [_tag_matches(tags, item) for item in filters]
    if category.tag_match == "all":
        return all(hits)
    return any(hits)


def _tag_matches(tags: dict[str, Any], filt: TagFilter) -> bool:
    if filt.key not in tags:
        return False
    raw = tags[filt.key]
    if filt.value is None:
        return raw is not None and str(raw) != ""
    return str(raw) == filt.value
