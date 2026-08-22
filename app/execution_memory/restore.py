"""Restore persistent execution datasets/places into a new request scope."""

from __future__ import annotations

from app.analytics.datasets import DatasetRecord, DatasetRegistry
from app.execution_memory.contracts import (
    DocumentationSourceSnapshot,
    ExecutionDatasetSnapshot,
    TrustedPlaceSnapshot,
)
from app.places.contracts import PlaceRegistry, ResolvedPlaceRecord
from app.rag.contracts import RetrievedPassage


def restore_place(registry: PlaceRegistry, place: TrustedPlaceSnapshot) -> ResolvedPlaceRecord:
    """Trusted Nominatim snapshot → new request-scoped place_N."""
    return registry.register(
        query=place.query,
        label=place.label,
        display_name=place.display_name,
        latitude=place.latitude,
        longitude=place.longitude,
        source=place.source,
        source_id=place.source_id,
    )


def restore_dataset(
    registry: DatasetRegistry,
    snapshot: ExecutionDatasetSnapshot,
) -> str:
    """Persistent execution dataset → fresh request-scoped osm_result_N.

    The previous request's ``osm_result_N`` is never reused as identity.
    """
    return registry.restore(
        feature_collection=snapshot.feature_collection,
        feature_count=snapshot.feature_count,
        resolved_tags=snapshot.resolved_tags,
        scope=snapshot.scope,
        effective_limit=snapshot.effective_limit,
        truncated=snapshot.truncated,
        retrieved_at=snapshot.retrieved_at,
        endpoint=snapshot.endpoint,
    )


def restored_ref_is_fresh(previous_ref: str | None, new_ref: str) -> bool:
    """True when restore assigned a new request-scoped dataset_ref."""
    if not previous_ref:
        return True
    return previous_ref != new_ref


def restore_documentation_passages(
    citations: tuple[DocumentationSourceSnapshot, ...] | list[DocumentationSourceSnapshot],
) -> list[RetrievedPassage]:
    """Turn stored Wiki citations into request-scoped passages for the UI.

    Does not replay corpus text. The UI citation panel uses title, section,
    URL and score; analysis provenance uses the same SourceReference list.
    """
    passages: list[RetrievedPassage] = []
    seen: set[tuple[str, str | None, str]] = set()
    for item in citations:
        key = (item.title, item.section, item.url)
        if key in seen:
            continue
        seen.add(key)
        passages.append(
            RetrievedPassage(
                content=(
                    "Restored OSM documentation citation from a prior validated "
                    "grounding. Not live map data."
                ),
                document_title=item.title,
                section=item.section,
                source_url=item.url,
                score=item.score,
            )
        )
    return passages


def as_dataset_record(dataset_ref: str, snapshot: ExecutionDatasetSnapshot) -> DatasetRecord:
    return DatasetRecord(
        dataset_ref=dataset_ref,
        feature_collection=snapshot.feature_collection,
        feature_count=snapshot.feature_count,
        resolved_tags=snapshot.resolved_tags,
        scope=snapshot.scope,
        effective_limit=snapshot.effective_limit,
        truncated=snapshot.truncated,
        retrieved_at=snapshot.retrieved_at,
        endpoint=snapshot.endpoint,
    )
