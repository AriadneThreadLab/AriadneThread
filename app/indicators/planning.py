"""Plan OSM data requirements for selected indicators.

The catalog answers *what concept* is needed. OSM RAG (passed in as offline
evidence, never fetched here) answers *how that concept is tagged*. This module
builds :class:`~app.osm.query_spec.OsmFeatureQuery` objects for ``query_osm``
and records ``resolve_place`` needs. It never executes Overpass, never accepts
raw QL, and never invents tags.

Not wired into the agent loop in Phase 3.
"""

from __future__ import annotations

from typing import Literal

from app.analytics.contracts import KnowledgeSourceRef
from app.analytics.datasets import DatasetRegistry
from app.core.errors import (
    GroundingConflictError,
    IndicatorPlanningError,
    InventedOsmTagError,
    RequirementBudgetExceededError,
    UnknownIndicatorError,
    UnsupportedDataRequirementError,
)
from app.indicators.catalog import INDICATOR_CATALOG, IndicatorCatalog
from app.indicators.contracts import (
    DataPlanningRequest,
    DataRequirement,
    DataRequirementPlan,
    FeatureSpecification,
    GroundedOsmConcept,
    IndicatorDefinition,
    IndicatorRequirementBinding,
    IndicatorSelection,
    PlannedAnalysisBoundary,
    PlannedOsmDataset,
    PlannedReferenceLocation,
    PlanningTarget,
    RagGroundingEvidence,
    parse_tag_literal,
)
from app.osm.query_spec import (
    ALL_ELEMENT_TYPES,
    OsmFeatureQuery,
    PlaceRefScope,
    TagFilter,
)
from app.rag.corpus import is_allowed_source

TagMatch = Literal["all", "any"]
GroundingSource = Literal["catalog_declared", "rag_grounding"]
DatasetIdentity = tuple[object, ...]

_CONCEPT_LABELS: dict[str, str] = {
    "green_polygons": "green polygons",
    "highway_features": "road geometries",
    "service_pois": "service POIs",
    "subject_features": "subject OSM features",
    "reference_point": "reference location",
    "analysis_area": "analysis boundary",
}

_SUPPORTED_KINDS = frozenset({"osm_features", "reference_point", "analysis_area"})
_SUPPORTED_SOURCES = frozenset({"openstreetmap", "user_input", "derived"})
_OSM_SOURCES = frozenset({"openstreetmap"})


class DataRequirementPlanner:
    """Turn catalog requirements and targets into a deduplicated OSM retrieval plan."""

    def __init__(self, catalog: IndicatorCatalog | None = None) -> None:
        self._catalog = catalog if catalog is not None else INDICATOR_CATALOG

    def plan(self, request: DataPlanningRequest) -> DataRequirementPlan:
        """Build the plan. Does not call Overpass, Nominatim, or OSM RAG."""
        definitions = self._definitions(request.indicator_ids)
        self._assert_supported(definitions)
        grounding_index = _index_grounding(request.rag_grounding)

        osm_rows: list[PlannedOsmDataset] = []
        identity_to_index: dict[DatasetIdentity, int] = {}
        references: list[PlannedReferenceLocation] = []
        ref_identity: dict[tuple[str, str], int] = {}
        areas: list[PlannedAnalysisBoundary] = []
        area_identity: dict[tuple[str, str], int] = {}
        concepts: dict[str, GroundedOsmConcept] = {}
        bindings: list[IndicatorRequirementBinding] = []
        consolidated: set[str] = set()

        for definition in definitions:
            dataset_keys: list[str] = []
            reference_targets: list[str] = []
            area_targets: list[str] = []
            for requirement in definition.requirements:
                evidence = _grounding_for(requirement.requirement_id, grounding_index)
                for target in request.targets:
                    if requirement.kind == "osm_features":
                        dataset, identity = self._plan_osm_dataset(
                            definition, requirement, target, evidence
                        )
                        existing = identity_to_index.get(identity)
                        if existing is None:
                            dataset_key = f"planned_osm_{len(osm_rows) + 1}"
                            stored = dataset.model_copy(update={"dataset_key": dataset_key})
                            identity_to_index[identity] = len(osm_rows)
                            osm_rows.append(stored)
                            dataset_keys.append(dataset_key)
                            _remember_concept(concepts, stored, definition.indicator_id)
                        else:
                            merged = _merge_dataset(
                                osm_rows[existing], definition.indicator_id, dataset
                            )
                            osm_rows[existing] = merged
                            consolidated.add(requirement.requirement_id)
                            dataset_keys.append(merged.dataset_key)
                            _remember_concept(concepts, merged, definition.indicator_id)
                    elif requirement.kind == "reference_point":
                        planned = self._plan_reference(requirement, target, definition)
                        key = (planned.requirement_id, planned.target_id)
                        existing_ref = ref_identity.get(key)
                        if existing_ref is None:
                            ref_identity[key] = len(references)
                            references.append(planned)
                        else:
                            previous = references[existing_ref]
                            references[existing_ref] = previous.model_copy(
                                update={
                                    "requested_by": _unique(
                                        (*previous.requested_by, definition.indicator_id)
                                    )
                                }
                            )
                            consolidated.add(requirement.requirement_id)
                        reference_targets.append(target.target_id)
                    else:
                        planned_area = PlannedAnalysisBoundary(
                            requirement_id=requirement.requirement_id,
                            target_id=target.target_id,
                            requested_by=(definition.indicator_id,),
                        )
                        key = (planned_area.requirement_id, planned_area.target_id)
                        existing_area = area_identity.get(key)
                        if existing_area is None:
                            area_identity[key] = len(areas)
                            areas.append(planned_area)
                        else:
                            previous_area = areas[existing_area]
                            areas[existing_area] = previous_area.model_copy(
                                update={
                                    "requested_by": _unique(
                                        (*previous_area.requested_by, definition.indicator_id)
                                    )
                                }
                            )
                            consolidated.add(requirement.requirement_id)
                        area_targets.append(target.target_id)

            bindings.append(
                IndicatorRequirementBinding(
                    indicator_id=definition.indicator_id,
                    requirement_ids=tuple(item.requirement_id for item in definition.requirements),
                    dataset_keys=_unique(tuple(dataset_keys)),
                    reference_target_ids=_unique(tuple(reference_targets)),
                    analysis_area_target_ids=_unique(tuple(area_targets)),
                )
            )

        if len(osm_rows) > DatasetRegistry.MAX_DATASETS:
            raise RequirementBudgetExceededError(
                f"planned {len(osm_rows)} OSM retrievals; at most "
                f"{DatasetRegistry.MAX_DATASETS} datasets may be registered per request"
            )

        required_data = _unique(
            tuple(
                item.requirement_id
                for definition in definitions
                for item in definition.requirements
            )
        )
        documentation = _unique_docs(
            tuple(doc for concept in concepts.values() for doc in concept.documentation)
        )
        return DataRequirementPlan(
            selected_indicators=tuple(item.indicator_id for item in definitions),
            required_data=required_data,
            indicator_bindings=tuple(bindings),
            grounded_concepts=tuple(concepts[key] for key in sorted(concepts)),
            planned_datasets=tuple(osm_rows),
            reference_locations=tuple(references),
            analysis_boundaries=tuple(areas),
            documentation=documentation,
            retrieved_dataset_refs=(),
            consolidated_requirement_ids=tuple(sorted(consolidated)),
            indicator_catalog_version=self._catalog.version,
        )

    def _definitions(self, indicator_ids: tuple[str, ...]) -> tuple[IndicatorDefinition, ...]:
        known = ", ".join(self._catalog.indicator_ids()) or "none"
        definitions: list[IndicatorDefinition] = []
        for indicator_id in indicator_ids:
            try:
                definitions.append(self._catalog.get_indicator(indicator_id))
            except UnknownIndicatorError:
                raise UnknownIndicatorError(
                    f"unknown indicator_id '{indicator_id}'; available: {known}"
                ) from None
        return tuple(definitions)

    def _assert_supported(self, definitions: tuple[IndicatorDefinition, ...]) -> None:
        for definition in definitions:
            for requirement in definition.requirements:
                if requirement.kind not in _SUPPORTED_KINDS:
                    raise UnsupportedDataRequirementError(
                        f"indicator '{definition.indicator_id}' requirement "
                        f"'{requirement.requirement_id}' has unsupported kind "
                        f"'{requirement.kind}'"
                    )
                if requirement.source not in _SUPPORTED_SOURCES:
                    raise UnsupportedDataRequirementError(
                        f"indicator '{definition.indicator_id}' requirement "
                        f"'{requirement.requirement_id}' has unsupported source "
                        f"'{requirement.source}'"
                    )
                if requirement.kind == "osm_features" and requirement.source not in _OSM_SOURCES:
                    raise UnsupportedDataRequirementError(
                        f"OSM features cannot be sourced from '{requirement.source}'"
                    )

    def _plan_osm_dataset(
        self,
        definition: IndicatorDefinition,
        requirement: DataRequirement,
        target: PlanningTarget,
        grounding: tuple[RagGroundingEvidence, ...],
    ) -> tuple[PlannedOsmDataset, DatasetIdentity]:
        spec = requirement.feature_spec
        if spec is None:
            raise UnsupportedDataRequirementError(
                f"requirement '{requirement.requirement_id}' is missing feature_spec"
            )
        tags, tag_match, docs, source = _resolve_tags(definition, requirement, spec, grounding)
        if not tags:
            raise IndicatorPlanningError(
                f"requirement '{requirement.requirement_id}' has no OSM tags to retrieve",
                reason="missing_rag_grounding",
            )
        query = OsmFeatureQuery(
            place=target.place,
            place_ref_scope=_place_ref_scope(target),
            tags=list(tags),
            tag_match=tag_match,
            element_types=list(spec.element_types),
            include_geometry=True,
        )
        planned = PlannedOsmDataset(
            dataset_key="planned_osm_1",
            requirement_id=requirement.requirement_id,
            target_id=target.target_id,
            query=query,
            requested_by=(definition.indicator_id,),
            documentation=docs,
            grounding_source=source,
        )
        identity: DatasetIdentity = (
            requirement.requirement_id,
            target.target_id,
            tuple((item.key, item.value) for item in tags),
            tag_match,
            tuple(kind for kind in ALL_ELEMENT_TYPES if kind in spec.element_types),
            target.place_ref,
            target.radius_m,
            target.place,
        )
        return planned, identity

    def _plan_reference(
        self,
        requirement: DataRequirement,
        target: PlanningTarget,
        definition: IndicatorDefinition,
    ) -> PlannedReferenceLocation:
        if target.place_ref is None:
            raise IndicatorPlanningError(
                f"indicator '{definition.indicator_id}' needs a resolve_place "
                f"reference location for target '{target.target_id}'",
                reason="missing_reference_location",
            )
        return PlannedReferenceLocation(
            requirement_id=requirement.requirement_id,
            target_id=target.target_id,
            place_ref=target.place_ref,
            requested_by=(definition.indicator_id,),
        )


def plan_indicator_data(
    request: DataPlanningRequest,
    *,
    catalog: IndicatorCatalog | None = None,
) -> DataRequirementPlan:
    """Convenience entry: selected indicators + targets → OSM data plan."""
    return DataRequirementPlanner(catalog).plan(request)


def plan_indicator_data_from_selection(
    selection: IndicatorSelection,
    targets: tuple[PlanningTarget, ...],
    *,
    rag_grounding: tuple[RagGroundingEvidence, ...] = (),
    catalog: IndicatorCatalog | None = None,
) -> DataRequirementPlan:
    """Plan OSM retrievals for a Phase 2 selection. Acquisition is not run."""
    if selection.primary_indicator_id is None:
        raise IndicatorPlanningError(
            "indicator selection has no primary indicator to plan data for",
            reason="empty_selection",
        )
    request = DataPlanningRequest(
        indicator_ids=(
            selection.primary_indicator_id,
            *selection.supporting_indicator_ids,
        ),
        targets=targets,
        rag_grounding=rag_grounding,
    )
    return plan_indicator_data(request, catalog=catalog)


def _place_ref_scope(target: PlanningTarget) -> PlaceRefScope | None:
    if target.place_ref is None or target.radius_m is None:
        return None
    return PlaceRefScope(place_ref=target.place_ref, radius_m=target.radius_m)


def _index_grounding(
    evidence: tuple[RagGroundingEvidence, ...],
) -> dict[str | None, tuple[RagGroundingEvidence, ...]]:
    indexed: dict[str | None, list[RagGroundingEvidence]] = {}
    for item in evidence:
        _assert_whitelisted_docs(item.documentation)
        indexed.setdefault(item.requirement_id, []).append(item)
    return {key: tuple(value) for key, value in indexed.items()}


def _grounding_for(
    requirement_id: str,
    indexed: dict[str | None, tuple[RagGroundingEvidence, ...]],
) -> tuple[RagGroundingEvidence, ...]:
    return (*indexed.get(requirement_id, ()), *indexed.get(None, ()))


def _assert_whitelisted_docs(documentation: tuple[KnowledgeSourceRef, ...]) -> None:
    for source in documentation:
        if source.url is None:
            continue
        if not is_allowed_source(source.url):
            raise IndicatorPlanningError(
                f"OSM RAG citation is not on the knowledge whitelist: {source.url}",
                reason="documentation_not_whitelisted",
            )


def _resolve_tags(
    definition: IndicatorDefinition,
    requirement: DataRequirement,
    spec: FeatureSpecification,
    grounding: tuple[RagGroundingEvidence, ...],
) -> tuple[tuple[TagFilter, ...], TagMatch, tuple[KnowledgeSourceRef, ...], GroundingSource]:
    catalog_tags, tag_match = _retrieval_tags(spec)
    catalog_docs = spec.documentation
    rag_tags = _grounding_tags(grounding)
    rag_docs = tuple(doc for item in grounding for doc in item.documentation)
    _assert_whitelisted_docs(catalog_docs)

    if definition.domain == "core" and not catalog_tags:
        if not rag_tags:
            raise IndicatorPlanningError(
                f"core indicator '{definition.indicator_id}' requires OSM RAG "
                "tag grounding for subject_features",
                reason="missing_rag_grounding",
            )
        if not rag_docs:
            raise IndicatorPlanningError(
                "OSM RAG tag grounding must include whitelist documentation provenance",
                reason="missing_rag_grounding",
            )
        return rag_tags, ("any" if len(rag_tags) > 1 else "all"), rag_docs, "rag_grounding"

    if rag_tags:
        catalog_set = {(item.key, item.value) for item in catalog_tags}
        rag_set = {(item.key, item.value) for item in rag_tags}
        extra = rag_set - catalog_set
        if extra:
            raise InventedOsmTagError(
                "OSM RAG/model tags are not in the catalog feature specification "
                f"for '{requirement.requirement_id}': {sorted(extra)}"
            )
        if rag_set != catalog_set:
            raise GroundingConflictError(
                f"OSM RAG tags conflict with catalog tags for '{requirement.requirement_id}'"
            )

    docs = _unique_docs((*catalog_docs, *rag_docs))
    return catalog_tags, tag_match, docs, "catalog_declared"


def _retrieval_tags(spec: FeatureSpecification) -> tuple[tuple[TagFilter, ...], TagMatch]:
    filters: list[TagFilter] = []
    seen: set[tuple[str, str | None]] = set()
    multi_category = len(spec.categories) > 1
    uses_any = False
    uses_all_multi = False
    for category in spec.categories:
        category_filters = category.tag_filters()
        if category.tag_match == "any" and len(category_filters) > 1:
            uses_any = True
        if category.tag_match == "all" and len(category_filters) > 1:
            uses_all_multi = True
        for item in category_filters:
            key = (item.key, item.value)
            if key in seen:
                continue
            seen.add(key)
            filters.append(item)
    if uses_all_multi and (uses_any or multi_category):
        raise UnsupportedDataRequirementError(
            "cannot flatten mixed AND/OR tag categories into one OsmFeatureQuery"
        )
    if len(filters) > 8:
        raise UnsupportedDataRequirementError(
            f"requirement expands to {len(filters)} tags; OsmFeatureQuery allows at most 8"
        )
    if uses_all_multi:
        return tuple(filters), "all"
    if multi_category or uses_any or len(filters) > 1:
        return tuple(filters), "any"
    return tuple(filters), "all"


def _grounding_tags(grounding: tuple[RagGroundingEvidence, ...]) -> tuple[TagFilter, ...]:
    tags: list[TagFilter] = []
    seen: set[tuple[str, str | None]] = set()
    for item in grounding:
        for literal in item.tags:
            parsed = parse_tag_literal(literal)
            key = (parsed.key, parsed.value)
            if key in seen:
                continue
            seen.add(key)
            tags.append(parsed)
    if len(tags) > 8:
        raise InventedOsmTagError("OSM RAG grounding supplied more than 8 tags")
    return tuple(tags)


def _merge_dataset(
    existing: PlannedOsmDataset,
    indicator_id: str,
    incoming: PlannedOsmDataset,
) -> PlannedOsmDataset:
    requested = _unique((*existing.requested_by, indicator_id))
    docs = _unique_docs((*existing.documentation, *incoming.documentation))
    return existing.model_copy(update={"requested_by": requested, "documentation": docs})


def _remember_concept(
    concepts: dict[str, GroundedOsmConcept],
    dataset: PlannedOsmDataset,
    indicator_id: str,
) -> None:
    key = f"{dataset.requirement_id}:{dataset.target_id}"
    tags = tuple(
        item.key if item.value is None else f"{item.key}={item.value}"
        for item in dataset.query.tags
    )
    existing = concepts.get(key)
    if existing is None:
        concepts[key] = GroundedOsmConcept(
            requirement_id=dataset.requirement_id,
            concept=_CONCEPT_LABELS.get(dataset.requirement_id, dataset.requirement_id),
            tags=tags,
            documentation=dataset.documentation,
            grounding_source=dataset.grounding_source,
            requested_by=dataset.requested_by,
        )
        return
    requested = _unique((*existing.requested_by, indicator_id))
    concepts[key] = existing.model_copy(
        update={
            "documentation": _unique_docs((*existing.documentation, *dataset.documentation)),
            "requested_by": requested,
        }
    )


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _unique_docs(values: tuple[KnowledgeSourceRef, ...]) -> tuple[KnowledgeSourceRef, ...]:
    seen: set[tuple[str, str | None]] = set()
    ordered: list[KnowledgeSourceRef] = []
    for item in values:
        key = (item.title, item.url)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return tuple(ordered)
