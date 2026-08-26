"""OSM data-requirement planning for selected indicators (offline)."""

from __future__ import annotations

import pytest
from app.analytics.contracts import KnowledgeSourceRef
from app.analytics.datasets import DatasetRegistry
from app.core.errors import (
    GroundingConflictError,
    IndicatorPlanningError,
    InventedOsmTagError,
    RequirementBudgetExceededError,
    UnknownIndicatorError,
)
from app.indicators.catalog import INDICATOR_CATALOG, INDICATOR_CATALOG_VERSION, get_indicator
from app.indicators.contracts import (
    DataPlanningRequest,
    DataRequirementPlan,
    IndicatorSelection,
    PlanningTarget,
    RagGroundingEvidence,
)
from app.indicators.planning import (
    plan_indicator_data,
    plan_indicator_data_from_selection,
)
from app.osm.query_builder import build_overpass_query
from app.osm.query_spec import OsmFeatureQuery, PointRadius
from app.rag.corpus import is_allowed_source
from pydantic import ValidationError

_PARK_URL = "https://wiki.openstreetmap.org/wiki/Tag:leisure%3Dpark"
_GRASS_URL = "https://wiki.openstreetmap.org/wiki/Tag:landuse%3Dgrass"
_WOOD_URL = "https://wiki.openstreetmap.org/wiki/Tag:natural%3Dwood"
_MAP_FEATURES_URL = "https://wiki.openstreetmap.org/wiki/Map_Features"

_PARK_GROUNDING = RagGroundingEvidence(
    requirement_id="subject_features",
    tags=("leisure=park",),
    documentation=(KnowledgeSourceRef(title="Tag: leisure=park", url=_PARK_URL),),
)

EXPECTED_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "feature_count": ("subject_features",),
    "green_space_ratio": ("green_polygons", "analysis_area"),
    "green_accessibility": ("green_polygons", "reference_point"),
    "green_diversity": ("green_polygons",),
    "road_accessibility": ("highway_features", "reference_point"),
    "road_density": ("highway_features", "analysis_area"),
    "intersection_density": ("highway_features", "analysis_area"),
    "poi_density": ("service_pois", "analysis_area"),
    "service_accessibility": ("service_pois", "reference_point"),
    "service_diversity": ("service_pois",),
}

GREEN_TAGS = ("leisure=park", "leisure=garden", "landuse=grass", "natural=wood")
HIGHWAY_TAGS = (
    "highway=motorway",
    "highway=trunk",
    "highway=primary",
    "highway=secondary",
    "highway=tertiary",
    "highway=unclassified",
    "highway=residential",
    "highway=living_street",
)
SERVICE_TAGS = (
    "amenity=school",
    "amenity=hospital",
    "amenity=clinic",
    "amenity=pharmacy",
    "amenity=bank",
    "shop=supermarket",
    "amenity=marketplace",
    "amenity=restaurant",
)


def _target(target_id: str = "campus_a", *, place_n: int = 1) -> PlanningTarget:
    return PlanningTarget(
        target_id=target_id,
        label=target_id.replace("_", " ").title(),
        place_ref=f"place_{place_n}",
        radius_m=2000,
    )


def _plan(
    indicator_ids: tuple[str, ...],
    targets: tuple[PlanningTarget, ...] | None = None,
    *,
    rag_grounding: tuple[RagGroundingEvidence, ...] = (),
) -> DataRequirementPlan:
    return plan_indicator_data(
        DataPlanningRequest(
            indicator_ids=indicator_ids,
            targets=targets or (_target(),),
            rag_grounding=rag_grounding,
        )
    )


def _tag_literals(query: OsmFeatureQuery) -> tuple[str, ...]:
    return tuple(
        item.key if item.value is None else f"{item.key}={item.value}" for item in query.tags
    )


@pytest.mark.parametrize("indicator_id,expected", tuple(EXPECTED_REQUIREMENTS.items()))
def test_each_indicator_resolves_catalog_requirements(
    indicator_id: str, expected: tuple[str, ...]
) -> None:
    grounding = (_PARK_GROUNDING,) if indicator_id == "feature_count" else ()
    plan = _plan((indicator_id,), rag_grounding=grounding)
    assert plan.required_data == expected
    assert plan.indicator_bindings[0].requirement_ids == expected
    assert plan.selected_indicators == (indicator_id,)
    assert plan.indicator_catalog_version == INDICATOR_CATALOG_VERSION
    assert plan.retrieved_dataset_refs == ()
    definition = get_indicator(indicator_id)
    assert tuple(item.requirement_id for item in definition.requirements) == expected


def test_green_space_ratio_plans_polygons_and_derived_boundary() -> None:
    plan = _plan(("green_space_ratio",))
    assert len(plan.planned_datasets) == 1
    dataset = plan.planned_datasets[0]
    assert dataset.requirement_id == "green_polygons"
    assert dataset.retrieval_tool == "query_osm"
    assert _tag_literals(dataset.query) == GREEN_TAGS
    assert dataset.query.tag_match == "any"
    assert dataset.query.place_ref_scope is not None
    assert dataset.query.place_ref_scope.place_ref == "place_1"
    assert plan.analysis_boundaries[0].requirement_id == "analysis_area"
    assert plan.reference_locations == ()


def test_green_accessibility_needs_reference_location() -> None:
    plan = _plan(("green_accessibility",))
    assert plan.planned_datasets[0].requirement_id == "green_polygons"
    assert plan.reference_locations[0].place_ref == "place_1"
    assert plan.reference_locations[0].retrieval_tool == "resolve_place"
    assert plan.analysis_boundaries == ()


def test_road_indicators_use_highway_feature_spec() -> None:
    density = _plan(("road_density",))
    access = _plan(("road_accessibility",))
    intersections = _plan(("intersection_density",))
    for plan in (density, access, intersections):
        assert _tag_literals(plan.planned_datasets[0].query) == HIGHWAY_TAGS
        assert plan.planned_datasets[0].query.tag_match == "any"
        assert plan.planned_datasets[0].query.element_types == ["way"]
    assert density.analysis_boundaries[0].requirement_id == "analysis_area"
    assert intersections.analysis_boundaries[0].requirement_id == "analysis_area"
    assert access.reference_locations[0].requirement_id == "reference_point"


def test_service_indicators_use_poi_feature_spec() -> None:
    poi = _plan(("poi_density",))
    access = _plan(("service_accessibility",))
    diversity = _plan(("service_diversity",))
    for plan in (poi, access, diversity):
        assert _tag_literals(plan.planned_datasets[0].query) == SERVICE_TAGS
    assert poi.analysis_boundaries != ()
    assert access.reference_locations != ()
    assert diversity.analysis_boundaries == ()


def test_duplicate_green_polygon_requirements_are_consolidated() -> None:
    targets = (_target("campus_a", place_n=1), _target("campus_b", place_n=2))
    plan = _plan(("green_space_ratio", "green_accessibility"), targets)
    assert len(plan.planned_datasets) == 2
    assert {item.target_id for item in plan.planned_datasets} == {"campus_a", "campus_b"}
    for dataset in plan.planned_datasets:
        assert dataset.requirement_id == "green_polygons"
        assert set(dataset.requested_by) == {"green_space_ratio", "green_accessibility"}
        assert _tag_literals(dataset.query) == GREEN_TAGS
    assert "green_polygons" in plan.consolidated_requirement_ids
    assert len(plan.reference_locations) == 2
    assert len(plan.analysis_boundaries) == 2


def test_osm_rag_provenance_is_retained() -> None:
    plan = _plan(("green_space_ratio",))
    urls = {item.url for item in plan.documentation if item.url is not None}
    assert urls <= {_PARK_URL, _GRASS_URL, _WOOD_URL, _MAP_FEATURES_URL}
    assert _PARK_URL in urls
    assert all(is_allowed_source(url) for url in urls if url is not None)
    concepts = [item for item in plan.grounded_concepts if item.requirement_id == "green_polygons"]
    assert concepts
    assert concepts[0].concept == "green polygons"
    assert concepts[0].grounding_source == "catalog_declared"
    assert _PARK_URL in {item.url for item in concepts[0].documentation}


def test_core_feature_count_uses_rag_tags_and_provenance() -> None:
    plan = _plan(("feature_count",), rag_grounding=(_PARK_GROUNDING,))
    dataset = plan.planned_datasets[0]
    assert dataset.requirement_id == "subject_features"
    assert dataset.grounding_source == "rag_grounding"
    assert _tag_literals(dataset.query) == ("leisure=park",)
    assert dataset.documentation[0].url == _PARK_URL
    assert plan.grounded_concepts[0].grounding_source == "rag_grounding"


def test_core_feature_count_without_rag_fails() -> None:
    with pytest.raises(IndicatorPlanningError, match="OSM RAG"):
        _plan(("feature_count",))


def test_unsupported_invented_requirement_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DataPlanningRequest.model_validate(
            {
                "indicator_ids": ["green_space_ratio"],
                "targets": [_target().model_dump()],
                "requirement_ids": ["gtfs_stops"],
            }
        )


def test_unknown_indicator_fails_cleanly() -> None:
    with pytest.raises(UnknownIndicatorError, match="satellite_ndvi"):
        _plan(("satellite_ndvi",))


def test_model_cannot_invent_osm_tags_on_the_request() -> None:
    with pytest.raises(ValidationError):
        DataPlanningRequest.model_validate(
            {
                "indicator_ids": ["green_space_ratio"],
                "targets": [_target().model_dump()],
                "tags": [{"key": "amenity", "value": "fuel"}],
            }
        )


def test_rag_cannot_inject_tags_absent_from_the_catalog() -> None:
    invented = RagGroundingEvidence(
        requirement_id="green_polygons",
        tags=("amenity=fuel",),
        documentation=(KnowledgeSourceRef(title="Tag: leisure=park", url=_PARK_URL),),
    )
    with pytest.raises(InventedOsmTagError, match="amenity"):
        _plan(("green_space_ratio",), rag_grounding=(invented,))


def test_partial_rag_tags_conflict_with_catalog_are_not_merged() -> None:
    partial = RagGroundingEvidence(
        requirement_id="green_polygons",
        tags=("leisure=park",),
        documentation=(KnowledgeSourceRef(title="Tag: leisure=park", url=_PARK_URL),),
    )
    with pytest.raises(GroundingConflictError, match="conflict"):
        _plan(("green_space_ratio",), rag_grounding=(partial,))


def test_arbitrary_overpass_ql_is_rejected_on_the_planning_request() -> None:
    with pytest.raises(ValidationError):
        DataPlanningRequest.model_validate(
            {
                "indicator_ids": ["green_space_ratio"],
                "targets": [_target().model_dump()],
                "overpass_ql": "[out:json];node;out;",
            }
        )


def test_planned_query_is_osm_feature_query_not_raw_ql() -> None:
    plan = _plan(("green_space_ratio",))
    query = plan.planned_datasets[0].query
    assert isinstance(query, OsmFeatureQuery)
    dumped = query.model_dump()
    assert "overpass_ql" not in dumped
    assert query.place_ref_scope is not None
    assert query.point is None
    with pytest.raises(ValidationError):
        OsmFeatureQuery.model_validate({**dumped, "overpass_ql": "[out:json];node;out;"})
    # query_osm binds trusted coordinates before the deterministic builder runs.
    executable = query.model_copy(
        update={
            "place_ref_scope": None,
            "point": PointRadius(lat=35.702, lon=51.395, radius_m=2000),
        }
    )
    built = build_overpass_query(executable, timeout_seconds=25)
    assert built.startswith("[out:json]")
    assert 'way["leisure"="park"]' in built
    assert "around:2000,35.702,51.395" in built


def test_spatial_trust_rejects_invented_coordinates_on_targets() -> None:
    with pytest.raises(ValidationError):
        PlanningTarget.model_validate(
            {
                "target_id": "campus_a",
                "label": "Campus A",
                "lat": 35.7,
                "lon": 51.4,
                "radius_m": 2000,
            }
        )


def test_accessibility_without_place_ref_fails() -> None:
    named = PlanningTarget(target_id="berlin", label="Berlin", place="Berlin")
    with pytest.raises(IndicatorPlanningError, match="resolve_place"):
        _plan(("green_accessibility",), (named,))


def test_non_whitelist_rag_url_is_rejected() -> None:
    evidence = RagGroundingEvidence(
        requirement_id="subject_features",
        tags=("leisure=park",),
        documentation=(
            KnowledgeSourceRef(
                title="Fuel",
                url="https://wiki.openstreetmap.org/wiki/Tag:amenity%3Dfuel",
            ),
        ),
    )
    with pytest.raises(IndicatorPlanningError, match="whitelist"):
        _plan(("feature_count",), rag_grounding=(evidence,))


def test_requirement_budget_matches_dataset_registry() -> None:
    targets = tuple(_target(f"t{index}", place_n=index) for index in range(1, 5))
    with pytest.raises(RequirementBudgetExceededError):
        _plan(("green_space_ratio", "poi_density", "road_density"), targets)
    assert DatasetRegistry.MAX_DATASETS == 8


def test_plan_from_phase2_selection_consolidates_supporting_indicators() -> None:
    selection = IndicatorSelection(
        primary_indicator_id="green_space_ratio",
        supporting_indicator_ids=("green_accessibility", "green_diversity"),
    )
    plan = plan_indicator_data_from_selection(selection, (_target(),))
    assert set(plan.selected_indicators) == {
        "green_space_ratio",
        "green_accessibility",
        "green_diversity",
    }
    assert len(plan.planned_datasets) == 1
    assert set(plan.planned_datasets[0].requested_by) == {
        "green_space_ratio",
        "green_accessibility",
        "green_diversity",
    }


def test_catalog_still_owns_every_phase1_indicator() -> None:
    assert tuple(sorted(INDICATOR_CATALOG.indicator_ids())) == tuple(sorted(EXPECTED_REQUIREMENTS))
