"""Two-layer Execution Memory: immutable history vs reusable methodology."""

from __future__ import annotations

import json

import pytest
from app.core.config import Settings
from app.execution_memory.analysis_memory import (
    assert_pattern_is_target_free,
    classify_target_category,
    derive_analysis_pattern,
    identity_tokens,
    select_pattern,
    validate_pattern_reuse,
)
from app.execution_memory.repository import InMemoryExecutionMemoryRepository
from app.execution_memory.service import ExecutionMemoryService

from tests.test_execution_memory import (
    _CONV,
    _TEHRAN_AMIRKABIR_MSG,
    _TEHRAN_SHARIF_MSG,
    _THREE_UNIVERSITIES_MSG,
    make_ab_snapshot,
)

_TEHRAN = "University of Tehran"
_SHARIF = "Sharif University of Technology"
_AMIRKABIR = "Amirkabir University of Technology"


def _pattern_blob(pattern: object) -> str:
    return json.dumps(pattern.model_dump(mode="json"), default=str).lower()  # type: ignore[attr-defined]


def test_pattern_generalises_targets_to_a_category():
    pattern = derive_analysis_pattern(make_ab_snapshot())
    assert pattern is not None
    assert pattern.analysis_pattern == "compare public parks around educational facilities"
    assert pattern.target_category == "educational_facility"
    assert pattern.pattern_key == "comparison|public parks|educational_facility|point"


def test_pattern_stores_methodology_not_answers():
    snapshot = make_ab_snapshot()
    pattern = derive_analysis_pattern(snapshot)
    assert pattern is not None
    assert pattern.dataset_tags == ("leisure=park",)
    assert pattern.radius_m == 2000
    assert pattern.metric == "count"
    assert "compatible_targets" in pattern.validation_rules
    blob = _pattern_blob(pattern)
    # No landmark identity, no rankings, no geometry.
    for forbidden in ("sharif", "amirkabir", "35.7", "51.3", "ranking", "features"):
        assert forbidden not in blob


def test_pattern_target_free_invariant_is_enforced():
    pattern = derive_analysis_pattern(make_ab_snapshot())
    assert pattern is not None
    assert identity_tokens(_SHARIF) == {"sharif"}
    assert identity_tokens(_TEHRAN) == {"tehran"}
    assert_pattern_is_target_free(pattern, [_TEHRAN, _SHARIF])
    leaky = pattern.model_copy(update={"analysis_pattern": "compare parks near Sharif"})
    with pytest.raises(ValueError, match="leaked target identity"):
        assert_pattern_is_target_free(leaky, [_SHARIF])


def test_target_category_classification():
    assert classify_target_category([_TEHRAN, _AMIRKABIR]) == "educational_facility"
    assert classify_target_category(["Mehrabad Airport", "Tehran Metro Station"]) == "transport_hub"
    assert classify_target_category(["Milad Tower"]) == "place"


def test_history_is_audit_only_and_carries_the_old_targets():
    """History keeps the old answer; it just must not drive the new one."""
    snapshot = make_ab_snapshot()
    assert [target.label for target in snapshot.targets] == [_TEHRAN, _SHARIF]
    assert snapshot.comparison is not None
    assert snapshot.comparison.ranking


# --- Test 1: first execution, Tehran + Sharif -------------------------------


def test_1_first_execution_two_targets_are_reusable():
    pattern = derive_analysis_pattern(make_ab_snapshot())
    reuse = validate_pattern_reuse(
        pattern,
        requested_labels=(_TEHRAN, _SHARIF),
        feature_concept="public parks",
        inferred_goal="abundance",
        radius_m=2000,
    )
    assert reuse.reusable is True
    assert set(reuse.reused_components) == {
        "dataset_definition",
        "validation_rules",
        "radius",
        "metric",
    }
    assert "targets" in reuse.recomputed_components
    assert "osm_query" in reuse.recomputed_components


# --- Test 2: second execution, Tehran + Amirkabir ---------------------------


@pytest.mark.asyncio
async def test_2_replacing_a_target_reuses_methodology_without_leaking(settings: Settings):
    repo = InMemoryExecutionMemoryRepository()
    await repo.save(make_ab_snapshot())
    service = ExecutionMemoryService(repo, settings)

    plan = await service.prepare_incremental(_CONV, _TEHRAN_AMIRKABIR_MSG)
    assert plan is not None
    labels = [target.label.lower() for target in plan.targets]
    assert any("tehran" in item for item in labels)
    assert any("amirkabir" in item for item in labels)
    assert all("sharif" not in item for item in labels)

    assert plan.pattern is not None
    assert plan.pattern.analysis_pattern == "compare public parks around educational facilities"
    assert "sharif" not in _pattern_blob(plan.pattern)
    assert plan.pattern_reuse is not None
    assert plan.pattern_reuse.reusable is True
    assert "metric" in plan.pattern_reuse.reused_components
    assert "targets" in plan.pattern_reuse.recomputed_components

    trace = service.trace_for(conversation_id=_CONV, plan=plan, snapshot=make_ab_snapshot())
    assert trace.pattern_reusable is True
    assert trace.analysis_pattern == "compare public parks around educational facilities"
    assert "dataset_definition" in trace.reused_components
    assert "osm_query" in trace.recomputed_components
    assert all("sharif" not in item.lower() for item in trace.requested_targets)


# --- Test 3: third execution, Tehran + Sharif + Amirkabir -------------------


@pytest.mark.asyncio
async def test_3_three_targets_are_all_planned(settings: Settings):
    repo = InMemoryExecutionMemoryRepository()
    await repo.save(make_ab_snapshot())
    service = ExecutionMemoryService(repo, settings)

    plan = await service.prepare_incremental(_CONV, _THREE_UNIVERSITIES_MSG)
    assert plan is not None
    assert len(plan.targets) == 3
    labels = " ".join(target.label.lower() for target in plan.targets)
    assert "tehran" in labels and "sharif" in labels and "amirkabir" in labels
    assert plan.pattern_reuse is not None
    assert plan.pattern_reuse.reusable is True
    sources = {target.source for target in plan.targets}
    assert sources == {"reused", "new"}


# --- Test 4: metric no longer suitable --------------------------------------


@pytest.mark.asyncio
async def test_4_unsuitable_metric_is_detected_and_replanned(settings: Settings):
    repo = InMemoryExecutionMemoryRepository()
    await repo.save(make_ab_snapshot())
    service = ExecutionMemoryService(repo, settings)

    plan = await service.prepare_incremental(_CONV, "Compare them by accessibility instead.")
    assert plan is not None
    assert plan.pattern_reuse is not None
    metric_check = plan.pattern_reuse.check("metric_still_meaningful")
    assert metric_check is not None and metric_check.passed is False
    assert plan.pattern_reuse.reusable is False
    # A wrong metric replans the metric only; the dataset definition survives.
    assert plan.pattern_reuse.blocks_reuse is False
    assert "metric" in plan.pattern_reuse.recomputed_components
    assert "dataset_definition" in plan.pattern_reuse.reused_components
    assert plan.reuse.metric_revalidation.status == "rejected"
    assert plan.reuse.final_metric == "nearest_distance"


@pytest.mark.asyncio
async def test_invalid_dataset_definition_forces_a_fresh_plan(settings: Settings):
    """Step 3 says no: the whole analysis is replanned instead of reused."""
    snapshot = make_ab_snapshot().model_copy(update={"grounding_tags": ("amenity=cafe",)})
    repo = InMemoryExecutionMemoryRepository()
    await repo.save(snapshot)
    service = ExecutionMemoryService(repo, settings)

    pattern = derive_analysis_pattern(snapshot)
    reuse = validate_pattern_reuse(
        pattern,
        requested_labels=(_TEHRAN, _AMIRKABIR),
        feature_concept="public parks",
        inferred_goal="abundance",
        radius_m=2000,
    )
    assert reuse.blocks_reuse is True
    assert await service.prepare_incremental(_CONV, _TEHRAN_AMIRKABIR_MSG) is None


@pytest.mark.asyncio
async def test_pattern_selection_prefers_a_compatible_concept(settings: Settings):
    cafes = make_ab_snapshot(feature_concept="cafes").model_copy(
        update={"grounding_tags": ("amenity=cafe",)}
    )
    parks = make_ab_snapshot()
    repo = InMemoryExecutionMemoryRepository()
    await repo.save(cafes)
    await repo.save(parks)
    service = ExecutionMemoryService(repo, settings)

    history = await service.history(_CONV)
    assert len(history) == 2
    patterns = service.patterns_from(history)
    chosen = select_pattern(
        patterns,
        feature_concept="cafes",
        target_category="educational_facility",
    )
    assert chosen is not None
    assert chosen.dataset_tags == ("amenity=cafe",)


@pytest.mark.asyncio
async def test_first_question_of_a_conversation_has_no_pattern(settings: Settings):
    service = ExecutionMemoryService(InMemoryExecutionMemoryRepository(), settings)
    assert await service.prepare_incremental(_CONV, _TEHRAN_SHARIF_MSG) is None
    empty = validate_pattern_reuse(
        None,
        requested_labels=(_TEHRAN, _SHARIF),
        feature_concept="public parks",
        inferred_goal="abundance",
        radius_m=2000,
    )
    assert empty.reusable is False
    assert empty.blocks_reuse is False
    assert "dataset_definition" in empty.recomputed_components
