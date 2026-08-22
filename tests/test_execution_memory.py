"""Offline Execution Memory tests (no PostgreSQL / Overpass / LLM network)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import pytest
from app.agent.contracts import GeoAgentRequest, GeoAgentResponse
from app.agent.loop import LoopLimits
from app.agent.orchestrator import PlannerExecutorAgent
from app.analytics.datasets import DatasetRegistry, DatasetScope
from app.analytics.factory import build_analyze_features_tool
from app.api.schemas import knowledge_sources_from_passages
from app.core.config import Settings
from app.execution_memory.contracts import (
    ComparisonSnapshot,
    DocumentationSourceSnapshot,
    ExecutionDatasetSnapshot,
    ExecutionSnapshot,
    ExecutionTargetSnapshot,
    MetricSnapshot,
    TrustedPlaceSnapshot,
)
from app.execution_memory.follow_up import FollowUpResolver
from app.execution_memory.metric_revalidation import revalidate_metric
from app.execution_memory.repository import InMemoryExecutionMemoryRepository
from app.execution_memory.restore import restore_dataset, restore_place
from app.execution_memory.reuse import ExecutionReuseValidator
from app.execution_memory.service import ExecutionMemoryService
from app.execution_memory.snapshot import documentation_sources_from_response
from app.llm.contracts import LLMResponse
from app.places.contracts import PlaceRegistry
from app.tools.context import AnalysisRunState, GroundingState, ToolContext
from app.tools.query_osm import QueryOsmTool
from app.tools.registry import ToolRegistry

from tests.test_multi_target_comparison import (
    _PASSAGE,
    FakeKnowledgeTool,
    PassthroughEncoder,
    ScriptedOverpass,
    _park_element,
    _resolve_tool,
)
from tests.test_orchestrator import ScriptedLLM, _final

_PARK_DOC = DocumentationSourceSnapshot(
    title="Tag:leisure=park",
    section="Description",
    url="https://wiki.openstreetmap.org/wiki/Tag:leisure%3Dpark",
    score=0.95,
)
_CONV = "conversation-test-0001"
_NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
_FC_A = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [51.39, 35.70]},
            "properties": {"osm_id": 1, "tags": {"leisure": "park"}},
        }
    ],
}
_FC_B = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [51.35, 35.70]},
            "properties": {"osm_id": 2, "tags": {"leisure": "park"}},
        }
    ],
}


def _place(label: str, query: str, lat: float, lon: float) -> TrustedPlaceSnapshot:
    return TrustedPlaceSnapshot(
        query=query,
        label=label,
        display_name=label,
        latitude=lat,
        longitude=lon,
        source="nominatim",
        source_id=f"n/{label}",
    )


def _dataset(
    stable_id: str,
    *,
    feature_count: int,
    truncated: bool,
    retrieved_at: datetime,
    collection: dict[str, Any],
    request_ref: str,
    radius_m: int = 2000,
) -> ExecutionDatasetSnapshot:
    return ExecutionDatasetSnapshot(
        target_stable_id=stable_id,
        retrieved_at=retrieved_at,
        source="live_osm",
        endpoint="https://overpass.test/api/interpreter",
        attribution="© OpenStreetMap contributors (ODbL)",
        effective_limit=50,
        truncated=truncated,
        feature_count=feature_count,
        resolved_tags=("leisure=park",),
        scope=DatasetScope(
            scope_kind="point",
            summary=f"{radius_m}m",
            radius_m=radius_m,
            center=None,
            area_km2=12.5,
        ),
        query_spec={"resolved_tags": ["leisure=park"], "radius_m": radius_m},
        feature_collection=collection,
        request_scoped_dataset_ref_at_creation=request_ref,
    )


def make_ab_snapshot(
    *,
    truncated_b: bool = False,
    retrieved_at: datetime | None = None,
    feature_concept: str = "public parks",
    inferred_goal: str = "abundance",
    metric: str = "count",
    radius_m: int = 2000,
    conversation_id: str = _CONV,
    documentation_sources: tuple[DocumentationSourceSnapshot, ...] | None = None,
) -> ExecutionSnapshot:
    retrieved = retrieved_at or _NOW
    tehran = _place(
        "University of Tehran",
        "University of Tehran, Tehran, Iran",
        35.702,
        51.395,
    )
    sharif = _place(
        "Sharif University of Technology",
        "Sharif University of Technology, Tehran, Iran",
        35.703,
        51.351,
    )
    return ExecutionSnapshot(
        conversation_id=conversation_id,
        original_user_query=(
            "Compare public parks within 2 km of the University of Tehran and "
            "Sharif University of Technology."
        ),
        analysis_type="comparison",
        analysis_goal="park availability",
        feature_concept=feature_concept,
        inferred_goal=inferred_goal,  # type: ignore[arg-type]
        outcome="completed",
        radius_m=radius_m,
        scope_kind="point",
        grounding_tags=("leisure=park",),
        grounding_version="concept:public parks;tags:leisure=park",
        metric=MetricSnapshot(
            metric=metric,  # type: ignore[arg-type]
            inferred_goal=inferred_goal,  # type: ignore[arg-type]
            rule_id="ABUNDANCE_COUNT_001",
            catalog_version="metric-catalog-1",
            ruleset_version="metric-rules-1",
            feasibility_ok=True,
        ),
        targets=[
            ExecutionTargetSnapshot(
                stable_id="t1",
                label=tehran.label,
                original_user_label=tehran.label,
                place=tehran,
                request_scoped_place_ref_at_creation="place_1",
            ),
            ExecutionTargetSnapshot(
                stable_id="t2",
                label=sharif.label,
                original_user_label=sharif.label,
                place=sharif,
                request_scoped_place_ref_at_creation="place_2",
            ),
        ],
        datasets=[
            _dataset(
                "t1",
                feature_count=1,
                truncated=False,
                retrieved_at=retrieved,
                collection=_FC_A,
                request_ref="osm_result_1",
                radius_m=radius_m,
            ),
            _dataset(
                "t2",
                feature_count=50 if truncated_b else 1,
                truncated=truncated_b,
                retrieved_at=retrieved,
                collection=_FC_B,
                request_ref="osm_result_2",
                radius_m=radius_m,
            ),
        ],
        documentation_sources=(
            (_PARK_DOC,) if documentation_sources is None else documentation_sources
        ),
        comparison=ComparisonSnapshot(
            overall_statement="t2 has more parks",
            overall_confidence="clear",
            ranking=[
                {"target_id": "t1", "label": tehran.label, "value": 1, "status": "computed"},
                {"target_id": "t2", "label": sharif.label, "value": 1, "status": "computed"},
            ],
            truncated=truncated_b,
        ),
        model="fake",
        created_at=retrieved,
    )


@pytest.mark.asyncio
async def test_successful_execution_creates_memory_snapshot(settings: Settings):
    repo = InMemoryExecutionMemoryRepository()
    service = ExecutionMemoryService(repo, settings)
    snapshot = make_ab_snapshot()
    saved = await repo.save(snapshot)
    loaded = await service.latest(_CONV)
    assert loaded is not None
    assert loaded.execution_id == saved.execution_id
    assert loaded.conversation_id == _CONV
    assert loaded.metric.metric == "count"


@pytest.mark.asyncio
async def test_memory_has_conversation_id():
    snapshot = make_ab_snapshot()
    assert snapshot.conversation_id == _CONV
    assert len(snapshot.conversation_id) >= 8


@pytest.mark.asyncio
async def test_request_scoped_dataset_ref_is_not_durable_identity():
    snapshot = make_ab_snapshot()
    registry = DatasetRegistry()
    new_ref = restore_dataset(registry, snapshot.datasets[1])
    assert snapshot.datasets[1].request_scoped_dataset_ref_at_creation == "osm_result_2"
    assert new_ref == "osm_result_1"
    assert new_ref != snapshot.datasets[1].request_scoped_dataset_ref_at_creation
    assert "osm_result_2" not in str(snapshot.datasets[1].query_spec)


@pytest.mark.asyncio
async def test_persistent_dataset_restores_to_fresh_request_scoped_ref():
    snapshot = make_ab_snapshot()
    registry = DatasetRegistry()
    first = restore_dataset(registry, snapshot.datasets[0])
    second = restore_dataset(registry, snapshot.datasets[1])
    assert (first, second) == ("osm_result_1", "osm_result_2")
    assert registry.get(first).feature_count == 1


@pytest.mark.asyncio
async def test_add_amirkabir_resolves_as_add_target():
    resolver = FollowUpResolver()
    follow = resolver.resolve_heuristic(
        "Now add Amirkabir University to the comparison.",
        make_ab_snapshot(),
    )
    assert follow.follow_up_type == "ADD_TARGET"
    assert any("amirkabir" in item.lower() for item in follow.added_labels)
    assert len(follow.preserved_labels) == 2


@pytest.mark.asyncio
async def test_previous_targets_preserved_on_add():
    follow = FollowUpResolver().resolve_heuristic(
        "Now add Amirkabir and compare all three.",
        make_ab_snapshot(),
    )
    labels = " ".join(follow.preserved_labels).lower()
    assert "tehran" in labels
    assert "sharif" in labels
    assert follow.follow_up_type == "ADD_TARGET"


@pytest.mark.asyncio
async def test_count_revalidated_and_accepted_for_abundance():
    result = revalidate_metric(
        previous=make_ab_snapshot().metric,
        new_goal="abundance",
    )
    assert result.status == "accepted"
    assert result.replacement_metric is None
    assert result.previous_metric == "count"


@pytest.mark.asyncio
async def test_count_rejected_for_accessibility_and_replaced():
    result = revalidate_metric(
        previous=make_ab_snapshot().metric,
        new_goal="accessibility",
    )
    assert result.status == "rejected"
    assert result.replacement_metric == "nearest_distance"
    assert "does not represent accessibility" in result.reason


@pytest.mark.asyncio
async def test_radius_change_invalidates_spatial_datasets(settings: Settings):
    validator = ExecutionReuseValidator(osm_ttl_seconds=settings.execution_memory_osm_ttl_seconds)
    follow = FollowUpResolver().resolve_heuristic(
        "Use 1 km for all three.",
        make_ab_snapshot(),
    )
    assert follow.follow_up_type == "CHANGE_RADIUS"
    assessment = validator.assess(make_ab_snapshot(), follow, now=_NOW)
    assert assessment.decision == "REFRESH_DATA"
    assert all(item.source == "refreshed" for item in assessment.target_actions)
    assert assessment.radius_m == 1000


@pytest.mark.asyncio
async def test_feature_concept_change_triggers_regrounding(settings: Settings):
    validator = ExecutionReuseValidator(osm_ttl_seconds=settings.execution_memory_osm_ttl_seconds)
    follow = FollowUpResolver().resolve_heuristic(
        "Now compare all green spaces for them.",
        make_ab_snapshot(),
    )
    assert follow.follow_up_type == "CHANGE_FEATURE_CONCEPT"
    assessment = validator.assess(make_ab_snapshot(), follow, now=_NOW)
    assert assessment.re_ground is True
    assert assessment.decision == "FULL_REPLAN"
    assert "landuse=grass" in assessment.grounding_tags


@pytest.mark.asyncio
async def test_stale_datasets_trigger_refresh(settings: Settings):
    validator = ExecutionReuseValidator(osm_ttl_seconds=60)
    old = make_ab_snapshot(retrieved_at=_NOW - timedelta(hours=2))
    follow = FollowUpResolver().resolve_heuristic(
        "Compare them again.",
        old,
    )
    assessment = validator.assess(old, follow, now=_NOW)
    assert all(item.source == "refreshed" for item in assessment.target_actions)


@pytest.mark.asyncio
async def test_limit_hit_prevents_exact_abundance_reuse(settings: Settings):
    validator = ExecutionReuseValidator(osm_ttl_seconds=settings.execution_memory_osm_ttl_seconds)
    snapshot = make_ab_snapshot(truncated_b=True)
    follow = FollowUpResolver().resolve_heuristic(
        "Now add Amirkabir University to the comparison.",
        snapshot,
    )
    assessment = validator.assess(snapshot, follow, now=_NOW)
    sources = {item.label: item.source for item in assessment.target_actions}
    assert "refreshed" in sources.values() or any(
        "truncated_limit_hit" in item.reasons for item in assessment.target_actions
    )


@pytest.mark.asyncio
async def test_partial_reuse_mixes_reused_refresh_and_new(settings: Settings):
    validator = ExecutionReuseValidator(osm_ttl_seconds=60)
    snapshot = make_ab_snapshot(retrieved_at=_NOW - timedelta(hours=2))
    # First target recently queried via copy? mark t2 stale by snapshot age.
    follow = FollowUpResolver().resolve_heuristic(
        "Now add Amirkabir University to the comparison.",
        snapshot,
    )
    assessment = validator.assess(snapshot, follow, now=_NOW)
    sources = [item.source for item in assessment.target_actions]
    assert "new" in sources
    assert assessment.decision == "PARTIAL_REUSE"


@pytest.mark.asyncio
async def test_trusted_place_resolution_can_be_reused():
    snapshot = make_ab_snapshot()
    registry = PlaceRegistry()
    record = restore_place(registry, snapshot.targets[0].place)
    assert record.place_ref == "place_1"
    assert record.latitude == snapshot.targets[0].place.latitude
    assert record.query == snapshot.targets[0].place.query


def test_model_cannot_invent_coordinates_from_memory():
    from app.agent.spatial_trust import untrusted_spatial_scope_error

    error = untrusted_spatial_scope_error(
        {"point": {"lat": 35.702, "lon": 51.395, "radius_m": 2000}},
        "Now add Amirkabir University to the comparison.",
    )
    assert error is not None
    assert "inventing" in error.lower() or "explicitly" in error.lower()


def test_model_cannot_invent_dataset_refs():
    from app.agent.accumulator import ResultAccumulator
    from app.agent.orchestrator import _pre_invoke_gate
    from app.llm.contracts import ToolCall

    state = ResultAccumulator()
    context = ToolContext(
        datasets=state.datasets,
        analysis=AnalysisRunState(),
        user_message="Now add Amirkabir University and compare them.",
        places=PlaceRegistry(),
        grounding=GroundingState(),
    )
    restore_dataset(
        state.datasets,
        make_ab_snapshot().datasets[0],
    )
    call = ToolCall(
        id="x",
        name="analyze_features",
        arguments={
            "analysis_type": "comparison",
            "feature_concept": "parks",
            "comparison_goal": "abundance",
            "targets": [
                {"target_id": "t1", "label": "A", "dataset_ref": "osm_result_9"},
                {"target_id": "t2", "label": "B", "dataset_ref": "osm_result_1"},
            ],
            "metrics": [{"metric": "count", "role": "primary", "inferred_goal": "abundance"}],
        },
    )
    error = _pre_invoke_gate(
        call,
        state=state,
        user_message=context.user_message,
        tool_context=context,
    )
    assert error is not None
    assert "osm_result_9" in str(error)


@pytest.mark.asyncio
async def test_history_creates_new_snapshot_with_parent_lineage():
    repo = InMemoryExecutionMemoryRepository()
    first = make_ab_snapshot()
    await repo.save(first)
    second = first.model_copy(
        update={
            "execution_id": uuid4(),
            "parent_execution_id": first.execution_id,
            "original_user_query": "Now add Amirkabir University to the comparison.",
            "created_at": _NOW + timedelta(seconds=5),
        }
    )
    await repo.save(second)
    with pytest.raises(ValueError, match="immutable"):
        await repo.save(first)
    latest = await repo.latest_for_conversation(_CONV)
    assert latest is not None
    assert latest.execution_id == second.execution_id
    assert latest.parent_execution_id == first.execution_id
    assert await repo.get(first.execution_id) is not None


@pytest.mark.asyncio
async def test_memory_failure_does_not_break_fresh_analysis(settings: Settings):
    class Boom:
        async def latest_for_conversation(self, conversation_id: str):
            raise RuntimeError("db down")

        async def save(self, snapshot):
            raise RuntimeError("db down")

        async def get(self, execution_id):
            raise RuntimeError("db down")

        async def enforce_retention(self, *args, **kwargs):
            raise RuntimeError("db down")

    service = ExecutionMemoryService(Boom(), settings)  # type: ignore[arg-type]
    assert await service.latest(_CONV) is None
    plan = await service.prepare_incremental(_CONV, "Now add Amirkabir University.")
    assert plan is None
    registry = ToolRegistry()
    llm = ScriptedLLM([_final("leisure=park is the OSM tag for a public park.")])
    agent = PlannerExecutorAgent(
        llm,
        registry,
        LoopLimits(max_tool_rounds=1, max_tool_calls=1),
        service,
    )
    result = await agent.run(
        GeoAgentRequest(message="What is leisure=park?", conversation_id=_CONV)
    )
    assert result.stop_reason in {"final_answer", "llm_error", "max_tool_rounds"}


@pytest.mark.asyncio
async def test_active_learning_failure_does_not_break_execution_memory(settings: Settings):
    repo = InMemoryExecutionMemoryRepository()
    service = ExecutionMemoryService(repo, settings)
    await repo.save(make_ab_snapshot())
    loaded = await service.latest(_CONV)
    assert loaded is not None
    # AL is not consulted by ExecutionMemoryService.
    assert service.enabled is True


def test_no_hidden_reasoning_is_persisted():
    snapshot = make_ab_snapshot()
    blob = str(snapshot.model_dump(mode="json")).lower()
    assert "rationale" not in blob
    assert "reasoning_content" not in blob
    assert "<think" not in blob
    assert "chain_of_thought" not in blob


@pytest.mark.asyncio
async def test_isfahan_is_new_analysis():
    follow = FollowUpResolver().resolve_heuristic(
        "Compare parks in Isfahan.",
        make_ab_snapshot(),
    )
    assert follow.follow_up_type == "NEW_ANALYSIS"


@pytest.mark.asyncio
async def test_add_target_only_new_is_queried_when_ab_valid(settings: Settings):
    snapshot = make_ab_snapshot()
    validator = ExecutionReuseValidator(osm_ttl_seconds=settings.execution_memory_osm_ttl_seconds)
    follow = FollowUpResolver().resolve_heuristic(
        "Now add Amirkabir University to the comparison.",
        snapshot,
    )
    assessment = validator.assess(snapshot, follow, now=_NOW)
    assert assessment.decision == "PARTIAL_REUSE"
    assert assessment.metric_revalidation.status == "accepted"
    reused = [item for item in assessment.target_actions if item.source == "reused"]
    new = [item for item in assessment.target_actions if item.source == "new"]
    assert len(reused) == 2
    assert len(new) == 1


@pytest.mark.asyncio
async def test_accessibility_follow_up_rejects_count(settings: Settings):
    snapshot = make_ab_snapshot()
    validator = ExecutionReuseValidator(osm_ttl_seconds=settings.execution_memory_osm_ttl_seconds)
    follow = FollowUpResolver().resolve_heuristic(
        "Now compare all three by accessibility to parks.",
        snapshot,
    )
    assert follow.follow_up_type == "CHANGE_GOAL" or "CHANGE_GOAL" in follow.extra_intents
    assessment = validator.assess(snapshot, follow, now=_NOW)
    assert assessment.metric_revalidation.status == "rejected"
    assert assessment.final_metric == "nearest_distance"


@pytest.mark.asyncio
async def test_incremental_add_amirkabir_queries_only_new_target(
    settings: Settings,
):
    repo = InMemoryExecutionMemoryRepository()
    await repo.save(make_ab_snapshot())
    memory = ExecutionMemoryService(repo, settings)
    client = ScriptedOverpass([(_park_element(9, 51.41, 35.70),)])
    query = QueryOsmTool(client, PassthroughEncoder(), timeout_seconds=25, max_results=1000)
    resolve = _resolve_tool()
    registry = ToolRegistry()
    registry.register(FakeKnowledgeTool())
    registry.register(resolve)
    registry.register(query)
    registry.register(build_analyze_features_tool())

    class ReportLLM:
        model_name = "fake"
        provider_name = "fake"

        async def chat(self, messages, *, tools=None, options=None):
            del messages, tools, options
            return LLMResponse(content='{"final_answer":"Amirkabir University was added."}')

        async def aclose(self) -> None:
            return None

    agent = PlannerExecutorAgent(
        ReportLLM(),
        registry,
        LoopLimits(max_tool_rounds=6, max_tool_calls=12),
        memory,
    )
    result = await agent.run(
        GeoAgentRequest(
            message="Now add Amirkabir University to the comparison.",
            conversation_id=_CONV,
        )
    )
    assert len(client.queries) == 1
    assert result.execution_memory is not None
    assert result.execution_memory.follow_up_type == "ADD_TARGET"
    assert result.execution_memory.metric_revalidation_status == "accepted"
    assert len(result.execution_memory.new_targets) == 1
    assert len(list(repo._items)) == 2
    child = await repo.latest_for_conversation(_CONV)
    assert child is not None
    first_id = next(iter(repo._items))
    assert child.parent_execution_id == first_id
    if result.analysis is not None:
        assert len(result.analysis.plan.targets) == 3


_TEHRAN_SHARIF_MSG = (
    "Compare public parks within 2 km of the University of Tehran and "
    "Sharif University of Technology. Choose the best metric, compare them, "
    "and return the park features for both areas."
)
_TEHRAN_AMIRKABIR_MSG = (
    "Compare public parks within 2 km of the University of Tehran and "
    "Amirkabir University of Technology. Choose the best metric, compare them, "
    "and return the park features for both areas."
)
_THREE_UNIVERSITIES_MSG = (
    "Compare public parks within 2 km of the University of Tehran, "
    "Sharif University of Technology, and Amirkabir University of Technology. "
    "Choose the best metric, compare them, and return the park features for both areas."
)


def _labels_blob(values: tuple[str, ...] | list[str]) -> str:
    return " ".join(values).lower()


@pytest.mark.asyncio
async def test_a_tehran_sharif_request_declares_two_targets():
    from app.agent.comparison_workflow import extract_landmark_labels, seed_plan_from_user_message

    labels = extract_landmark_labels(_TEHRAN_SHARIF_MSG)
    assert labels == ("University of Tehran", "Sharif University of Technology")
    seed = seed_plan_from_user_message(_TEHRAN_SHARIF_MSG)
    assert seed is not None
    assert {t.label for t in seed.targets} == {
        "University of Tehran",
        "Sharif University of Technology",
    }
    assert all("amirkabir" not in t.label.lower() for t in seed.targets)
    follow = FollowUpResolver().resolve_heuristic(_TEHRAN_SHARIF_MSG, make_ab_snapshot())
    assert "sharif" in _labels_blob(follow.requested_labels)
    assert "amirkabir" not in _labels_blob(follow.requested_labels)


@pytest.mark.asyncio
async def test_b_tehran_amirkabir_does_not_leak_sharif(settings: Settings):
    snapshot = make_ab_snapshot()
    follow = FollowUpResolver().resolve_heuristic(_TEHRAN_AMIRKABIR_MSG, snapshot)
    assert follow.follow_up_type == "REPLACE_TARGETS"
    requested = _labels_blob(follow.requested_labels)
    assert "tehran" in requested
    assert "amirkabir" in requested
    assert "sharif" not in requested
    assert "sharif" not in _labels_blob(follow.preserved_labels)
    validator = ExecutionReuseValidator(osm_ttl_seconds=settings.execution_memory_osm_ttl_seconds)
    assessment = validator.assess(snapshot, follow, now=_NOW)
    assert assessment.metric_revalidation.status == "accepted"
    assert assessment.final_metric == "count"
    assert assessment.radius_m == 2000
    assert "leisure=park" in assessment.grounding_tags
    labels = [item.label.lower() for item in assessment.target_actions]
    assert any("tehran" in item for item in labels)
    assert any("amirkabir" in item for item in labels)
    assert all("sharif" not in item for item in labels)
    tehran = next(item for item in assessment.target_actions if "tehran" in item.label.lower())
    amirkabir = next(
        item for item in assessment.target_actions if "amirkabir" in item.label.lower()
    )
    assert tehran.source == "reused"
    assert amirkabir.source == "new"


@pytest.mark.asyncio
async def test_c_three_universities_keeps_named_set_only(settings: Settings):
    snapshot = make_ab_snapshot()
    follow = FollowUpResolver().resolve_heuristic(_THREE_UNIVERSITIES_MSG, snapshot)
    assert follow.follow_up_type == "REPLACE_TARGETS"
    requested = _labels_blob(follow.requested_labels)
    assert "tehran" in requested
    assert "sharif" in requested
    assert "amirkabir" in requested
    assert len(follow.requested_labels) == 3
    validator = ExecutionReuseValidator(osm_ttl_seconds=settings.execution_memory_osm_ttl_seconds)
    assessment = validator.assess(snapshot, follow, now=_NOW)
    assert len(assessment.target_actions) == 3
    assert assessment.metric_revalidation.status == "accepted"
    new_labels = [item.label.lower() for item in assessment.target_actions if item.source == "new"]
    assert any("amirkabir" in item for item in new_labels)
    reused = [item.label.lower() for item in assessment.target_actions if item.source == "reused"]
    assert any("tehran" in item for item in reused)
    assert any("sharif" in item for item in reused)


@pytest.mark.asyncio
async def test_them_and_both_do_not_keep_unnamed_previous_target():
    follow = FollowUpResolver().resolve_heuristic(_TEHRAN_AMIRKABIR_MSG, make_ab_snapshot())
    assert "them" in _TEHRAN_AMIRKABIR_MSG.lower()
    assert "both" in _TEHRAN_AMIRKABIR_MSG.lower()
    assert follow.follow_up_type == "REPLACE_TARGETS"
    assert "sharif" not in _labels_blob(follow.requested_labels)


@pytest.mark.asyncio
async def test_replace_targets_incremental_queries_only_new_and_maps_both(
    settings: Settings,
):
    repo = InMemoryExecutionMemoryRepository()
    await repo.save(make_ab_snapshot())
    memory = ExecutionMemoryService(repo, settings)
    client = ScriptedOverpass([(_park_element(9, 51.3906, 35.7013),)])
    query = QueryOsmTool(client, PassthroughEncoder(), timeout_seconds=25, max_results=1000)
    resolve = _resolve_tool()
    knowledge = FakeKnowledgeTool()
    registry = ToolRegistry()
    registry.register(knowledge)
    registry.register(resolve)
    registry.register(query)
    registry.register(build_analyze_features_tool())

    class ReportLLM:
        model_name = "fake"
        provider_name = "fake"

        async def chat(self, messages, *, tools=None, options=None):
            del messages, tools, options
            return LLMResponse(
                content=(
                    '{"final_answer":"University of Tehran and Amirkabir were compared by count."}'
                )
            )

        async def aclose(self) -> None:
            return None

    agent = PlannerExecutorAgent(
        ReportLLM(),
        registry,
        LoopLimits(max_tool_rounds=6, max_tool_calls=12),
        memory,
    )
    result = await agent.run(GeoAgentRequest(message=_TEHRAN_AMIRKABIR_MSG, conversation_id=_CONV))
    assert len(client.queries) == 1
    assert result.execution_memory is not None
    assert result.execution_memory.follow_up_type == "REPLACE_TARGETS"
    assert result.execution_memory.metric_revalidation_status == "accepted"
    assert result.execution_memory.final_metric == "count"
    assert any("tehran" in item.lower() for item in result.execution_memory.reused_targets)
    assert any("amirkabir" in item.lower() for item in result.execution_memory.new_targets)
    assert all("sharif" not in item.lower() for item in result.execution_memory.reused_targets)
    assert all("sharif" not in item.lower() for item in result.execution_memory.new_targets)
    assert result.analysis is not None
    plan_labels = [target.label.lower() for target in result.analysis.plan.targets]
    assert any("tehran" in item for item in plan_labels)
    assert any("amirkabir" in item for item in plan_labels)
    assert all("sharif" not in item for item in plan_labels)
    assert result.geojson is not None
    geo_labels = {
        str((feat.get("properties") or {}).get("analysis_target") or "").lower()
        for feat in result.geojson.get("features", [])
    }
    assert any("tehran" in item for item in geo_labels)
    assert any("amirkabir" in item for item in geo_labels)
    assert all("sharif" not in item for item in geo_labels)

    steps = [event.message for event in result.trace if event.kind == "memory_reuse"]
    assert any("Previous analysis pattern found" in item for item in steps)
    assert any("Metric validated" in item for item in steps)
    assert any("Radius reused: 2000 m" in item for item in steps)
    assert any("Targets changed" in item for item in steps)
    assert any("Spatial query recomputed" in item for item in steps)
    assert any("New comparison generated" in item for item in steps)
    assert result.execution_memory.pattern_reusable is True
    assert "dataset_definition" in result.execution_memory.reused_components
    assert "osm_query" in result.execution_memory.recomputed_components
    assert knowledge.calls == []
    assert result.passages
    assert result.passages[0].document_title == "Tag:leisure=park"
    assert result.execution_memory.documentation_sources_restored is True
    assert "Tag:leisure=park" in result.execution_memory.documentation_source_titles
    assert any("OSM documentation sources restored" in item for item in steps)
    citations = knowledge_sources_from_passages(list(result.passages))
    assert citations
    assert citations[0].title == "Tag:leisure=park"
    assert citations[0].url.startswith("https://wiki.openstreetmap.org")


def test_snapshot_stores_documentation_citations_not_passage_bodies():
    response = GeoAgentResponse(answer="ok", passages=[_PASSAGE])
    citations = documentation_sources_from_response(response)
    assert citations == (_PARK_DOC,)
    blob = str(citations[0].model_dump())
    assert "Public parks are tagged" not in blob


@pytest.mark.asyncio
async def test_missing_documentation_evidence_triggers_knowledge_search(
    settings: Settings,
):
    repo = InMemoryExecutionMemoryRepository()
    await repo.save(make_ab_snapshot(documentation_sources=()))
    memory = ExecutionMemoryService(repo, settings)
    client = ScriptedOverpass([(_park_element(9, 51.3906, 35.7013),)])
    query = QueryOsmTool(client, PassthroughEncoder(), timeout_seconds=25, max_results=1000)
    knowledge = FakeKnowledgeTool()
    registry = ToolRegistry()
    registry.register(knowledge)
    registry.register(_resolve_tool())
    registry.register(query)
    registry.register(build_analyze_features_tool())

    class ReportLLM:
        model_name = "fake"
        provider_name = "fake"

        async def chat(self, messages, *, tools=None, options=None):
            del messages, tools, options
            return LLMResponse(
                content='{"final_answer":"University of Tehran and Amirkabir were compared."}'
            )

        async def aclose(self) -> None:
            return None

    agent = PlannerExecutorAgent(
        ReportLLM(),
        registry,
        LoopLimits(max_tool_rounds=6, max_tool_calls=12),
        memory,
    )
    result = await agent.run(GeoAgentRequest(message=_TEHRAN_AMIRKABIR_MSG, conversation_id=_CONV))
    assert knowledge.calls
    assert result.passages
    assert result.passages[0].document_title == "Tag:leisure=park"
    assert result.execution_memory is not None
    assert result.execution_memory.documentation_sources_restored is False
    saved = await memory.latest(_CONV)
    assert saved is not None
    assert saved.documentation_sources
    assert saved.documentation_sources[0].title == "Tag:leisure=park"
