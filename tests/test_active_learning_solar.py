"""SolarDSS Active Learning: routing/preference labels, never PV physics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from app.active_learning.contracts import (
    ExportTask,
    FeedbackSentiment,
    FeedbackSubmission,
    ReviewDecision,
    ReviewStatus,
    SelectionReason,
    TaskType,
)
from app.active_learning.export import export_dataset
from app.active_learning.repository import InMemoryCandidateRepository
from app.active_learning.scoring import REASON_WEIGHTS
from app.active_learning.service import ActiveLearningService, SelectionPolicy
from app.active_learning.signals import extract_run_signals
from app.active_learning.solar import (
    DOES_NOT_REPLACE,
    SCIENCE_NOT_COMPUTED,
    SolarDecisionLearningTarget,
    SolarExpertFeedback,
    SolarFeedbackKind,
    SolarRankingEntry,
    learning_target_from_workflow,
    physics_keys_in_payload,
    snapshot_solar_workflow,
)
from app.agent.contracts import GeoAgentResponse
from app.core.config import Settings
from app.main import create_app
from app.solar.contracts import SolarDecisionReport, SolarFollowUpQuestion
from pydantic import ValidationError

from tests.active_learning_fixtures import failing_run, successful_run, trace
from tests.test_active_learning_boundaries import FakeAgent, build_services
from tests.test_active_learning_selection import build_service

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTED_AT = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
RANK_QUERY = "Rank these solar rooftops A, B and C in Tehran."
SOLAR_SITE_QUERY = "Find a good solar farm site in Yazd."


def solar_ranking_report(**updates: object) -> SolarDecisionReport:
    payload: dict[str, object] = {
        "status": "routed",
        "mode": "site_selection",
        "site_selection_scope": "candidate_ranking",
        "project_type": "rooftop",
        "capability_id": "solar_candidate_ranking",
        "problem": {
            "mode": "site_selection",
            "site_selection_scope": "candidate_ranking",
            "project_type": "rooftop",
            "candidates": {"candidate_ids": ["A", "B", "C"]},
            "objective": "rank rooftop PV sites",
        },
        "filled_fields": ("mode", "site_selection_scope", "candidates", "capability_id"),
        "requirements_source": "host_mirror",
    }
    payload.update(updates)
    return SolarDecisionReport.model_validate(payload)


def solar_ranking_run(
    *,
    report: SolarDecisionReport | None = None,
    extra_events: list[object] | None = None,
) -> GeoAgentResponse:
    events = [
        trace("tool_call", "calling analyze_solar_site", tool="analyze_solar_site"),
        trace(
            "tool_result",
            "routed solar_candidate_ranking",
            tool="analyze_solar_site",
        ),
    ]
    if extra_events:
        events.extend(extra_events)  # type: ignore[arg-type]
    return GeoAgentResponse(
        answer="Candidate A is ranked first among the supplied rooftops.",
        trace=events,  # type: ignore[arg-type]
        solar_decision=report or solar_ranking_report(),
        stop_reason="final_answer",
        model="deepseek-r1:7b",
    )


def reject_candidate_a() -> SolarExpertFeedback:
    return SolarExpertFeedback(
        kind=SolarFeedbackKind.CANDIDATE_ACCEPT_REJECT,
        labels=[
            SolarRankingEntry(
                candidate_id="A",
                original_rank=1,
                decision="reject",
                reason="Grid connection assumption is unrealistic",
            )
        ],
        reason="Grid connection assumption is unrealistic",
    )


def test_every_solar_feedback_kind_has_a_selection_reason() -> None:
    from app.active_learning.service import _SOLAR_FEEDBACK_REASONS

    assert set(_SOLAR_FEEDBACK_REASONS) == set(SolarFeedbackKind)


def test_every_selection_reason_has_a_weight() -> None:
    missing = [reason for reason in SelectionReason if reason not in REASON_WEIGHTS]
    assert missing == []


def test_learning_target_rejects_pvgis_and_ghi_payloads() -> None:
    with pytest.raises(ValidationError):
        SolarDecisionLearningTarget.model_validate({"science": "not_computed", "ghi": 1800})
    with pytest.raises(ValidationError):
        SolarDecisionLearningTarget.model_validate(
            {"science": "not_computed", "preference": {"pvgis": {"annual_yield": 1400}}}
        )
    with pytest.raises(ValidationError):
        SolarDecisionLearningTarget.model_validate(
            {"science": "not_computed", "preference": {"mcda": 0.9}}
        )


def test_snapshot_copies_leaked_indicators_and_drops_them_from_the_target() -> None:
    report = solar_ranking_report(
        problem={
            "mode": "site_selection",
            "site_selection_scope": "candidate_ranking",
            "candidates": {"candidate_ids": ["A"]},
            "ghi": 1820.5,
            "slope": 8.0,
        }
    )

    snapshot = snapshot_solar_workflow(report, tool_sequence=("analyze_solar_site",))
    assert snapshot is not None
    assert "ghi" not in snapshot.structured_decision
    assert snapshot.indicators.science == "solardss_copy"
    assert snapshot.indicators.values["ghi"] == 1820.5
    target = learning_target_from_workflow(snapshot)
    blob = json.dumps(target.model_dump(mode="json"))
    assert "1820.5" not in blob
    assert '"ghi"' not in blob
    assert target.science == SCIENCE_NOT_COMPUTED
    assert "PVGIS" in target.does_not_replace


def test_solar_ranking_run_is_a_successful_solar_decision() -> None:
    response = solar_ranking_run()
    signals = extract_run_signals(RANK_QUERY, response)
    assert signals.task_type is TaskType.SOLAR_DECISION
    assert signals.outcome_successful
    assert SelectionReason.SUCCESSFUL_HIGH_VALUE_TRACE in signals.reasons
    assert signals.solar_capability == "solar_candidate_ranking"
    assert signals.scope_kind == "candidate_ranking"


def test_asking_ghi_is_a_question_selection_issue() -> None:
    report = solar_ranking_report(
        status="needs_input",
        questions=(
            SolarFollowUpQuestion(
                field="ghi",
                prompt="What is the annual GHI at this roof?",
                required=True,
            ),
        ),
    )
    signals = extract_run_signals(RANK_QUERY, solar_ranking_run(report=report))
    assert SelectionReason.SOLAR_QUESTION_SELECTION_ISSUE in signals.reasons
    assert SelectionReason.GUARDRAIL_TRIGGERED in signals.reasons


def test_solar_siting_via_query_osm_is_an_intent_misroute() -> None:
    signals = extract_run_signals(SOLAR_SITE_QUERY, successful_run())
    assert signals.task_type is TaskType.SOLAR_DECISION
    assert not signals.outcome_successful
    assert SelectionReason.SOLAR_INTENT_MISROUTED in signals.reasons
    assert SelectionReason.WRONG_OR_INELIGIBLE_TOOL in signals.reasons


def test_unknown_solar_capability_is_a_model_selection_mistake() -> None:
    response = failing_run(
        error_code="solar_unknown_capability",
        message="capability_id is not in the SolarDSS registry",
        tool="analyze_solar_site",
    )
    signals = extract_run_signals(SOLAR_SITE_QUERY, response)
    assert SelectionReason.SOLAR_CAPABILITY_MISSELECTED in signals.reasons


async def test_observe_run_stores_the_solar_snapshot_without_mutating_indicators() -> None:
    service, repository = build_service()
    report = solar_ranking_report()
    response = solar_ranking_run(report=report)
    report_id = id(response.solar_decision)
    problem_before = dict(report.problem)

    candidate = await service.observe_run(
        request_id="req-solar-1",
        user_query=RANK_QUERY,
        response=response,
    )

    assert candidate is not None
    assert id(response.solar_decision) == report_id
    assert response.solar_decision is not None
    assert dict(response.solar_decision.problem) == problem_before
    workflow = candidate.solar_workflow
    assert workflow is not None
    assert workflow.structured_decision["candidates"]["candidate_ids"] == ["A", "B", "C"]
    assert workflow.indicators.science == SCIENCE_NOT_COMPUTED
    assert workflow.indicators.values == {}
    assert workflow.ranking_profile.candidate_ids == ["A", "B", "C"]
    assert workflow.model_planning_decision.capability_id == "solar_candidate_ranking"
    assert workflow.model_planning_decision.tool_sequence == ["analyze_solar_site"]
    assert workflow.provenance.science_not_from_active_learning is True
    assert workflow.provenance.does_not_replace == DOES_NOT_REPLACE
    target = json.loads(candidate.model_output or "{}")
    assert target["science"] == SCIENCE_NOT_COMPUTED
    assert "ghi" not in target
    assert await repository.get(candidate.candidate_id) is not None


async def test_expert_reject_of_rank_one_is_stored_and_exportable() -> None:
    service, _repository = build_service()
    stored = await service.observe_run(
        request_id="req-solar-reject",
        user_query=RANK_QUERY,
        response=solar_ranking_run(),
    )
    assert stored is not None

    updated = await service.record_feedback(
        FeedbackSubmission(
            request_id="req-solar-reject",
            sentiment=FeedbackSentiment.NEGATIVE,
            solar_feedback=reject_candidate_a(),
        )
    )
    assert updated is not None
    assert updated.approved_for_training is False
    assert SelectionReason.SOLAR_CANDIDATE_ACCEPT_REJECT in updated.selection_reasons
    assert SelectionReason.HUMAN_CORRECTION_AVAILABLE in updated.selection_reasons
    workflow = updated.solar_workflow
    assert workflow is not None
    assert workflow.expert_feedback is not None
    assert workflow.expert_feedback.labels[0].candidate_id == "A"
    assert workflow.expert_feedback.labels[0].original_rank == 1
    assert workflow.expert_feedback.labels[0].decision == "reject"
    assert workflow.expert_feedback.labels[0].reason == (
        "Grid connection assumption is unrealistic"
    )
    assert updated.corrected_output is not None
    assert updated.corrected_output["ranking_labels"][0]["decision"] == "reject"
    assert physics_keys_in_payload(updated.corrected_output) == []

    approved = await service.review(
        updated.candidate_id,
        ReviewDecision(status=ReviewStatus.APPROVED, reviewer="expert"),
    )
    export = export_dataset(
        [approved],
        dataset_version="v1",
        task=ExportTask.SOLAR_DECISION,
        exported_at=EXPORTED_AT,
    )
    assert export.manifest.example_count == 1
    assert export.manifest.target_schema_name == "SolarDecisionLearningTarget"
    entry = export.manifest.lineage[0]
    sample = export.splits[entry.split.value][0]
    assert "Do not compute or copy PVGIS" in sample.messages[0].content
    target = json.loads(sample.messages[1].content)
    SolarDecisionLearningTarget.model_validate(target)
    assert target["science"] == SCIENCE_NOT_COMPUTED
    assert target["ranking_labels"][0]["candidate_id"] == "A"
    assert target["ranking_labels"][0]["decision"] == "reject"
    assert target["ranking_labels"][0]["reason"] == ("Grid connection assumption is unrealistic")
    assert approved.solar_workflow is not None
    assert approved.solar_workflow.indicators.science == SCIENCE_NOT_COMPUTED
    assert physics_keys_in_payload(target) == []


async def test_preference_and_missing_data_feedback_map_to_bounded_reasons() -> None:
    service, _repository = build_service()
    stored = await service.observe_run(
        request_id="req-solar-pref",
        user_query=RANK_QUERY,
        response=solar_ranking_run(),
    )
    assert stored is not None
    updated = await service.record_feedback(
        FeedbackSubmission(
            request_id="req-solar-pref",
            sentiment=FeedbackSentiment.NEGATIVE,
            solar_feedback=SolarExpertFeedback(
                kind=SolarFeedbackKind.PREFERENCE_CORRECTION,
                preference_label="grid_connection_realism",
                corrected_preference={"grid_connection_assumption": "must_be_realistic"},
                reason="Grid connection assumption is unrealistic",
            ),
        )
    )
    assert updated is not None
    assert SelectionReason.SOLAR_PREFERENCE_CORRECTION in updated.selection_reasons
    assert updated.solar_workflow is not None
    assert updated.solar_workflow.corrected_preference == {
        "grid_connection_assumption": "must_be_realistic"
    }

    missing = await service.record_feedback(
        FeedbackSubmission(
            request_id="req-solar-pref",
            sentiment=FeedbackSentiment.NEGATIVE,
            solar_feedback=SolarExpertFeedback(
                kind=SolarFeedbackKind.MISSING_DATA,
                corrected_question_fields=["budget_amount"],
            ),
        )
    )
    assert missing is not None
    assert SelectionReason.SOLAR_QUESTION_SELECTION_ISSUE in missing.selection_reasons


async def test_ranking_usefulness_feedback_is_collected() -> None:
    service, _repository = build_service()
    stored = await service.observe_run(
        request_id="req-solar-useful",
        user_query=RANK_QUERY,
        response=solar_ranking_run(),
    )
    assert stored is not None
    updated = await service.record_feedback(
        FeedbackSubmission(
            request_id="req-solar-useful",
            sentiment=FeedbackSentiment.NEGATIVE,
            solar_feedback=SolarExpertFeedback(
                kind=SolarFeedbackKind.RANKING_USEFULNESS,
                ranking_useful=False,
                reason="Top-ranked site ignores grid-connection realism",
            ),
        )
    )
    assert updated is not None
    assert SelectionReason.SOLAR_RANKING_USEFULNESS_FEEDBACK in updated.selection_reasons
    assert updated.solar_workflow is not None
    assert updated.solar_workflow.expert_feedback is not None
    assert updated.solar_workflow.expert_feedback.ranking_useful is False


def test_expert_preference_cannot_smuggle_ghi() -> None:
    with pytest.raises(ValidationError):
        SolarExpertFeedback(
            kind=SolarFeedbackKind.PREFERENCE_CORRECTION,
            corrected_preference={"ghi": 1900},
        )


async def test_analysis_plan_export_skips_solar_candidates() -> None:
    service, _repository = build_service()
    stored = await service.observe_run(
        request_id="req-solar-skip",
        user_query=RANK_QUERY,
        response=solar_ranking_run(),
    )
    assert stored is not None
    approved = await service.review(
        stored.candidate_id,
        ReviewDecision(status=ReviewStatus.APPROVED, reviewer="expert"),
    )
    export = export_dataset(
        [approved],
        dataset_version="v1",
        task=ExportTask.ANALYSIS_PLAN,
        exported_at=EXPORTED_AT,
    )
    assert export.manifest.example_count == 0


def test_solar_workflow_migration_revises_execution_memory() -> None:
    path = REPO_ROOT / "alembic" / "versions" / "20260919_0004_active_learning_solar_workflow.py"
    source = path.read_text(encoding="utf-8")
    assert "solar_workflow" in source
    config = Config(str(REPO_ROOT / "alembic.ini"))
    script = ScriptDirectory.from_config(config)
    assert len(script.get_heads()) == 1
    revision = script.get_revision("20260919_0004")
    assert revision is not None
    assert revision.down_revision == "20260821_0003"


@pytest.fixture
def api_settings() -> Settings:
    return Settings(
        app_env="test",
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:5433/osm_geoagent_test",
        ollama_model="deepseek-r1:7b",
        agent_request_timeout_seconds=5.0,
        _env_file=None,  # type: ignore[call-arg]
    )


async def test_solar_feedback_endpoint_never_approves_or_trains(api_settings: Settings) -> None:
    repository = InMemoryCandidateRepository()
    service = ActiveLearningService(repository, policy=SelectionPolicy())
    app = create_app(
        api_settings,
        services=build_services(api_settings, FakeAgent(solar_ranking_run()), service),
    )
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        query = await http.post("/api/v1/agent/query", json={"message": RANK_QUERY})
        request_id = query.json()["request_id"]
        feedback = await http.post(
            "/api/v1/agent/feedback",
            json={
                "request_id": request_id,
                "sentiment": "negative",
                "solar_feedback": {
                    "kind": "candidate_accept_reject",
                    "labels": [
                        {
                            "candidate_id": "A",
                            "original_rank": 1,
                            "decision": "reject",
                            "reason": "Grid connection assumption is unrealistic",
                        }
                    ],
                    "reason": "Grid connection assumption is unrealistic",
                },
            },
        )

    assert query.status_code == 200
    assert feedback.status_code == 200
    body = feedback.json()
    assert body["accepted"] is True
    assert body["review_status"] == "pending"
    candidate = await repository.get_by_request(request_id)
    assert candidate is not None
    assert candidate.approved_for_training is False
    assert candidate.task_type is TaskType.SOLAR_DECISION
    assert candidate.solar_workflow is not None
    assert candidate.solar_workflow.expert_feedback is not None
    assert candidate.solar_workflow.expert_feedback.labels[0].decision == "reject"
