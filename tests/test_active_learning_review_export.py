"""Review lifecycle, feedback, privacy and curated dataset export."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from app.active_learning.contracts import (
    ActiveLearningCandidate,
    DatasetSplit,
    ExportTask,
    FeedbackSentiment,
    FeedbackSubmission,
    ReviewDecision,
    ReviewStatus,
    SelectionReason,
    TaskType,
)
from app.active_learning.export import assign_split, export_dataset, write_export
from app.active_learning.repository import InMemoryCandidateRepository
from app.active_learning.service import ActiveLearningService, SelectionPolicy
from app.analytics.contracts import AnalysisPlan
from app.core.errors import ReviewTransitionError, SanitizationError
from pydantic import ValidationError

from tests.active_learning_fixtures import (
    ANALYSIS_PLAN,
    COMPARISON_PLAN,
    COMPARISON_QUERY,
    failing_run,
    successful_run,
)

EXPORTED_AT = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)


def cid(suffix: str) -> str:
    """Candidate ids have a minimum length, so tests use a shared prefix."""
    return f"candidate-{suffix}"


def make_candidate(
    candidate_id: str,
    *,
    user_query: str = COMPARISON_QUERY,
    successful: bool = True,
    review_status: ReviewStatus = ReviewStatus.APPROVED,
    approved: bool = True,
    corrected: dict[str, object] | None = None,
    analysis_plan: dict[str, object] | None = None,
    task_signature: str = "comparison|park|radius|count",
) -> ActiveLearningCandidate:
    full_id = cid(candidate_id)
    return ActiveLearningCandidate(
        candidate_id=full_id,
        request_id=f"req-{candidate_id}",
        user_query=user_query,
        task_type=TaskType.COMPARISON,
        task_signature=task_signature,
        query_hash=full_id,
        analysis_plan=ANALYSIS_PLAN if analysis_plan is None else analysis_plan,
        comparison_plan=COMPARISON_PLAN,
        outcome_successful=successful,
        review_status=review_status,
        approved_for_training=approved,
        corrected_output=corrected,
        model_id="deepseek-r1:7b",
        prompt_version="ariadne-system-prompt-1",
    )


def build_service() -> tuple[ActiveLearningService, InMemoryCandidateRepository]:
    repository = InMemoryCandidateRepository()
    counter = {"n": 0}

    def next_id() -> str:
        counter["n"] += 1
        return f"candidate{counter['n']:032d}"

    service = ActiveLearningService(
        repository,
        policy=SelectionPolicy(),
        id_factory=next_id,
    )
    return service, repository


# --- feedback -------------------------------------------------------------


async def test_negative_feedback_creates_a_review_candidate():
    service, repository = build_service()
    # A repeat success is filtered out on its own...
    await service.observe_run(
        request_id="req-1", user_query=COMPARISON_QUERY, response=successful_run()
    )
    filtered = await service.observe_run(
        request_id="req-2", user_query=COMPARISON_QUERY, response=successful_run()
    )
    assert filtered is None

    # ...until a human says it was wrong.
    candidate = await service.record_feedback(
        FeedbackSubmission(
            request_id="req-2",
            sentiment=FeedbackSentiment.NEGATIVE,
            failure_category="wrong_metric",
            note="density would have been the right indicator",
        )
    )

    assert candidate is not None
    assert SelectionReason.USER_NEGATIVE_FEEDBACK in candidate.selection_reasons
    assert candidate.review_status is ReviewStatus.PENDING
    assert candidate.approved_for_training is False
    assert "[wrong_metric]" in (candidate.review_notes or "")
    assert await repository.get_by_request("req-2") is not None


async def test_positive_feedback_does_not_approve_anything():
    service, _ = build_service()
    stored = await service.observe_run(
        request_id="req-1", user_query=COMPARISON_QUERY, response=successful_run()
    )
    assert stored is not None

    candidate = await service.record_feedback(
        FeedbackSubmission(request_id="req-1", sentiment=FeedbackSentiment.POSITIVE)
    )

    assert candidate is not None
    assert candidate.approved_for_training is False
    assert candidate.review_status is ReviewStatus.PENDING


# --- review lifecycle -----------------------------------------------------


async def test_corrected_candidate_can_be_approved():
    service, repository = build_service()
    stored = await service.observe_run(
        request_id="req-1",
        user_query=COMPARISON_QUERY,
        response=failing_run(
            error_code="tool_argument_error",
            message="unknown place_ref 'Tehran'; call resolve_place first",
        ),
    )
    assert stored is not None and stored.outcome_successful is False

    reviewed = await service.review(
        stored.candidate_id,
        ReviewDecision(
            status=ReviewStatus.CORRECTED,
            reviewer="rasoul",
            notes="plan rewritten by hand",
            corrected_output=ANALYSIS_PLAN,
        ),
    )

    assert reviewed.review_status is ReviewStatus.CORRECTED
    assert reviewed.approved_for_training is True
    assert reviewed.training_target == ANALYSIS_PLAN
    assert (await repository.get(stored.candidate_id)) == reviewed


async def test_unreviewed_failed_candidate_cannot_be_approved_or_exported():
    service, _ = build_service()
    stored = await service.observe_run(
        request_id="req-1",
        user_query=COMPARISON_QUERY,
        response=failing_run(
            error_code="llm_protocol_error",
            message="prose before the JSON envelope",
        ),
    )
    assert stored is not None

    with pytest.raises(ReviewTransitionError):
        await service.review(
            stored.candidate_id,
            ReviewDecision(status=ReviewStatus.APPROVED, reviewer="rasoul"),
        )

    export = export_dataset([stored], dataset_version="v1", exported_at=EXPORTED_AT)
    assert export.manifest.example_count == 0


def test_model_rejects_approval_without_a_usable_target():
    with pytest.raises(ValidationError):
        make_candidate("c1", successful=False, corrected=None)


def test_reviewer_cannot_set_the_exported_status_directly():
    with pytest.raises(ValidationError):
        ReviewDecision(status=ReviewStatus.EXPORTED)


async def test_exported_candidate_cannot_be_reviewed_again():
    service, repository = build_service()
    candidate = await repository.add(make_candidate("c1"))
    await service.mark_exported(candidate, split=DatasetSplit.TRAIN)

    with pytest.raises(ReviewTransitionError):
        await service.review(
            cid("c1"), ReviewDecision(status=ReviewStatus.REJECTED, reviewer="rasoul")
        )


# --- export ---------------------------------------------------------------


def test_rejected_and_pending_candidates_are_excluded():
    approved = make_candidate("c1")
    rejected = make_candidate(
        "c2", review_status=ReviewStatus.REJECTED, approved=False, user_query="Reject me"
    )
    pending = make_candidate(
        "c3", review_status=ReviewStatus.PENDING, approved=False, user_query="Pending question"
    )

    export = export_dataset(
        [approved, rejected, pending], dataset_version="v1", exported_at=EXPORTED_AT
    )

    exported_ids = {entry.candidate_id for entry in export.manifest.lineage}
    assert exported_ids == {cid("c1")}


def test_approved_successful_candidate_exports_a_conversational_sample():
    export = export_dataset([make_candidate("c1")], dataset_version="v1", exported_at=EXPORTED_AT)

    entry = export.manifest.lineage[0]
    example = export.splits[entry.split.value][0]
    assert [message.role for message in example.messages] == ["user", "assistant"]
    assert COMPARISON_QUERY in example.messages[0].content
    assert "osm_result_1: University of Tehran" in example.messages[0].content

    target = json.loads(example.messages[1].content)
    # The assistant target is the authoritative plan and nothing else.
    assert AnalysisPlan.model_validate(target).analysis_type == "comparison"
    assert "dataset_version" not in target
    assert "candidate_id" not in target


def test_corrected_candidate_exports_the_correction_not_the_bad_output():
    bad_plan = {**ANALYSIS_PLAN, "feature_concept": "invented concept"}
    good_plan = {**ANALYSIS_PLAN, "feature_concept": "public parks"}
    candidate = make_candidate(
        "c1",
        successful=False,
        review_status=ReviewStatus.CORRECTED,
        analysis_plan=bad_plan,
        corrected=good_plan,
    )

    export = export_dataset([candidate], dataset_version="v1", exported_at=EXPORTED_AT)

    entry = export.manifest.lineage[0]
    target = json.loads(export.splits[entry.split.value][0].messages[1].content)
    assert target["feature_concept"] == "public parks"
    assert entry.corrected is True


def test_target_must_satisfy_the_authoritative_schema():
    invalid = make_candidate(
        "c1",
        successful=False,
        review_status=ReviewStatus.CORRECTED,
        corrected={"analysis_type": "comparison", "feature_concept": "parks"},
    )

    export = export_dataset([invalid], dataset_version="v1", exported_at=EXPORTED_AT)

    assert export.manifest.example_count == 0
    assert export.manifest.skipped[0].candidate_id == cid("c1")
    assert "AnalysisPlan".lower() in export.manifest.skipped[0].reason.replace("_", "").lower()


def test_export_is_deterministic_for_a_dataset_version():
    candidates = [make_candidate(f"c{index}", user_query=f"Question {index}") for index in range(6)]

    first = export_dataset(candidates, dataset_version="v1", exported_at=EXPORTED_AT)
    second = export_dataset(
        list(reversed(candidates)),
        dataset_version="v1",
        exported_at=EXPORTED_AT,
    )

    assert first.manifest.content_digest == second.manifest.content_digest
    assert first.manifest.split_candidate_ids == second.manifest.split_candidate_ids
    for split in DatasetSplit:
        assert first.jsonl(split) == second.jsonl(split)


def test_near_duplicates_never_cross_the_train_test_boundary():
    variants = [
        "Compare public parks within 2 km of University of Tehran and Sharif University.",
        "compare the public parks within 2 km of university of tehran and sharif university",
        "Compare public parks within 2 km of University of Tehran and Sharif University!",
    ]
    candidates = [
        make_candidate(
            f"c{index}",
            user_query=query,
            task_signature=f"comparison|park|radius|metric{index}",
        )
        for index, query in enumerate(variants)
    ]

    export = export_dataset(candidates, dataset_version="v1", exported_at=EXPORTED_AT)

    splits = {entry.split for entry in export.manifest.lineage}
    assert len(splits) == 1, "near-duplicate wording must stay inside one split"


def test_split_assignment_is_stable_per_dataset_version():
    key = "comparison|park|radius|count"
    assert assign_split(key, dataset_version="v1") == assign_split(key, dataset_version="v1")


def test_manifest_records_lineage_and_keeps_metadata_out_of_the_samples():
    export = export_dataset(
        [make_candidate("c1"), make_candidate("c2", user_query="Another distinct question")],
        dataset_version="v3",
        exported_at=EXPORTED_AT,
        task=ExportTask.ANALYSIS_PLAN,
    )

    manifest = export.manifest
    assert manifest.dataset_version == "v3"
    assert manifest.target_schema_name == "AnalysisPlan"
    assert manifest.source_models == ["deepseek-r1:7b"]
    assert manifest.prompt_versions == ["ariadne-system-prompt-1"]
    assert sorted(entry.candidate_id for entry in manifest.lineage) == [cid("c1"), cid("c2")]
    assert manifest.content_digest
    assert export.target_json_schema["title"] == "AnalysisPlan"


def test_comparison_plan_task_exports_the_planner_schema():
    export = export_dataset(
        [make_candidate("c1")],
        dataset_version="v1",
        exported_at=EXPORTED_AT,
        task=ExportTask.COMPARISON_PLAN,
    )

    entry = export.manifest.lineage[0]
    target = json.loads(export.splits[entry.split.value][0].messages[1].content)
    assert target["radius_m"] == 2000
    assert [item["label"] for item in target["targets"]] == [
        "University of Tehran",
        "Sharif University of Technology",
    ]


def test_write_export_produces_split_files_manifest_and_schema(tmp_path):
    export = export_dataset([make_candidate("c1")], dataset_version="v1", exported_at=EXPORTED_AT)

    written = write_export(export, tmp_path)

    assert set(written) == {"train", "validation", "test", "manifest", "schema"}
    manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == "v1"
    schema = json.loads(written["schema"].read_text(encoding="utf-8"))
    assert schema["title"] == "AnalysisPlan"


# --- privacy --------------------------------------------------------------


def test_hidden_reasoning_is_never_stored():
    with pytest.raises(SanitizationError):
        make_candidate("c1", user_query="<think>the user probably means parks</think> find parks")


async def test_a_run_carrying_hidden_reasoning_is_dropped_not_stored():
    service, repository = build_service()
    response = successful_run()
    leaked = response.model_copy(update={"answer": "<think>internal</think> Found 12 parks."})

    candidate = await service.observe_run(
        request_id="req-1", user_query=COMPARISON_QUERY, response=leaked
    )

    assert candidate is None
    assert await repository.list_candidates(limit=10) == []


def test_credentials_are_redacted_before_they_can_be_exported():
    candidate = make_candidate(
        "c1",
        user_query=(
            "Find parks in Tehran. api_key=sk-abcdef0123456789abcdef "
            "and postgresql://user:hunter2@127.0.0.1:5433/db"
        ),
    )

    assert "hunter2" not in candidate.user_query
    assert "sk-abcdef0123456789abcdef" not in candidate.user_query

    export = export_dataset([candidate], dataset_version="v1", exported_at=EXPORTED_AT)
    blob = "".join(export.jsonl(split) for split in DatasetSplit)
    assert "hunter2" not in blob
    assert "[redacted" in blob
