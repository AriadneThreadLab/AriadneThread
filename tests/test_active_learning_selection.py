"""Candidate selection, scoring, attribution and diversity control."""

from __future__ import annotations

import pytest
from app.active_learning.contracts import (
    ErrorAttribution,
    ReviewStatus,
    SelectionReason,
    TaskType,
)
from app.active_learning.novelty import (
    NoveltyAssessment,
    build_task_signature,
    is_near_duplicate,
    query_fingerprint,
)
from app.active_learning.repository import InMemoryCandidateRepository
from app.active_learning.scoring import ActiveLearningScorer
from app.active_learning.service import ActiveLearningService, SelectionPolicy
from app.active_learning.signals import classify_error, extract_run_signals
from app.agent.contracts import GeoAgentResponse

from tests.active_learning_fixtures import (
    COMPARISON_QUERY,
    failing_run,
    successful_run,
    trace,
)


def novelty_of(query: str) -> NoveltyAssessment:
    return NoveltyAssessment(
        query_hash=query_fingerprint(query),
        task_signature="spatial_search|park|place",
        duplicate_count=0,
        nearest_similarity=0.0,
        is_novel=True,
    )


def build_service(**policy: object) -> tuple[ActiveLearningService, InMemoryCandidateRepository]:
    repository = InMemoryCandidateRepository()
    counter = {"n": 0}

    def next_id() -> str:
        counter["n"] += 1
        return f"candidate{counter['n']:032d}"

    service = ActiveLearningService(
        repository,
        policy=SelectionPolicy(**policy),  # type: ignore[arg-type]
        id_factory=next_id,
    )
    return service, repository


async def test_successful_request_may_become_a_candidate():
    service, repository = build_service()

    candidate = await service.observe_run(
        request_id="req-1",
        user_query=COMPARISON_QUERY,
        response=successful_run(),
    )

    assert candidate is not None
    assert candidate.outcome_successful
    assert SelectionReason.SUCCESSFUL_HIGH_VALUE_TRACE in candidate.selection_reasons
    assert SelectionReason.NOVEL_QUERY in candidate.selection_reasons
    assert candidate.review_status is ReviewStatus.PENDING
    assert candidate.approved_for_training is False
    assert await repository.get(candidate.candidate_id) is not None


async def test_ordinary_duplicate_success_is_filtered():
    service, repository = build_service()

    first = await service.observe_run(
        request_id="req-1", user_query=COMPARISON_QUERY, response=successful_run()
    )
    second = await service.observe_run(
        request_id="req-2", user_query=COMPARISON_QUERY, response=successful_run()
    )

    assert first is not None
    assert second is None, "a repeat of an already-captured success is not informative"
    assert len(await repository.list_candidates(limit=10)) == 1


async def test_invalid_tool_arguments_increase_the_score():
    service, _ = build_service()

    success = await service.observe_run(
        request_id="req-1", user_query=COMPARISON_QUERY, response=successful_run()
    )
    failure = await service.observe_run(
        request_id="req-2",
        user_query="Find hospitals within 800 m of Azadi Square.",
        response=failing_run(
            error_code="tool_argument_error",
            message="query_osm arguments failed validation: limit must be <= 1000",
        ),
    )

    assert success is not None and failure is not None
    assert SelectionReason.TOOL_VALIDATION_FAILED in failure.selection_reasons
    assert failure.informativeness_score > success.informativeness_score


async def test_guardrail_block_scores_above_a_plain_validation_failure():
    service, _ = build_service()

    plain = await service.observe_run(
        request_id="req-1",
        user_query="Find hospitals within 800 m of Azadi Square.",
        response=failing_run(
            error_code="tool_argument_error",
            message="query_osm arguments failed validation: limit must be <= 1000",
        ),
    )
    guardrail = await service.observe_run(
        request_id="req-2",
        user_query="Find cafes within 500 m of Milad Tower.",
        response=failing_run(
            error_code="tool_argument_error",
            message=(
                "point-radius scope requires latitude and longitude explicitly provided "
                "by the user or a trusted place_ref_scope from resolve_place; inventing "
                "coordinates is not allowed."
            ),
        ),
    )

    assert plain is not None and guardrail is not None
    assert SelectionReason.HALLUCINATED_COORDINATE_BLOCKED in guardrail.selection_reasons
    assert SelectionReason.GUARDRAIL_TRIGGERED in guardrail.selection_reasons
    assert guardrail.informativeness_score > plain.informativeness_score


async def test_ambiguous_place_resolution_increases_the_score():
    service, _ = build_service()

    candidate = await service.observe_run(
        request_id="req-1",
        user_query="Find parks within 1 km of Central Station.",
        response=failing_run(
            error_code="place_ambiguous",
            message="multiple incompatible candidates for 'Central Station'",
            tool="resolve_place",
        ),
    )

    assert candidate is not None
    assert SelectionReason.PLACE_RESOLUTION_AMBIGUOUS in candidate.selection_reasons
    assert candidate.informativeness_score >= service.policy.min_score_for_review
    assert [event.status for event in candidate.place_resolution_events] == ["ambiguous"]


async def test_external_overpass_timeout_alone_is_not_a_training_signal():
    service, repository = build_service()
    response = GeoAgentResponse(
        answer="The live query timed out. Documentation grounding remains valid.",
        trace=[
            trace("tool_call", "calling query_osm", tool="query_osm"),
            trace(
                "tool_error",
                "Overpass timed out after 2 attempts",
                tool="query_osm",
                error_code="overpass_timeout",
            ),
        ],
        errors=["overpass_timeout: Overpass timed out after 2 attempts"],
        live_query_failed=True,
        live_error_code="overpass_timeout",
        validated_tags=["leisure=park"],
        stop_reason="final_answer",
        model="deepseek-r1:7b",
    )

    candidate = await service.observe_run(
        request_id="req-1", user_query=COMPARISON_QUERY, response=response
    )

    assert candidate is None
    assert await repository.list_candidates(limit=10) == []

    signals = extract_run_signals(COMPARISON_QUERY, response)
    breakdown = ActiveLearningScorer().score(signals, novelty_of("brand new question"))
    assert breakdown.external_only is True
    assert breakdown.score < 0.2


async def test_model_attributed_place_failure_still_scores():
    """Nominatim answered; the model just asked for the wrong thing."""
    service, _ = build_service()

    candidate = await service.observe_run(
        request_id="req-1",
        user_query="Find parks within 1 km of University of Tehran.",
        response=failing_run(
            error_code="place_resolution_error",
            message="no semantically matching place resolution result (rejected=name_mismatch)",
            tool="resolve_place",
        ),
    )

    assert candidate is not None
    assert SelectionReason.PLACE_RESOLUTION_FAILED in candidate.selection_reasons
    assert candidate.external_error_codes == []


@pytest.mark.parametrize(
    ("code", "message", "expected"),
    [
        ("overpass_timeout", "gateway timeout", ErrorAttribution.EXTERNAL),
        ("overpass_rate_limited", "429", ErrorAttribution.EXTERNAL),
        ("tool_argument_error", "unknown place_ref 'Tehran'", ErrorAttribution.MODEL),
        ("llm_protocol_error", "prose before JSON", ErrorAttribution.MODEL),
        (
            "place_resolution_error",
            "Nominatim request failed: timed out",
            ErrorAttribution.EXTERNAL,
        ),
        (
            "place_resolution_error",
            "no semantically matching result (rejected=name_mismatch)",
            ErrorAttribution.MODEL,
        ),
    ],
)
def test_error_attribution_separates_model_from_infrastructure(code, message, expected):
    assert classify_error(code, message) is expected


def test_task_signature_is_structural():
    signature = build_task_signature(
        task_type=TaskType.COMPARISON,
        feature_concept="public parks",
        scope_kind="radius",
        metric="count",
    )
    assert signature == "comparison|park|radius|count"


def test_near_duplicate_detection_respects_numbers():
    a = "Compare public parks within 2 km of University of Tehran and Sharif University."
    b = "compare the public parks within 2 km of university of tehran and sharif university"
    c = "Compare public parks within 5 km of University of Tehran and Sharif University."

    assert is_near_duplicate(a, b)
    assert not is_near_duplicate(a, c), "a different radius is a different training example"


async def test_signature_saturation_stops_collecting_more_of_the_same():
    service, repository = build_service(max_duplicates_per_signature=2)

    queries = [
        "Find parks within 900 m of Azadi Square.",
        "Find gardens within 950 m of Vanak Square.",
        "Find greens within 970 m of Tajrish Square.",
        "Find lawns within 980 m of Enghelab Square.",
    ]
    for index, query in enumerate(queries):
        await service.observe_run(
            request_id=f"req-{index}",
            user_query=query,
            response=failing_run(
                error_code="tool_argument_error",
                message="unknown place_ref 'square'; call resolve_place first",
            ),
        )

    stored = await repository.list_candidates(limit=10)
    assert len(stored) == 2, "the same failure shape is capped by the duplicate policy"
