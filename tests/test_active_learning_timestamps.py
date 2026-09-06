"""Regression: TIMESTAMPTZ columns must receive datetime objects, not ISO strings."""

from __future__ import annotations

from datetime import datetime, timezone

from app.active_learning.contracts import ActiveLearningCandidate, ReviewStatus, TaskType
from app.active_learning.repository import _as_timestamptz, _to_row_values

ISO_ZULU = "2026-09-05T14:32:45.346796Z"
OBSERVED_AT = datetime(2026, 9, 5, 14, 32, 45, 346796, tzinfo=timezone.utc)


def _candidate(*, reviewed: bool = False) -> ActiveLearningCandidate:
    return ActiveLearningCandidate(
        candidate_id="candidate-timestamp-01",
        request_id="req-timestamp-01",
        created_at=OBSERVED_AT,
        user_query="Compare parks near two campuses.",
        task_type=TaskType.COMPARISON,
        outcome_successful=True,
        review_status=ReviewStatus.APPROVED if reviewed else ReviewStatus.PENDING,
        reviewed_at=OBSERVED_AT if reviewed else None,
        approved_for_training=reviewed,
        model_id="deepseek-r1:7b",
    )


def test_json_dump_emits_the_zulu_iso_string_asyncpg_rejected() -> None:
    dumped = _candidate(reviewed=True).model_dump(mode="json")
    assert dumped["created_at"] == ISO_ZULU
    assert dumped["reviewed_at"] == ISO_ZULU


def test_to_row_values_converts_iso_strings_to_aware_datetimes() -> None:
    values = _to_row_values(_candidate(reviewed=True))
    assert values["observed_at"] == OBSERVED_AT
    assert values["reviewed_at"] == OBSERVED_AT
    assert isinstance(values["observed_at"], datetime)
    assert isinstance(values["reviewed_at"], datetime)
    assert values["observed_at"].tzinfo is not None
    assert values["reviewed_at"].tzinfo is not None


def test_unreviewed_candidate_keeps_null_reviewed_at() -> None:
    values = _to_row_values(_candidate())
    assert values["observed_at"] == OBSERVED_AT
    assert values["reviewed_at"] is None


def test_as_timestamptz_parses_the_reported_iso_string() -> None:
    parsed = _as_timestamptz(ISO_ZULU)
    assert parsed == OBSERVED_AT
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(parsed)
