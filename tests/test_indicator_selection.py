"""Domain detection, catalog candidate retrieval, and indicator selection (offline)."""

from __future__ import annotations

import pytest
from app.analytics.rules import RULESET_VERSION
from app.core.errors import UnknownIndicatorError
from app.indicators.catalog import INDICATOR_CATALOG, INDICATOR_CATALOG_VERSION
from app.indicators.contracts import IndicatorSelectionTrace, ProposedIndicatorChoice
from app.indicators.selection import plan_indicator_analysis

_CATALOG_IDS = frozenset(INDICATOR_CATALOG.indicator_ids())


def _eligible_ids(trace: IndicatorSelectionTrace) -> set[str]:
    return {item.indicator_id for item in trace.candidates if item.eligible}


def _rejected_ids(trace: IndicatorSelectionTrace) -> set[str]:
    return {item.indicator_id for item in trace.rejected_indicators}


def _assert_trace_bounds(trace: IndicatorSelectionTrace) -> None:
    assert set(trace.candidate_indicators) <= _CATALOG_IDS
    assert _eligible_ids(trace) <= _CATALOG_IDS
    if trace.selection.primary_indicator_id is not None:
        assert trace.selection.primary_indicator_id in _CATALOG_IDS
        assert trace.selection.primary_indicator_id in _eligible_ids(trace)
    assert set(trace.selection.supporting_indicator_ids) <= _eligible_ids(trace)
    assert trace.indicator_catalog_version == INDICATOR_CATALOG_VERSION
    assert trace.ruleset_version == RULESET_VERSION
    assert "<think>" not in trace.selection_reason


def test_green_space_ratio_intent() -> None:
    trace = plan_indicator_analysis("What is the green-space ratio around these two universities?")
    _assert_trace_bounds(trace)
    assert trace.domain_selection.domains == ("green_space",)
    assert trace.inferred_goal == "coverage"
    assert trace.goal_explicit is True
    assert trace.selection.primary_indicator_id == "green_space_ratio"
    assert _eligible_ids(trace) == {"green_space_ratio"}
    assert _rejected_ids(trace) == {"green_accessibility", "green_diversity"}
    assert "COVERAGE_PERCENTAGE_001" in trace.cited_rule_ids
    assert trace.execution_status == "completed"
    assert trace.methodology == "area_share"


def test_green_accessibility_intent() -> None:
    trace = plan_indicator_analysis("Which university has better access to green space?")
    _assert_trace_bounds(trace)
    assert trace.domain_selection.domains == ("green_space",)
    assert trace.inferred_goal == "accessibility"
    assert trace.selection.primary_indicator_id == "green_accessibility"
    assert _eligible_ids(trace) == {"green_accessibility"}
    assert "green_space_ratio" in _rejected_ids(trace)
    assert "green_diversity" in _rejected_ids(trace)
    assert "ACCESSIBILITY_DISTANCE_001" in trace.cited_rule_ids
    assert trace.selection.supporting_indicator_ids == ()


def test_green_diversity_intent() -> None:
    trace = plan_indicator_analysis("Which campus has greater green-space diversity?")
    _assert_trace_bounds(trace)
    assert trace.domain_selection.domains == ("green_space",)
    assert trace.inferred_goal == "variability"
    assert trace.selection.primary_indicator_id == "green_diversity"
    assert _eligible_ids(trace) == {"green_diversity"}
    assert "VARIABILITY_STDDEV_001" in trace.cited_rule_ids


def test_road_accessibility_intent() -> None:
    trace = plan_indicator_analysis("Which campus has better road accessibility?")
    _assert_trace_bounds(trace)
    assert trace.domain_selection.domains == ("mobility",)
    assert trace.inferred_goal == "accessibility"
    assert trace.selection.primary_indicator_id == "road_accessibility"
    assert _eligible_ids(trace) == {"road_accessibility"}
    assert _rejected_ids(trace) == {"road_density", "intersection_density"}


def test_road_density_intent() -> None:
    trace = plan_indicator_analysis("What is the road density near the campus?")
    _assert_trace_bounds(trace)
    assert trace.domain_selection.domains == ("mobility",)
    assert trace.inferred_goal == "concentration"
    assert trace.selection.primary_indicator_id == "road_density"
    assert "road_density" in _eligible_ids(trace)
    assert "intersection_density" in _eligible_ids(trace)
    assert "road_accessibility" in _rejected_ids(trace)
    assert "CONCENTRATION_DENSITY_001" in trace.cited_rule_ids


def test_intersection_density_intent() -> None:
    trace = plan_indicator_analysis("What is the intersection density near the campus?")
    _assert_trace_bounds(trace)
    assert trace.domain_selection.domains == ("mobility",)
    assert trace.inferred_goal == "concentration"
    assert trace.selection.primary_indicator_id == "intersection_density"
    assert "intersection_density" in _eligible_ids(trace)
    assert "road_density" in _eligible_ids(trace)
    assert trace.selection.primary_indicator_id != "road_density"


def test_poi_density_intent() -> None:
    trace = plan_indicator_analysis("What is the POI density in this neighbourhood?")
    _assert_trace_bounds(trace)
    assert trace.domain_selection.domains == ("urban_services",)
    assert trace.inferred_goal == "concentration"
    assert trace.selection.primary_indicator_id == "poi_density"
    assert _eligible_ids(trace) == {"poi_density"}
    assert "service_accessibility" in _rejected_ids(trace)
    assert "service_diversity" in _rejected_ids(trace)


def test_service_accessibility_intent() -> None:
    trace = plan_indicator_analysis("Which area has better access to urban services?")
    _assert_trace_bounds(trace)
    assert trace.domain_selection.domains == ("urban_services",)
    assert trace.inferred_goal == "accessibility"
    assert trace.selection.primary_indicator_id == "service_accessibility"
    assert _eligible_ids(trace) == {"service_accessibility"}


def test_service_diversity_intent() -> None:
    trace = plan_indicator_analysis("How diverse are urban services around the campus?")
    _assert_trace_bounds(trace)
    assert trace.domain_selection.domains == ("urban_services",)
    assert trace.inferred_goal == "variability"
    assert trace.selection.primary_indicator_id == "service_diversity"
    assert _eligible_ids(trace) == {"service_diversity"}


def test_generic_count_request() -> None:
    trace = plan_indicator_analysis("How many benches are in this area?")
    _assert_trace_bounds(trace)
    assert trace.domain_selection.domains == ("core",)
    assert trace.inferred_goal == "abundance"
    assert trace.selection.primary_indicator_id == "feature_count"
    assert _eligible_ids(trace) == {"feature_count"}
    assert "ABUNDANCE_COUNT_001" in trace.cited_rule_ids
    assert trace.methodology == "count"


def test_park_count_still_selects_feature_count() -> None:
    trace = plan_indicator_analysis("Which area has more parks?")
    _assert_trace_bounds(trace)
    assert "core" in trace.domain_selection.domains
    assert trace.inferred_goal == "abundance"
    assert trace.selection.primary_indicator_id == "feature_count"
    assert trace.execution_status == "completed"


def test_unsupported_invented_indicator_rejection() -> None:
    with pytest.raises(UnknownIndicatorError, match="invented_green_score"):
        plan_indicator_analysis(
            "Which university has better access to green space?",
            proposal=ProposedIndicatorChoice(
                primary_indicator_id="invented_green_score",
                selection_reason="Invented composite green score.",
            ),
        )


def test_known_but_ineligible_proposal_falls_back() -> None:
    trace = plan_indicator_analysis(
        "Which university has better access to green space?",
        proposal=ProposedIndicatorChoice(
            primary_indicator_id="feature_count",
            selection_reason="Count the parks instead.",
        ),
    )
    _assert_trace_bounds(trace)
    assert trace.selection.primary_indicator_id == "green_accessibility"
    assert any(
        item.indicator_id == "feature_count" and item.reason == "goal_mismatch"
        for item in trace.rejected_indicators
    )


def test_multi_domain_candidate_retrieval() -> None:
    trace = plan_indicator_analysis(
        "Which university has better access to green space and urban services?"
    )
    _assert_trace_bounds(trace)
    assert set(trace.domain_selection.domains) == {"green_space", "urban_services"}
    assert trace.inferred_goal == "accessibility"
    assert _eligible_ids(trace) == {"green_accessibility", "service_accessibility"}
    assert "green_space_ratio" in _rejected_ids(trace)
    assert "poi_density" in _rejected_ids(trace)
    assert trace.selection.primary_indicator_id in {
        "green_accessibility",
        "service_accessibility",
    }
    assert "mobility" not in trace.domain_selection.domains
    assert "road_accessibility" not in trace.candidate_indicators


def test_primary_plus_optional_supporting_indicators() -> None:
    trace = plan_indicator_analysis("Compare the green environment around two universities.")
    _assert_trace_bounds(trace)
    assert trace.domain_selection.domains == ("green_space",)
    assert trace.goal_explicit is False
    assert trace.selection.primary_indicator_id == "green_space_ratio"
    assert set(trace.selection.supporting_indicator_ids) == {
        "green_accessibility",
        "green_diversity",
    }
    assert _eligible_ids(trace) == {
        "green_space_ratio",
        "green_accessibility",
        "green_diversity",
    }
    assert trace.execution_status == "completed"


def test_llm_may_choose_among_eligible_indicators() -> None:
    trace = plan_indicator_analysis(
        "Compare the green environment around two universities.",
        proposal=ProposedIndicatorChoice(
            primary_indicator_id="green_accessibility",
            supporting_indicator_ids=("green_diversity",),
            selection_reason="Access and mix of green types around campuses.",
        ),
    )
    _assert_trace_bounds(trace)
    assert trace.selection.primary_indicator_id == "green_accessibility"
    assert trace.selection.supporting_indicator_ids == ("green_diversity",)
    assert trace.selection_reason == "Access and mix of green types around campuses."


def test_specific_access_does_not_select_ratio_or_diversity() -> None:
    trace = plan_indicator_analysis("Which campus has better access to green space?")
    assert trace.selection.primary_indicator_id == "green_accessibility"
    assert "green_space_ratio" not in trace.selection.supporting_indicator_ids
    assert "green_diversity" not in trace.selection.supporting_indicator_ids
