"""End-to-end indicator selection for the analytical workflow (offline)."""

from __future__ import annotations

from app.agent.comparison_executor import should_use_multi_target_runner
from app.agent.comparison_workflow import extract_landmark_labels, seed_plan_from_user_message
from app.indicators.selection import plan_indicator_analysis

_A = (
    "Which university has a larger proportion of green space within 2 km: "
    "Istanbul Technical University or Boğaziçi University?"
)
_B = (
    "Which university has better access to green space within 2 km: "
    "Istanbul Technical University or Boğaziçi University?"
)
_C = "Compare road density around Istanbul Technical University and Boğaziçi University."
_D = "Which university is closer to the road network?"
_E = "Compare the diversity of urban services around the two universities."
_F = "Which area has more parks?"


def test_acceptance_a_green_space_ratio() -> None:
    assert should_use_multi_target_runner(_A)
    labels = extract_landmark_labels(_A)
    assert labels[0] == "Istanbul Technical University"
    assert "Boğaziçi" in labels[1]
    trace = plan_indicator_analysis(_A)
    assert trace.domain_selection.domains == ("green_space",)
    assert trace.selection.primary_indicator_id == "green_space_ratio"
    seed = seed_plan_from_user_message(_A)
    assert seed is not None
    assert seed.radius_m == 2000
    assert len(seed.targets) == 2


def test_acceptance_b_green_accessibility_if_listed() -> None:
    trace = plan_indicator_analysis(_B)
    assert trace.domain_selection.domains == ("green_space",)
    assert trace.selection.primary_indicator_id == "green_accessibility"


def test_acceptance_c_road_density() -> None:
    assert should_use_multi_target_runner(_C)
    trace = plan_indicator_analysis(_C)
    assert trace.domain_selection.domains == ("mobility",)
    assert trace.selection.primary_indicator_id == "road_density"


def test_acceptance_d_road_accessibility() -> None:
    trace = plan_indicator_analysis(_D)
    assert trace.domain_selection.domains == ("mobility",)
    assert trace.selection.primary_indicator_id == "road_accessibility"


def test_acceptance_e_service_diversity() -> None:
    trace = plan_indicator_analysis(_E)
    assert "urban_services" in trace.domain_selection.domains
    assert trace.selection.primary_indicator_id == "service_diversity"


def test_acceptance_f_feature_count() -> None:
    trace = plan_indicator_analysis(_F)
    assert trace.selection.primary_indicator_id == "feature_count"
