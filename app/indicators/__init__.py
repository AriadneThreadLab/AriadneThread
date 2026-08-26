"""Indicator Catalog — domain-aware measures above the Metric Catalog.

Public API is the default catalog loaded from ``definitions/``. Nothing here
executes analysis or talks to the agent.
"""

from __future__ import annotations

from app.indicators.catalog import (
    INDICATOR_CATALOG,
    INDICATOR_CATALOG_VERSION,
    AvailableCapabilities,
    IndicatorCatalog,
    find_candidates,
    get_indicator,
    list_by_domain,
    list_indicators,
    load_indicator_catalog,
)
from app.indicators.compute import IndicatorExecutor, compute_indicator
from app.indicators.contracts import (
    AnalysisDomainSelection,
    DataPlanningRequest,
    DataRequirement,
    DataRequirementPlan,
    DomainId,
    IndicatorCandidate,
    IndicatorComputationResult,
    IndicatorComputeRequest,
    IndicatorDefinition,
    IndicatorSelection,
    IndicatorSelectionEvidence,
    IndicatorSelectionTrace,
    PlanningTarget,
    ProposedIndicatorChoice,
    RagGroundingEvidence,
    parse_tag_literal,
)
from app.indicators.planning import (
    DataRequirementPlanner,
    plan_indicator_data,
    plan_indicator_data_from_selection,
)
from app.indicators.selection import (
    DomainResolver,
    GoalInterpreter,
    IndicatorSelector,
    plan_indicator_analysis,
)

__all__ = [
    "INDICATOR_CATALOG",
    "INDICATOR_CATALOG_VERSION",
    "AnalysisDomainSelection",
    "AvailableCapabilities",
    "DataPlanningRequest",
    "DataRequirement",
    "DataRequirementPlan",
    "DataRequirementPlanner",
    "DomainId",
    "DomainResolver",
    "GoalInterpreter",
    "IndicatorCandidate",
    "IndicatorCatalog",
    "IndicatorComputationResult",
    "IndicatorComputeRequest",
    "IndicatorDefinition",
    "IndicatorExecutor",
    "IndicatorSelection",
    "IndicatorSelectionEvidence",
    "IndicatorSelectionTrace",
    "IndicatorSelector",
    "PlanningTarget",
    "ProposedIndicatorChoice",
    "RagGroundingEvidence",
    "compute_indicator",
    "find_candidates",
    "get_indicator",
    "list_by_domain",
    "list_indicators",
    "load_indicator_catalog",
    "parse_tag_literal",
    "plan_indicator_analysis",
    "plan_indicator_data",
    "plan_indicator_data_from_selection",
]
