"""Analysis domain contracts.

Plan models are the only schema the model may fill for ``analyze_features``.
Result and comparison models are produced exclusively by deterministic code.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --- Enumerations (closed sets; inventing values is a schema rejection) ---

MetricType = Literal[
    "count",
    "density",
    "sum",
    "mean",
    "median",
    "standard_deviation",
    "min",
    "max",
    "ratio",
    "total_area",
    "mean_area",
    "median_area",
    "nearest_distance",
    "mean_nearest_distance",
    "median_nearest_distance",
    "coverage_percentage",
]

MetricRole = Literal["primary", "supporting"]
MetricDirection = Literal["higher_is_better", "lower_is_better", "neutral"]
AnalysisType = Literal["single_target", "comparison"]

AnalysisGoal = Literal[
    "abundance",
    "concentration",
    "accessibility",
    "total_provision",
    "typical_value",
    "variability",
    "coverage",
    "relative_share",
]

MetricStatus = Literal["computed", "insufficient_data", "not_applicable"]

RuleId = Literal[
    "ABUNDANCE_COUNT_001",
    "CONCENTRATION_DENSITY_001",
    "ACCESSIBILITY_DISTANCE_001",
    "TOTAL_PROVISION_SUM_001",
    "GREENSPACE_TOTAL_AREA_001",
    "TYPICAL_VALUE_MEDIAN_001",
    "EXPLICIT_AVERAGE_MEAN_001",
    "VARIABILITY_STDDEV_001",
    "COVERAGE_PERCENTAGE_001",
    "RELATIVE_SHARE_RATIO_001",
    "RANGE_EXTREMES_001",
]

LimitationCode = Literal[
    "osm_completeness",
    "missing_property_values",
    "nonstandard_property_values",
    "count_is_not_capacity",
    "euclidean_not_route",
    "scope_definition_affects_results",
    "polygon_quality_varies",
    "polygon_geometry_partially_available",
    "overlapping_polygons",
    "osm_not_official_inventory",
    "results_truncated_at_limit",
    "property_units_undeclared",
    "single_reference_point",
]

FeasibilityRequirement = Literal[
    "metric_in_catalog",
    "semantic_rule_supports_metric",
    "role_permitted",
    "explicit_request_present",
    "dataset_reference_resolvable",
    "dataset_not_empty",
    "numeric_property_present",
    "sufficient_observations",
    "required_geometry_present",
    "analysis_area_available",
    "reference_geometry_available",
    "denominator_non_zero",
    "unit_defined",
    "comparable_target_scopes",
    "ratio_pair_supported",
]

CheckStatus = Literal["passed", "failed", "not_applicable"]

RejectionReason = Literal[
    "metric_not_in_catalog",
    "semantic_rule_violation",
    "role_not_permitted",
    "explicit_request_required",
    "unknown_dataset_reference",
    "empty_dataset",
    "numeric_property_unavailable",
    "insufficient_observations",
    "required_geometry_unavailable",
    "analysis_area_unavailable",
    "reference_geometry_unavailable",
    "zero_denominator",
    "unit_undefined",
    "incomparable_target_scopes",
    "unsupported_ratio_pair",
]

DecisionBasisType = Literal[
    "user_explicit",
    "semantic_rule",
    "metric_catalog",
    "data_feasibility",
    "statistical_guidance",
    "fallback",
]

ComparisonVerdict = Literal[
    "higher",
    "lower",
    "tie",
    "insufficient_data",
    "not_comparable",
]

ReportOrigin = Literal["deterministic", "model_narrative"]

ReportSectionKey = Literal[
    "analysis_goal",
    "selected_indicator",
    "why_this_indicator",
    "analysis_method",
    "comparison",
    "supporting_indicators",
    "interpretation",
    "data_limitations",
    "conclusion",
]

AnalysisBlockStatus = Literal["completed", "rejected", "abandoned"]

DATASET_REF_PATTERN = r"^osm_result_(?:[1-9]|[1-9][0-9])$"
TARGET_ID_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"

MAX_TARGETS = 4
MAX_METRICS = 4
MAX_REFERENCE_POINTS = 4
MAX_ANALYSIS_REPLANS = 1
MAX_ANALYSIS_FEATURES = 1000

#: Metrics that require a numeric OSM tag property.
NUMERIC_PROPERTY_METRICS: frozenset[str] = frozenset(
    {"sum", "mean", "median", "standard_deviation", "min", "max"}
)

_PROTOCOL_LEAK_RE = re.compile(
    r"(tool_calls|final_answer|```|<think\b)",
    re.IGNORECASE,
)
_FORBIDDEN_LABEL_CHARS = frozenset('"\\\n\r\t')


def _reject_unsafe_label(value: str, *, field: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    if _FORBIDDEN_LABEL_CHARS.intersection(text):
        raise ValueError(f"{field} contains characters that are not allowed")
    if _PROTOCOL_LEAK_RE.search(text):
        raise ValueError(f"{field} must not contain protocol or reasoning markers")
    return text


class GeoPoint(BaseModel):
    """WGS84 point used as a distance reference origin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class AnalysisTarget(BaseModel):
    """One comparison target bound to a request-scoped dataset reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_id: str = Field(pattern=TARGET_ID_PATTERN)
    label: str = Field(min_length=1, max_length=80)
    dataset_ref: str = Field(pattern=DATASET_REF_PATTERN)
    reference_points: list[GeoPoint] = Field(
        default_factory=list,
        max_length=MAX_REFERENCE_POINTS,
    )

    @model_validator(mode="after")
    def _validate_label(self) -> AnalysisTarget:
        _reject_unsafe_label(self.label, field="label")
        return self


class RatioSpec(BaseModel):
    """Numerator/denominator pair for the bounded ratio metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    numerator: MetricType
    denominator: MetricType


class MetricRequest(BaseModel):
    """One metric the model proposes for the plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: MetricType
    role: MetricRole = "supporting"
    inferred_goal: AnalysisGoal
    claimed_rule_ids: list[RuleId] = Field(default_factory=list, max_length=3)
    user_explicit: bool = False
    property_key: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z0-9_:-]*$",
    )
    direction: MetricDirection | None = None
    ratio: RatioSpec | None = None


class AnalysisPlan(BaseModel):
    """Validated analytical plan — the argument model for analyze_features."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_type: AnalysisType
    feature_concept: str = Field(min_length=1, max_length=80)
    comparison_goal: str = Field(min_length=3, max_length=200)
    targets: list[AnalysisTarget] = Field(min_length=1, max_length=MAX_TARGETS)
    metrics: list[MetricRequest] = Field(min_length=1, max_length=MAX_METRICS)

    @model_validator(mode="after")
    def _validate_plan(self) -> AnalysisPlan:
        _reject_unsafe_label(self.feature_concept, field="feature_concept")
        _reject_unsafe_label(self.comparison_goal, field="comparison_goal")

        if self.analysis_type == "comparison" and len(self.targets) < 2:
            raise ValueError("comparison analysis requires at least two targets")

        target_ids = [t.target_id for t in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("target_id values must be unique")
        dataset_refs = [t.dataset_ref for t in self.targets]
        if len(dataset_refs) != len(set(dataset_refs)):
            raise ValueError("dataset_ref values must be unique")

        primaries = [m for m in self.metrics if m.role == "primary"]
        if len(primaries) != 1:
            raise ValueError("exactly one metric must have role 'primary'")

        pairs = [(m.metric, m.property_key) for m in self.metrics]
        if len(pairs) != len(set(pairs)):
            raise ValueError("(metric, property_key) pairs must be unique")

        for metric_req in self.metrics:
            is_ratio = metric_req.metric == "ratio"
            if is_ratio and metric_req.ratio is None:
                raise ValueError("ratio metric requires a ratio specification")
            if not is_ratio and metric_req.ratio is not None:
                raise ValueError("ratio specification is only allowed for metric 'ratio'")
            needs_property = metric_req.metric in NUMERIC_PROPERTY_METRICS
            if needs_property and not metric_req.property_key:
                raise ValueError(f"metric '{metric_req.metric}' requires property_key")
            if not needs_property and metric_req.property_key is not None:
                raise ValueError(f"metric '{metric_req.metric}' must not set property_key")
        return self


class KnowledgeSourceRef(BaseModel):
    """Compact documentation citation for data provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(max_length=160)
    url: str | None = None


class CalculationProvenance(BaseModel):
    """Operational provenance for one deterministic metric computation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: MetricType
    dataset_ref: str
    target_ref: str
    implementation_id: str
    geometry_strategy: str | None = None
    observation_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    invalid_count: int = Field(ge=0)
    unit: str
    property_key: str | None = None
    analysis_area_km2: float | None = None
    reference_point_count: int = Field(default=0, ge=0)
    metric_catalog_version: str


class DataProvenance(BaseModel):
    """Compact analytical data provenance for one target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_concept: str
    resolved_tags: list[str]
    grounding_sources: list[KnowledgeSourceRef] = Field(default_factory=list)
    dataset_ref: str
    target_id: str
    analysis_scope: str
    scope_kind: Literal["place", "point", "bbox"]
    analysis_area_km2: float | None = None
    retrieved_feature_count: int = Field(ge=0)
    valid_observation_count: int = Field(ge=0)
    missing_observation_count: int = Field(ge=0)
    invalid_observation_count: int = Field(ge=0)
    geometry_counts: dict[str, int] = Field(default_factory=dict)
    truncated: bool = False
    effective_limit: int | None = Field(default=None, ge=1)
    limit_reached: bool = False
    retrieved_at: str
    attribution: str = "© OpenStreetMap contributors (ODbL)"


class MetricResult(BaseModel):
    """One computed (or insufficient) metric value for one target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: MetricType
    role: MetricRole
    label: str
    status: MetricStatus
    value: float | None
    unit: str
    direction: MetricDirection
    observation_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    invalid_count: int = Field(ge=0)
    property_key: str | None = None
    provenance: CalculationProvenance
    notes: list[str] = Field(default_factory=list)


class TargetAnalysisResult(BaseModel):
    """All metric results for one analysis target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_id: str
    label: str
    dataset_ref: str
    data_provenance: DataProvenance
    metrics: list[MetricResult]


class AnalysisResult(BaseModel):
    """Deterministic analytics output for a validated plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_type: AnalysisType
    feature_concept: str
    comparison_goal: str
    targets: list[TargetAnalysisResult]
    metric_catalog_version: str
    ruleset_version: str
    warnings: list[str] = Field(default_factory=list)
    limitations: list[LimitationCode] = Field(default_factory=list)


class TargetMetricValue(BaseModel):
    """One target's value inside a metric comparison ranking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_id: str
    label: str
    value: float | None
    status: MetricStatus
    observation_count: int = Field(ge=0)


class MetricComparison(BaseModel):
    """Deterministic comparison of one metric across targets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: MetricType
    role: MetricRole
    label: str
    unit: str
    direction: MetricDirection
    ranking: list[TargetMetricValue]
    verdict: ComparisonVerdict
    leading_target_id: str | None = None
    preferred_target_id: str | None = None
    absolute_difference: float | None = None
    relative_difference_percent: float | None = None
    difference_unit: str = ""
    notes: list[str] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    """Deterministic comparison outcome for a plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_type: AnalysisType
    primary: MetricComparison | None = None
    supporting: list[MetricComparison] = Field(default_factory=list)
    overall_statement: str
    overall_confidence: Literal["clear", "close", "insufficient_data"]


class ReportSection(BaseModel):
    """One section of the comparison report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: ReportSectionKey
    title: str
    body: str
    origin: ReportOrigin


class ComparisonReport(BaseModel):
    """Human-readable report assembled after deterministic results exist."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = "Comparison Report"
    sections: list[ReportSection]
    numeric_fidelity: Literal[
        "verified",
        "unverified_numbers_removed",
        "not_applicable",
    ] = "not_applicable"


class MetricSelectionEvidence(BaseModel):
    """One structured reason supporting a metric choice."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    basis_type: DecisionBasisType
    rule_id: RuleId | None = None
    statement: str = Field(max_length=240)
    source_ref: str | None = None


class MetricFeasibilityCheck(BaseModel):
    """One feasibility requirement evaluation (pass or fail)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement: FeasibilityRequirement
    status: CheckStatus
    detail: str = Field(max_length=240)


class MetricSelectionTrace(BaseModel):
    """Full selection lifecycle for one proposed metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: MetricType
    role: MetricRole
    inferred_goal: AnalysisGoal
    property_key: str | None = None
    target_ids: list[str]
    evidence: list[MetricSelectionEvidence]
    feasibility_checks: list[MetricFeasibilityCheck]
    final_status: Literal["executed", "rejected", "superseded"]
    replacement_metric: MetricType | None = None
    rejection_reason: RejectionReason | None = None
    supported_alternatives: list[MetricType] = Field(default_factory=list)
    attempt_index: int = Field(ge=0)


class AnalysisDecisionTrace(BaseModel):
    """Decision provenance for one analytical plan attempt sequence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inferred_analysis_type: AnalysisType
    inferred_comparison_goal: str = Field(max_length=200)
    feature_concept: str = Field(max_length=80)
    metric_selections: list[MetricSelectionTrace]
    # May exceed MAX_ANALYSIS_REPLANS by one when the abandon path is recorded.
    plan_revision_count: int = Field(ge=0, le=MAX_ANALYSIS_REPLANS + 1)
    final_primary_metric: MetricType | None = None
    final_supporting_metrics: list[MetricType] = Field(default_factory=list)
    ruleset_version: str
    metric_catalog_version: str


class AnalysisBlock(BaseModel):
    """Optional analytics payload on the agent/API response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: AnalysisPlan
    decision_trace: AnalysisDecisionTrace
    result: AnalysisResult | None = None
    comparison: ComparisonResult | None = None
    report: ComparisonReport | None = None
    status: AnalysisBlockStatus
