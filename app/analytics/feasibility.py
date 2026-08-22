"""Deterministic metric feasibility validation and selection traces."""

from __future__ import annotations

from dataclasses import dataclass

from app.analytics.catalog import METRIC_CATALOG, METRIC_CATALOG_VERSION, get_metric_definition
from app.analytics.contracts import (
    AnalysisDecisionTrace,
    AnalysisPlan,
    CheckStatus,
    FeasibilityRequirement,
    MetricFeasibilityCheck,
    MetricRequest,
    MetricSelectionEvidence,
    MetricSelectionTrace,
    MetricType,
    RejectionReason,
)
from app.analytics.datasets import DatasetRegistry, UnknownDatasetError
from app.analytics.engine import SUPPORTED_RATIO_PAIRS
from app.analytics.observations import (
    extract_areas,
    extract_distances,
    extract_numeric,
)
from app.analytics.rules import RULESET_VERSION, validate_metric_for_goal


@dataclass(frozen=True, slots=True)
class FeasibilityOutcome:
    """Result of validating an entire AnalysisPlan."""

    ok: bool
    traces: tuple[MetricSelectionTrace, ...]
    decision_trace: AnalysisDecisionTrace
    rejection_reason: RejectionReason | None
    supported_alternatives: tuple[MetricType, ...]
    feedback: str


class MetricFeasibilityValidator:
    """Validate semantic sense and data support for a candidate AnalysisPlan."""

    def validate(
        self,
        plan: AnalysisPlan,
        datasets: DatasetRegistry,
        *,
        attempt_index: int,
        plan_revision_count: int,
        prior_rejected: list[MetricSelectionTrace] | None = None,
    ) -> FeasibilityOutcome:
        traces: list[MetricSelectionTrace] = list(prior_rejected or [])
        all_failed: list[MetricSelectionTrace] = []
        alternatives: list[MetricType] = []
        primary_rejection: RejectionReason | None = None

        # Plan-level comparable scopes check (recorded on each metric).
        scopes_ok, scopes_detail = self._comparable_scopes(plan, datasets)

        for metric_req in plan.metrics:
            trace = self._validate_metric(
                metric_req,
                plan,
                datasets,
                attempt_index=attempt_index,
                scopes_ok=scopes_ok,
                scopes_detail=scopes_detail,
            )
            traces.append(trace)
            if trace.final_status == "rejected":
                all_failed.append(trace)
                alternatives.extend(trace.supported_alternatives)
                if metric_req.role == "primary" and primary_rejection is None:
                    primary_rejection = trace.rejection_reason

        ok = not all_failed
        unique_alts = tuple(dict.fromkeys(alternatives))
        final_primary = None
        final_supporting: list[MetricType] = []
        if ok:
            for metric_req in plan.metrics:
                if metric_req.role == "primary":
                    final_primary = metric_req.metric
                else:
                    final_supporting.append(metric_req.metric)

        decision = AnalysisDecisionTrace(
            inferred_analysis_type=plan.analysis_type,
            inferred_comparison_goal=plan.comparison_goal,
            feature_concept=plan.feature_concept,
            metric_selections=traces,
            plan_revision_count=plan_revision_count,
            final_primary_metric=final_primary,
            final_supporting_metrics=final_supporting,
            ruleset_version=RULESET_VERSION,
            metric_catalog_version=METRIC_CATALOG_VERSION,
        )

        feedback = ""
        if not ok:
            sample = all_failed[0]
            failed = [c.requirement for c in sample.feasibility_checks if c.status == "failed"]
            detail = next(
                (c.detail for c in sample.feasibility_checks if c.status == "failed"),
                "",
            )
            alts = ",".join(unique_alts) if unique_alts else "none"
            feedback = (
                f"status=rejected metric={sample.metric} "
                f"property={sample.property_key or '-'} "
                f"failed_checks={','.join(failed)} "
                f"reason={sample.rejection_reason} "
                f"detail={detail} "
                f"supported_alternatives={alts} "
                "Choose one supported metric or answer without analytics. "
                "Do not invent values."
            )

        return FeasibilityOutcome(
            ok=ok,
            traces=tuple(traces),
            decision_trace=decision,
            rejection_reason=primary_rejection
            or (all_failed[0].rejection_reason if all_failed else None),
            supported_alternatives=unique_alts,
            feedback=feedback,
        )

    def _validate_metric(
        self,
        metric_req: MetricRequest,
        plan: AnalysisPlan,
        datasets: DatasetRegistry,
        *,
        attempt_index: int,
        scopes_ok: bool,
        scopes_detail: str,
    ) -> MetricSelectionTrace:
        checks: list[MetricFeasibilityCheck] = []
        evidence: list[MetricSelectionEvidence] = []
        rejection: RejectionReason | None = None
        alternatives: list[MetricType] = []

        # metric_in_catalog
        in_catalog = metric_req.metric in METRIC_CATALOG
        checks.append(
            _check(
                "metric_in_catalog",
                "passed" if in_catalog else "failed",
                f"metric={metric_req.metric}",
            )
        )
        if not in_catalog:
            rejection = "metric_not_in_catalog"

        definition = get_metric_definition(metric_req.metric) if in_catalog else None

        # semantic rule
        rule_result = validate_metric_for_goal(
            metric_req.inferred_goal,
            metric_req.metric,
            user_explicit=metric_req.user_explicit,
            role=metric_req.role,
            claimed_rule_ids=list(metric_req.claimed_rule_ids),
        )
        if rule_result.status == "matched" and rule_result.matched_rule is not None:
            checks.append(
                _check(
                    "semantic_rule_supports_metric",
                    "passed",
                    f"rule={rule_result.matched_rule.rule_id}",
                )
            )
            evidence.append(
                MetricSelectionEvidence(
                    basis_type="semantic_rule",
                    rule_id=rule_result.matched_rule.rule_id,
                    statement=rule_result.matched_rule.statement,
                    source_ref=f"{RULESET_VERSION}:{rule_result.matched_rule.rule_id}",
                )
            )
            if rule_result.matched_rule.rule_id in {
                "TYPICAL_VALUE_MEDIAN_001",
                "EXPLICIT_AVERAGE_MEAN_001",
                "VARIABILITY_STDDEV_001",
            }:
                evidence.append(
                    MetricSelectionEvidence(
                        basis_type="statistical_guidance",
                        rule_id=rule_result.matched_rule.rule_id,
                        statement=rule_result.matched_rule.statement,
                        source_ref=f"{RULESET_VERSION}:statistical_guidance",
                    )
                )
            if "rule_claim_ignored" in rule_result.notes:
                # Evidence stays derived from the matched rule only.
                pass
        elif rule_result.status == "explicit_request_required":
            checks.append(
                _check(
                    "semantic_rule_supports_metric",
                    "passed",
                    "rule matched but explicit request required",
                )
            )
            checks.append(
                _check(
                    "explicit_request_present",
                    "failed",
                    "user_explicit=false",
                )
            )
            rejection = rejection or "explicit_request_required"
            alternatives = list(rule_result.supported_alternatives)
        elif rule_result.status == "role_not_permitted":
            checks.append(
                _check(
                    "semantic_rule_supports_metric",
                    "passed",
                    "rule matched",
                )
            )
            checks.append(
                _check("role_permitted", "failed", "supporting_only rule used as primary")
            )
            rejection = rejection or "role_not_permitted"
            alternatives = list(rule_result.supported_alternatives)
        else:
            checks.append(
                _check(
                    "semantic_rule_supports_metric",
                    "failed",
                    "no rule permits this goal/metric pair",
                )
            )
            rejection = rejection or "semantic_rule_violation"
            alternatives = list(rule_result.supported_alternatives)

        if metric_req.user_explicit:
            evidence.append(
                MetricSelectionEvidence(
                    basis_type="user_explicit",
                    statement="Metric was declared as explicitly requested by the user.",
                    source_ref="user_request",
                )
            )

        if definition is not None:
            evidence.append(
                MetricSelectionEvidence(
                    basis_type="metric_catalog",
                    statement=f"Catalog metric '{definition.display_label}' ({definition.unit}).",
                    source_ref=f"{METRIC_CATALOG_VERSION}:{metric_req.metric}",
                )
            )

        # Role check when not already failed
        if not any(c.requirement == "role_permitted" for c in checks):
            checks.append(_check("role_permitted", "passed", f"role={metric_req.role}"))

        if not any(c.requirement == "explicit_request_present" for c in checks):
            need_explicit = bool(
                rule_result.matched_rule and rule_result.matched_rule.requires_user_explicit
            )
            if need_explicit:
                checks.append(
                    _check(
                        "explicit_request_present",
                        "passed" if metric_req.user_explicit else "failed",
                        f"user_explicit={metric_req.user_explicit}",
                    )
                )
            else:
                checks.append(_check("explicit_request_present", "not_applicable", "not required"))

        # Per-target data checks
        for target in plan.targets:
            try:
                record = datasets.get(target.dataset_ref)
                checks.append(
                    _check(
                        "dataset_reference_resolvable",
                        "passed",
                        f"{target.target_id}->{target.dataset_ref}",
                    )
                )
            except UnknownDatasetError:
                checks.append(
                    _check(
                        "dataset_reference_resolvable",
                        "failed",
                        f"unknown {target.dataset_ref}",
                    )
                )
                rejection = rejection or "unknown_dataset_reference"
                continue

            zero_ok = bool(definition and definition.zero_observations_allowed)
            empty_ok = record.feature_count > 0 or zero_ok
            checks.append(
                _check(
                    "dataset_not_empty",
                    "passed" if empty_ok else "failed",
                    f"{target.target_id} features={record.feature_count}",
                )
            )
            if not empty_ok:
                rejection = rejection or "empty_dataset"

            if definition and definition.requires_numeric_property:
                assert metric_req.property_key is not None
                numeric_obs = extract_numeric(record, metric_req.property_key)
                enough = numeric_obs.observation_count >= definition.min_observations
                checks.append(
                    _check(
                        "numeric_property_present",
                        "passed" if enough else "failed",
                        (
                            f"{numeric_obs.observation_count} of {numeric_obs.candidate_count} "
                            f"features carry parseable '{metric_req.property_key}' "
                            f"({numeric_obs.missing_count} missing, "
                            f"{numeric_obs.invalid_count} invalid)"
                        ),
                    )
                )
                checks.append(
                    _check(
                        "sufficient_observations",
                        "passed" if enough else "failed",
                        f"{numeric_obs.observation_count} observations, "
                        f"{definition.min_observations} required",
                    )
                )
                if not enough:
                    rejection = rejection or "numeric_property_unavailable"
                    alternatives = alternatives or [
                        "count",
                        "total_area",
                        "mean_area",
                        "median_area",
                    ]
            else:
                checks.append(_check("numeric_property_present", "not_applicable", "not required"))

            if definition and definition.required_geometry == "polygon":
                area_obs = extract_areas(record)
                enough = area_obs.observation_count >= definition.min_observations
                checks.append(
                    _check(
                        "required_geometry_present",
                        "passed" if enough else "failed",
                        f"polygons={area_obs.observation_count} of {area_obs.candidate_count}",
                    )
                )
                if not enough and not (
                    definition.zero_observations_allowed and area_obs.candidate_count == 0
                ):
                    rejection = rejection or "required_geometry_unavailable"
            elif definition and definition.required_geometry == "positioned":
                distance_obs = extract_distances(record, list(target.reference_points))
                positioned = (
                    distance_obs.candidate_count
                    - distance_obs.missing_count
                    - distance_obs.invalid_count
                )
                enough_geom = positioned >= 1 or (
                    definition.zero_observations_allowed and distance_obs.candidate_count == 0
                )
                checks.append(
                    _check(
                        "required_geometry_present",
                        "passed" if enough_geom else "failed",
                        f"positioned={positioned} of {distance_obs.candidate_count}",
                    )
                )
                if not enough_geom:
                    rejection = rejection or "required_geometry_unavailable"
            else:
                checks.append(_check("required_geometry_present", "not_applicable", "any geometry"))

            if definition and definition.requires_analysis_area:
                area_ok = record.scope.area_km2 is not None and record.scope.area_km2 > 0
                checks.append(
                    _check(
                        "analysis_area_available",
                        "passed" if area_ok else "failed",
                        (
                            f"scope={record.scope.scope_kind} area_km2={record.scope.area_km2}"
                            if area_ok
                            else (
                                f"scope={record.scope.scope_kind} has no computable area; "
                                "supply a point+radius or bounding box"
                            )
                        ),
                    )
                )
                if not area_ok:
                    rejection = rejection or "analysis_area_unavailable"
                elif metric_req.metric in {"density", "coverage_percentage"}:
                    checks.append(_check("denominator_non_zero", "passed", "analysis area > 0"))
            else:
                checks.append(_check("analysis_area_available", "not_applicable", "not required"))

            if definition and definition.requires_reference_points:
                needed = definition.min_observations
                have = len(target.reference_points)
                ref_ok = have >= needed
                checks.append(
                    _check(
                        "reference_geometry_available",
                        "passed" if ref_ok else "failed",
                        f"reference_points={have}, required>={needed}",
                    )
                )
                if not ref_ok:
                    rejection = rejection or "reference_geometry_unavailable"
                    if have >= 1:
                        alternatives = alternatives or ["nearest_distance", "count"]
                    # Also mark sufficient_observations for distance mean/median
                    checks.append(
                        _check(
                            "sufficient_observations",
                            "failed",
                            f"{have} access distances, {needed} required",
                        )
                    )
                    rejection = rejection or "insufficient_observations"
            else:
                checks.append(
                    _check(
                        "reference_geometry_available",
                        "not_applicable",
                        "not required",
                    )
                )

            if definition and definition.requires_numeric_property:
                unit_ok = bool(metric_req.property_key)
                checks.append(
                    _check(
                        "unit_defined",
                        "passed" if unit_ok else "failed",
                        f"property_key={metric_req.property_key}",
                    )
                )
                if not unit_ok:
                    rejection = rejection or "unit_undefined"
            else:
                checks.append(
                    _check(
                        "unit_defined", "passed", f"unit={definition.unit if definition else '-'}"
                    )
                )

        checks.append(
            _check(
                "comparable_target_scopes",
                "passed" if scopes_ok else "failed",
                scopes_detail,
            )
        )
        if not scopes_ok:
            rejection = rejection or "incomparable_target_scopes"

        if metric_req.metric == "ratio":
            pair_ok = (
                metric_req.ratio is not None
                and (metric_req.ratio.numerator, metric_req.ratio.denominator)
                in SUPPORTED_RATIO_PAIRS
            )
            checks.append(
                _check(
                    "ratio_pair_supported",
                    "passed" if pair_ok else "failed",
                    (
                        f"{metric_req.ratio.numerator}/{metric_req.ratio.denominator}"
                        if metric_req.ratio
                        else "missing ratio spec"
                    ),
                )
            )
            if not pair_ok:
                rejection = rejection or "unsupported_ratio_pair"
            # denominator check deferred to engine; mark n/a unless we can peek count
            checks.append(
                _check("denominator_non_zero", "not_applicable", "checked at compute time")
            )
        else:
            checks.append(_check("ratio_pair_supported", "not_applicable", "not a ratio metric"))

        # sufficient_observations for count/density always pass when zero_ok
        if (
            definition
            and metric_req.metric in {"count", "density"}
            and not any(c.requirement == "sufficient_observations" for c in checks)
        ):
            checks.append(_check("sufficient_observations", "passed", "zero observations allowed"))

        if not any(c.requirement == "sufficient_observations" for c in checks):
            checks.append(_check("sufficient_observations", "not_applicable", "covered elsewhere"))
        if not any(c.requirement == "denominator_non_zero" for c in checks):
            checks.append(_check("denominator_non_zero", "not_applicable", "not required"))

        failed = any(c.status == "failed" for c in checks)
        return MetricSelectionTrace(
            metric=metric_req.metric,
            role=metric_req.role,
            inferred_goal=metric_req.inferred_goal,
            property_key=metric_req.property_key,
            target_ids=[t.target_id for t in plan.targets],
            evidence=evidence,
            feasibility_checks=_dedupe_checks(checks),
            final_status="rejected" if failed or rejection else "executed",
            replacement_metric=None,
            rejection_reason=rejection if failed or rejection else None,
            supported_alternatives=list(dict.fromkeys(alternatives)),
            attempt_index=attempt_index,
        )

    def _comparable_scopes(
        self,
        plan: AnalysisPlan,
        datasets: DatasetRegistry,
    ) -> tuple[bool, str]:
        if len(plan.targets) < 2:
            return True, "single target"
        records = []
        for target in plan.targets:
            try:
                records.append(datasets.get(target.dataset_ref))
            except UnknownDatasetError:
                return False, "one or more dataset refs unresolved"
        kinds = {r.scope.scope_kind for r in records}
        tags = {r.resolved_tags for r in records}
        if len(kinds) != 1:
            return False, f"mixed scope kinds: {sorted(kinds)}"
        if len(tags) != 1:
            return False, "resolved tags differ across targets"
        if next(iter(kinds)) == "point":
            radii = {r.scope.radius_m for r in records}
            if len(radii) != 1:
                ordered = sorted(r for r in radii if r is not None)
                return False, f"point radii differ: {ordered}"
        return True, f"comparable scope={next(iter(kinds))} tags={next(iter(tags))}"


def _check(
    requirement: FeasibilityRequirement,
    status: CheckStatus,
    detail: str,
) -> MetricFeasibilityCheck:
    return MetricFeasibilityCheck(
        requirement=requirement,
        status=status,
        detail=detail[:240],
    )


def _dedupe_checks(checks: list[MetricFeasibilityCheck]) -> list[MetricFeasibilityCheck]:
    """Keep first failure per requirement, else first occurrence."""
    by_req: dict[FeasibilityRequirement, MetricFeasibilityCheck] = {}
    for check in checks:
        existing = by_req.get(check.requirement)
        if existing is None or (existing.status != "failed" and check.status == "failed"):
            by_req[check.requirement] = check
    # Preserve a stable order matching the requirement list intent.
    order: list[FeasibilityRequirement] = [
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
    return [by_req[r] for r in order if r in by_req]
