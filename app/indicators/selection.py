"""Deterministic domain detection, candidate retrieval, and indicator selection.

The LLM may propose a catalog id. This module interprets the question, retrieves
candidates from the Indicator Catalog, and either honours an eligible proposal
or selects deterministically. It never invents ids, formulas, or datasets.

Not wired into the agent loop in Phase 2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.analytics.contracts import AnalysisGoal, RuleId
from app.analytics.methods import METHOD_REGISTRY
from app.analytics.rules import RULESET_VERSION, SEMANTIC_RULES
from app.core.errors import UnknownIndicatorError
from app.indicators.catalog import (
    INDICATOR_CATALOG,
    INDICATOR_CATALOG_VERSION,
    IndicatorCatalog,
)
from app.indicators.contracts import (
    MAX_SUPPORTING_INDICATORS,
    AnalysisDomainSelection,
    DomainId,
    IndicatorCandidate,
    IndicatorDefinition,
    IndicatorRejection,
    IndicatorRejectionReason,
    IndicatorSelection,
    IndicatorSelectionEvidence,
    IndicatorSelectionTrace,
    ProposedIndicatorChoice,
)

_GREEN_SPACE = re.compile(
    r"\b(green[\s-]?spaces?|green[\s-]?areas?|green[\s-]?environment|"
    r"greenery|urban forest)\b",
    re.IGNORECASE,
)
_MOBILITY = re.compile(
    r"\b(road accessibility|road access|road density|road network|"
    r"street network|intersection density|intersections?|highways?|"
    r"roads?|mobility)\b",
    re.IGNORECASE,
)
_URBAN_SERVICES = re.compile(
    r"\b(urban services?|points? of interest|\bpois?\b|amenities|"
    r"service accessibility|service diversity|service access|"
    r"hospitals?|schools?|pharmac(?:y|ies)|supermarkets?)\b",
    re.IGNORECASE,
)

_DIVERSITY = re.compile(
    r"\b(diversity|diverse|variety|evenness|mix of types|shannon)\b",
    re.IGNORECASE,
)
_ACCESSIBILITY = re.compile(
    r"\b(accessibility|accessible|better access|access to|access|how close|nearest|closer)\b",
    re.IGNORECASE,
)
_CONCENTRATION = re.compile(
    r"\b(density|denser|concentration|per km|per square|intersections? per)\b",
    re.IGNORECASE,
)
_COVERAGE = re.compile(
    r"\b(ratio|coverage|proportion|share of|how much green|green provision)\b",
    re.IGNORECASE,
)
_ABUNDANCE = re.compile(
    r"\b(how many|more parks|abundance|availability|count them|\bcount\b)\b",
    re.IGNORECASE,
)
_PARKS = re.compile(r"\bparks?\b", re.IGNORECASE)
_INTERSECTION = re.compile(r"\bintersections?\b", re.IGNORECASE)
_POI = re.compile(r"\b(poi|points? of interest)\b", re.IGNORECASE)
_ROAD_DENSITY = re.compile(r"\broad density\b", re.IGNORECASE)
_BROAD_GREEN = re.compile(
    r"\bgreen (?:environment|areas?|spaces?)\b",
    re.IGNORECASE,
)

_DOMAIN_PATTERNS: tuple[tuple[DomainId, re.Pattern[str], str], ...] = (
    ("green_space", _GREEN_SPACE, "User asked about green space."),
    ("mobility", _MOBILITY, "User asked about roads, highways, or intersections."),
    ("urban_services", _URBAN_SERVICES, "User asked about urban services or POIs."),
)

_DEFAULT_GOAL: dict[DomainId, AnalysisGoal] = {
    "core": "abundance",
    "green_space": "coverage",
    "mobility": "concentration",
    "urban_services": "concentration",
}

_GOAL_RULE: dict[AnalysisGoal, RuleId] = {
    "abundance": "ABUNDANCE_COUNT_001",
    "concentration": "CONCENTRATION_DENSITY_001",
    "accessibility": "ACCESSIBILITY_DISTANCE_001",
    "coverage": "COVERAGE_PERCENTAGE_001",
    "variability": "VARIABILITY_STDDEV_001",
    "total_provision": "GREENSPACE_TOTAL_AREA_001",
    "typical_value": "TYPICAL_VALUE_MEDIAN_001",
    "relative_share": "RELATIVE_SHARE_RATIO_001",
}


@dataclass(frozen=True, slots=True)
class GoalInterpretation:
    """Inferred AnalysisGoal plus whether the user stated it explicitly."""

    goal: AnalysisGoal
    explicit: bool
    evidence: tuple[IndicatorSelectionEvidence, ...]


class DomainResolver:
    """Detect zero or more thematic domains from a user question."""

    def resolve(self, user_message: str) -> AnalysisDomainSelection:
        hits: list[tuple[int, DomainId, str]] = []
        for domain, pattern, statement in _DOMAIN_PATTERNS:
            match = pattern.search(user_message)
            if match is None:
                continue
            hits.append((match.start(), domain, statement))
        hits.sort(key=lambda item: item[0])

        evidence: tuple[IndicatorSelectionEvidence, ...]
        domains: tuple[DomainId, ...]
        if hits:
            domains = tuple(domain for _, domain, _ in hits)
            basis = "user_explicit"
            evidence = tuple(
                IndicatorSelectionEvidence(
                    basis_type="user_explicit",
                    statement=statement[:240],
                    source_ref=f"user_message:{domain}",
                )
                for _, domain, statement in hits
            )
        else:
            domains = ("core",)
            basis = "fallback"
            evidence = (
                IndicatorSelectionEvidence(
                    basis_type="fallback",
                    statement="No thematic domain matched; using core feature indicators.",
                    source_ref="indicator-catalog-1:core",
                ),
            )

        if _PARKS.search(user_message) and _ABUNDANCE.search(user_message):
            if "core" not in domains:
                domains = ("core", *domains)
            evidence = (
                *evidence,
                IndicatorSelectionEvidence(
                    basis_type="user_explicit",
                    statement="Count/abundance language about parks uses feature_count.",
                    source_ref="user_message:core",
                ),
            )

        return AnalysisDomainSelection(domains=domains, basis=basis, evidence=evidence)


class GoalInterpreter:
    """Map phrasing onto the existing AnalysisGoal vocabulary."""

    def interpret(
        self,
        user_message: str,
        domains: tuple[DomainId, ...],
    ) -> GoalInterpretation:
        explicit_goal, statement, rule_id = _explicit_goal(user_message)
        if explicit_goal is not None:
            evidence = (
                IndicatorSelectionEvidence(
                    basis_type="semantic_rule" if rule_id else "user_explicit",
                    rule_id=rule_id,
                    statement=statement or "Goal inferred from user phrasing.",
                    source_ref=(f"{RULESET_VERSION}:{rule_id}" if rule_id else "user_message:goal"),
                ),
            )
            return GoalInterpretation(goal=explicit_goal, explicit=True, evidence=evidence)

        primary_domain = domains[0] if domains else "core"
        goal = _DEFAULT_GOAL[primary_domain]
        return GoalInterpretation(
            goal=goal,
            explicit=False,
            evidence=(
                IndicatorSelectionEvidence(
                    basis_type="fallback",
                    statement=f"No explicit goal; default for domain {primary_domain} is {goal}.",
                    source_ref=f"indicator-catalog-1:default_goal:{primary_domain}",
                ),
            ),
        )


class IndicatorSelector:
    """Retrieve catalog candidates and select a primary plus optional supporting."""

    def __init__(self, catalog: IndicatorCatalog | None = None) -> None:
        self._catalog = catalog if catalog is not None else INDICATOR_CATALOG
        self._domains = DomainResolver()
        self._goals = GoalInterpreter()

    def plan(
        self,
        user_message: str,
        *,
        proposal: ProposedIndicatorChoice | None = None,
    ) -> IndicatorSelectionTrace:
        """Run domain detection → candidates → validated selection."""
        if proposal is not None:
            self._require_catalog_ids(proposal)

        domain_selection = self._domains.resolve(user_message)
        goal_info = self._goals.interpret(user_message, domain_selection.domains)
        pool = self._domain_pool(domain_selection.domains)
        broad = _is_broad_query(user_message, goal_info.explicit, domain_selection.domains)

        eligible, rejected, candidate_rows = self._classify(
            pool,
            goal_info,
            domains=domain_selection.domains,
            broad=broad,
        )
        ranked = self._rank(eligible, user_message, goal_info.goal)
        candidate_rows = _apply_ranks(candidate_rows, ranked)

        selection, extra_rejected, reason, status = self._choose(
            ranked,
            proposal=proposal,
            broad=broad,
        )
        rejected = rejected + extra_rejected

        evidence = domain_selection.evidence + goal_info.evidence
        cited: tuple[RuleId, ...] = tuple(
            item.rule_id for item in evidence if item.rule_id is not None
        )
        if selection.primary_indicator_id is not None:
            definition = self._catalog.get_indicator(selection.primary_indicator_id)
            evidence = (
                *evidence,
                IndicatorSelectionEvidence(
                    basis_type="metric_catalog",
                    statement=(
                        f"Selected {definition.display_label} "
                        f"({definition.method_id}, {definition.unit})."
                    )[:240],
                    source_ref=f"{INDICATOR_CATALOG_VERSION}:{definition.indicator_id}",
                ),
            )
            rule_id = _GOAL_RULE.get(goal_info.goal)
            if rule_id is not None:
                evidence = (
                    *evidence,
                    IndicatorSelectionEvidence(
                        basis_type="semantic_rule",
                        rule_id=rule_id,
                        statement=_rule_statement(rule_id),
                        source_ref=f"{RULESET_VERSION}:{rule_id}",
                    ),
                )
                if rule_id not in cited:
                    cited = (*cited, rule_id)

        required, methodology, parameters = self._selected_bindings(selection)
        return IndicatorSelectionTrace(
            inferred_goal=goal_info.goal,
            goal_explicit=goal_info.explicit,
            domain_selection=domain_selection,
            candidate_indicators=tuple(item.indicator_id for item in pool),
            candidates=candidate_rows,
            rejected_indicators=tuple(rejected),
            selection=selection,
            selection_reason=reason[:240],
            selection_evidence=evidence,
            required_data=required,
            methodology=methodology,
            parameters=parameters,
            execution_status=status,
            indicator_catalog_version=self._catalog.version,
            ruleset_version=RULESET_VERSION,
            cited_rule_ids=cited,
        )

    def _require_catalog_ids(self, proposal: ProposedIndicatorChoice) -> None:
        ids = (proposal.primary_indicator_id, *proposal.supporting_indicator_ids)
        known = ", ".join(self._catalog.indicator_ids()) or "none"
        for indicator_id in ids:
            try:
                self._catalog.get_indicator(indicator_id)
            except UnknownIndicatorError:
                raise UnknownIndicatorError(
                    f"unknown indicator_id '{indicator_id}'; available: {known}"
                ) from None

    def _domain_pool(self, domains: tuple[DomainId, ...]) -> tuple[IndicatorDefinition, ...]:
        seen: set[str] = set()
        pool: list[IndicatorDefinition] = []
        for domain in domains:
            for definition in self._catalog.list_by_domain(domain):
                if definition.status == "deprecated" or definition.indicator_id in seen:
                    continue
                seen.add(definition.indicator_id)
                pool.append(definition)
        return tuple(pool)

    def _classify(
        self,
        pool: tuple[IndicatorDefinition, ...],
        goal_info: GoalInterpretation,
        *,
        domains: tuple[DomainId, ...],
        broad: bool,
    ) -> tuple[
        list[IndicatorDefinition],
        list[IndicatorRejection],
        tuple[IndicatorCandidate, ...],
    ]:
        catalog_eligible = {
            item.indicator_id
            for domain in domains
            for item in self._catalog.find_candidates(domain, goal_info.goal)
        }
        eligible: list[IndicatorDefinition] = []
        rejected: list[IndicatorRejection] = []
        rows: list[IndicatorCandidate] = []
        for definition in pool:
            ok = broad or definition.indicator_id in catalog_eligible
            if ok:
                eligible.append(definition)
                rows.append(_candidate(definition, eligible=True, reason=None))
            else:
                reason: IndicatorRejectionReason = "goal_mismatch"
                rejected.append(
                    IndicatorRejection(
                        indicator_id=definition.indicator_id,
                        reason=reason,
                        detail=f"goal={goal_info.goal} not in {list(definition.goals)}"[:240],
                    )
                )
                rows.append(_candidate(definition, eligible=False, reason=reason))
        return eligible, rejected, tuple(rows)

    def _rank(
        self,
        eligible: list[IndicatorDefinition],
        user_message: str,
        goal: AnalysisGoal,
    ) -> list[IndicatorDefinition]:
        permitted = _permitted_primitives(goal)
        scored: list[tuple[int, int, int, IndicatorDefinition]] = []
        for index, definition in enumerate(eligible):
            spec = METHOD_REGISTRY[definition.method_id]
            rule_boost = 2 if permitted and set(spec.metric_primitives) & permitted else 0
            lexical = _lexical_score(definition, user_message)
            scored.append((-rule_boost, -lexical, index, definition))
        scored.sort()
        return [item[3] for item in scored]

    def _choose(
        self,
        ranked: list[IndicatorDefinition],
        *,
        proposal: ProposedIndicatorChoice | None,
        broad: bool,
    ) -> tuple[IndicatorSelection, list[IndicatorRejection], str, str]:
        extra: list[IndicatorRejection] = []
        eligible_ids = {item.indicator_id for item in ranked}

        if proposal is not None:
            if proposal.primary_indicator_id not in eligible_ids:
                extra.append(
                    IndicatorRejection(
                        indicator_id=proposal.primary_indicator_id,
                        reason="goal_mismatch",
                        detail="Proposed primary indicator is not eligible for this request.",
                    )
                )
            else:
                supporting: list[str] = []
                for sid in proposal.supporting_indicator_ids:
                    if sid not in eligible_ids:
                        extra.append(
                            IndicatorRejection(
                                indicator_id=sid,
                                reason="goal_mismatch",
                                detail="Proposed supporting indicator is not eligible.",
                            )
                        )
                        continue
                    supporting.append(sid)
                reason = proposal.selection_reason.strip() or (
                    f"LLM selected {proposal.primary_indicator_id} from eligible catalog ids."
                )
                return (
                    IndicatorSelection(
                        primary_indicator_id=proposal.primary_indicator_id,
                        supporting_indicator_ids=tuple(supporting[:MAX_SUPPORTING_INDICATORS]),
                    ),
                    extra,
                    reason,
                    "completed",
                )

        if not ranked:
            return (
                IndicatorSelection(),
                extra,
                "No catalog indicator is eligible for the inferred domain and goal.",
                "rejected",
            )

        primary = ranked[0]
        supporting_defs = ranked[1:] if broad else []
        supporting_ids = tuple(
            item.indicator_id for item in supporting_defs[:MAX_SUPPORTING_INDICATORS]
        )
        if supporting_ids:
            reason = (
                f"Selected {primary.indicator_id} as primary with supporting "
                f"{', '.join(supporting_ids)}."
            )
        else:
            reason = f"Selected {primary.indicator_id} as the eligible primary indicator."
        return (
            IndicatorSelection(
                primary_indicator_id=primary.indicator_id,
                supporting_indicator_ids=supporting_ids,
            ),
            extra,
            reason,
            "completed",
        )

    def _selected_bindings(
        self, selection: IndicatorSelection
    ) -> tuple[tuple[str, ...], str | None, dict[str, float]]:
        if selection.primary_indicator_id is None:
            return (), None, {}
        definition = self._catalog.get_indicator(selection.primary_indicator_id)
        required = tuple(item.requirement_id for item in definition.requirements)
        return required, definition.method_id, dict(definition.parameters)


def plan_indicator_analysis(
    user_message: str,
    *,
    proposal: ProposedIndicatorChoice | None = None,
    catalog: IndicatorCatalog | None = None,
) -> IndicatorSelectionTrace:
    """Convenience entry: question (+ optional LLM proposal) → selection trace."""
    return IndicatorSelector(catalog).plan(user_message, proposal=proposal)


def _is_broad_query(
    user_message: str,
    goal_explicit: bool,
    domains: tuple[DomainId, ...],
) -> bool:
    if goal_explicit:
        return False
    return "green_space" in domains and _BROAD_GREEN.search(user_message) is not None


def _candidate(
    definition: IndicatorDefinition,
    *,
    eligible: bool,
    reason: IndicatorRejectionReason | None,
) -> IndicatorCandidate:
    return IndicatorCandidate(
        indicator_id=definition.indicator_id,
        domain=definition.domain,
        display_label=definition.display_label,
        goals=definition.goals,
        method_id=definition.method_id,
        eligible=eligible,
        rank=0,
        ineligibility_reason=reason,
    )


def _apply_ranks(
    rows: tuple[IndicatorCandidate, ...],
    ranked: list[IndicatorDefinition],
) -> tuple[IndicatorCandidate, ...]:
    order = {item.indicator_id: index + 1 for index, item in enumerate(ranked)}
    updated: list[IndicatorCandidate] = []
    for row in rows:
        rank = order.get(row.indicator_id, 0)
        updated.append(row.model_copy(update={"rank": rank}))
    return tuple(updated)


def _explicit_goal(
    user_message: str,
) -> tuple[AnalysisGoal | None, str | None, RuleId | None]:
    if _DIVERSITY.search(user_message):
        return (
            "variability",
            "Diversity language maps to a variability goal.",
            "VARIABILITY_STDDEV_001",
        )
    if _ACCESSIBILITY.search(user_message):
        return (
            "accessibility",
            "Access language maps to an accessibility goal.",
            "ACCESSIBILITY_DISTANCE_001",
        )
    if _COVERAGE.search(user_message):
        return (
            "coverage",
            "Ratio/coverage language maps to a coverage goal.",
            "COVERAGE_PERCENTAGE_001",
        )
    if _CONCENTRATION.search(user_message):
        return (
            "concentration",
            "Density/concentration language maps to a concentration goal.",
            "CONCENTRATION_DENSITY_001",
        )
    if _ABUNDANCE.search(user_message):
        return (
            "abundance",
            "Count/abundance language maps to an abundance goal.",
            "ABUNDANCE_COUNT_001",
        )
    return None, None, None


def _permitted_primitives(goal: AnalysisGoal) -> frozenset[str]:
    permitted: set[str] = set()
    for rule in SEMANTIC_RULES:
        if rule.supporting_only:
            continue
        if rule.goal == goal or rule.any_goal:
            permitted.update(rule.permitted_metrics)
    return frozenset(permitted)


def _rule_statement(rule_id: RuleId) -> str:
    for rule in SEMANTIC_RULES:
        if rule.rule_id == rule_id:
            return rule.statement[:240]
    return f"Semantic rule {rule_id}."


def _lexical_score(definition: IndicatorDefinition, user_message: str) -> int:
    text = user_message.lower()
    score = 0
    if definition.indicator_id == "intersection_density" and _INTERSECTION.search(user_message):
        score += 8
    if definition.indicator_id == "road_density" and _ROAD_DENSITY.search(user_message):
        score += 8
    if definition.indicator_id == "poi_density" and _POI.search(user_message):
        score += 8
    if definition.indicator_id == "feature_count" and _PARKS.search(user_message):
        score += 6
    for token in definition.indicator_id.split("_"):
        if token in {"green", "space", "urban", "service", "road"}:
            continue
        if token in text:
            score += 3
    return score
