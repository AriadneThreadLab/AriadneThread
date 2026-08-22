"""Build an immutable execution snapshot from a finished request."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from app.agent.contracts import GeoAgentRequest, GeoAgentResponse
from app.analytics.catalog import METRIC_CATALOG_VERSION
from app.analytics.contracts import AnalysisGoal
from app.analytics.datasets import DatasetRecord
from app.analytics.rules import RULESET_VERSION
from app.execution_memory.contracts import (
    ComparisonSnapshot,
    DocumentationSourceSnapshot,
    ExecutionDatasetSnapshot,
    ExecutionOutcome,
    ExecutionSnapshot,
    ExecutionTargetSnapshot,
    MetricSnapshot,
    TrustedPlaceSnapshot,
    strip_hidden_reasoning,
)
from app.osm.contracts import OSM_ATTRIBUTION
from app.places.contracts import ResolvedPlaceRecord
from app.tools.context import ToolContext


def build_snapshot(
    *,
    request: GeoAgentRequest,
    response: GeoAgentResponse,
    tool_context: ToolContext,
    conversation_id: str,
    parent_execution_id: UUID | None,
    provider: str | None,
    prompt_version: str | None,
    tool_schema_version: str | None,
    radius_m: int | None = None,
    feature_concept: str | None = None,
    analysis_goal: str | None = None,
    inferred_goal: AnalysisGoal | None = None,
) -> ExecutionSnapshot | None:
    """Create a snapshot after a successful or partial analytical run."""
    analysis = response.analysis
    if (analysis is None or analysis.status == "abandoned") and not tool_context.datasets.refs():
        return None
    targets: list[ExecutionTargetSnapshot] = []
    datasets: list[ExecutionDatasetSnapshot] = []
    analysis_targets = (
        list(analysis.result.targets)
        if analysis is not None and analysis.result is not None
        else []
    )
    plan_targets = list(analysis.plan.targets) if analysis is not None else []

    place_by_index = list(tool_context.places.refs())
    dataset_refs = list(tool_context.datasets.refs())

    count = max(len(plan_targets), len(dataset_refs), len(place_by_index))
    if count < 1:
        return None

    for index in range(count):
        stable_id = f"t{index + 1}"
        plan_target = plan_targets[index] if index < len(plan_targets) else None
        analysis_target = analysis_targets[index] if index < len(analysis_targets) else None
        place_ref = place_by_index[index] if index < len(place_by_index) else None
        dataset_ref = (
            plan_target.dataset_ref
            if plan_target is not None
            else (dataset_refs[index] if index < len(dataset_refs) else None)
        )
        place = tool_context.places.get(place_ref) if place_ref else None
        if place is None:
            continue
        label = (
            (plan_target.label if plan_target is not None else None)
            or (analysis_target.label if analysis_target is not None else None)
            or place.label
        )
        targets.append(
            ExecutionTargetSnapshot(
                stable_id=stable_id,
                label=label[:120],
                original_user_label=label[:120],
                place=_place_snapshot(place),
                request_scoped_place_ref_at_creation=place.place_ref,
            )
        )
        if dataset_ref is None:
            continue
        record = tool_context.datasets.get(dataset_ref)
        datasets.append(_dataset_snapshot(stable_id, record, dataset_ref))

    outcome: ExecutionOutcome = "failed"
    if analysis is not None and analysis.status == "completed":
        outcome = "completed"
    elif datasets:
        outcome = "partial"

    metric = _metric_snapshot(response, inferred_goal)
    comparison = _comparison_snapshot(response)
    concept = feature_concept or (analysis.plan.feature_concept if analysis is not None else "")
    goal = analysis_goal or (analysis.plan.comparison_goal if analysis is not None else "")
    tags = tuple(tool_context.grounding.tags or response.validated_tags)
    resolved_radius = radius_m
    if resolved_radius is None and datasets:
        resolved_radius = datasets[0].scope.radius_m

    return ExecutionSnapshot(
        conversation_id=conversation_id,
        request_id=request.request_id,
        parent_execution_id=parent_execution_id,
        original_user_query=request.message,
        analysis_type=(analysis.plan.analysis_type if analysis is not None else "comparison"),
        analysis_goal=goal,
        feature_concept=concept,
        inferred_goal=inferred_goal or metric.inferred_goal,
        outcome=outcome,
        radius_m=resolved_radius,
        scope_kind=datasets[0].scope.scope_kind if datasets else "point",
        grounding_tags=tags,
        grounding_version=_grounding_version(concept, tags),
        metric=metric,
        targets=targets,
        datasets=datasets,
        documentation_sources=documentation_sources_from_response(response),
        comparison=comparison,
        model=response.model,
        provider=provider,
        prompt_version=prompt_version,
        tool_schema_version=tool_schema_version,
    )


def _place_snapshot(place: ResolvedPlaceRecord) -> TrustedPlaceSnapshot:
    return TrustedPlaceSnapshot(
        query=place.query,
        label=place.label,
        display_name=place.display_name,
        latitude=place.latitude,
        longitude=place.longitude,
        source=place.source,
        source_id=place.source_id,
    )


def _dataset_snapshot(
    stable_id: str,
    record: DatasetRecord,
    dataset_ref: str,
) -> ExecutionDatasetSnapshot:
    query_spec = strip_hidden_reasoning(
        {
            "resolved_tags": list(record.resolved_tags),
            "scope_kind": record.scope.scope_kind,
            "radius_m": record.scope.radius_m,
            "place": record.scope.place,
            "effective_limit": record.effective_limit,
            "truncated": record.truncated,
        }
    )
    return ExecutionDatasetSnapshot(
        execution_dataset_id=uuid4(),
        target_stable_id=stable_id,
        retrieved_at=record.retrieved_at,
        source="live_osm",
        endpoint=record.endpoint,
        attribution=OSM_ATTRIBUTION,
        effective_limit=record.effective_limit,
        truncated=record.truncated,
        feature_count=record.feature_count,
        resolved_tags=record.resolved_tags,
        scope=record.scope,
        query_spec=query_spec if isinstance(query_spec, dict) else {},
        feature_collection=record.feature_collection,
        request_scoped_dataset_ref_at_creation=dataset_ref,
    )


def _metric_snapshot(
    response: GeoAgentResponse,
    inferred_goal: AnalysisGoal | None,
) -> MetricSnapshot:
    analysis = response.analysis
    if analysis is None:
        return MetricSnapshot()
    primary = analysis.decision_trace.final_primary_metric
    goal: AnalysisGoal | None = inferred_goal
    rule_id = None
    for selection in analysis.decision_trace.metric_selections:
        if selection.final_status == "executed" and selection.role == "primary":
            goal = selection.inferred_goal
            if selection.evidence:
                rule_id = selection.evidence[0].rule_id
            break
    return MetricSnapshot(
        metric=primary,
        role="primary",
        inferred_goal=goal,
        rule_id=rule_id,
        catalog_version=analysis.decision_trace.metric_catalog_version,
        ruleset_version=analysis.decision_trace.ruleset_version,
        feasibility_ok=analysis.status == "completed",
    )


def _comparison_snapshot(response: GeoAgentResponse) -> ComparisonSnapshot | None:
    analysis = response.analysis
    if analysis is None or analysis.comparison is None:
        return None
    comparison = analysis.comparison
    ranking: list[dict[str, Any]] = []
    truncated = False
    if comparison.primary is not None:
        for item in comparison.primary.ranking:
            ranking.append(
                strip_hidden_reasoning(
                    {
                        "target_id": item.target_id,
                        "label": item.label,
                        "value": item.value,
                        "status": item.status,
                    }
                )
            )
    if analysis.result is not None:
        truncated = any(
            target.data_provenance.truncated or target.data_provenance.limit_reached
            for target in analysis.result.targets
        )
    warnings = list(analysis.result.warnings) if analysis.result is not None else []
    return ComparisonSnapshot(
        overall_statement=comparison.overall_statement,
        overall_confidence=comparison.overall_confidence,
        ranking=ranking,
        warnings=warnings,
        truncated=truncated,
    )


def documentation_sources_from_response(
    response: GeoAgentResponse,
) -> tuple[DocumentationSourceSnapshot, ...]:
    """Persist OSM Wiki citations that grounded this run (not passage bodies)."""
    seen: set[tuple[str, str | None, str]] = set()
    items: list[DocumentationSourceSnapshot] = []
    for passage in response.passages:
        title = passage.document_title.strip()[:160]
        url = passage.source_url.strip()[:1024]
        if not title or not url:
            continue
        section = passage.section.strip()[:160] if passage.section else None
        key = (title, section, url)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            DocumentationSourceSnapshot(
                title=title,
                section=section or None,
                url=url,
                score=passage.score,
            )
        )
    if items:
        return tuple(items)
    analysis = response.analysis
    if analysis is None or analysis.result is None:
        return ()
    for target in analysis.result.targets:
        for source in target.data_provenance.grounding_sources:
            title = source.title.strip()[:160]
            url = (source.url or "").strip()[:1024]
            if not title or not url:
                continue
            key = (title, None, url)
            if key in seen:
                continue
            seen.add(key)
            items.append(DocumentationSourceSnapshot(title=title, section=None, url=url, score=0.0))
    return tuple(items)


def _grounding_version(concept: str, tags: tuple[str, ...]) -> str:
    tag_part = ",".join(tags) if tags else ""
    return (
        f"concept:{concept};tags:{tag_part};"
        f"catalog:{METRIC_CATALOG_VERSION};rules:{RULESET_VERSION}"
    )


def assert_no_hidden_reasoning(snapshot: ExecutionSnapshot) -> None:
    dumped = snapshot.model_dump(mode="json")
    blob = str(dumped).lower()
    if "<think" in blob or "reasoning_content" in blob:
        raise ValueError("execution snapshot must not persist hidden reasoning")
