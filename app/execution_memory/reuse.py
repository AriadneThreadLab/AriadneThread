"""Validate whether a prior execution can be reused for a follow-up."""

from __future__ import annotations

from collections.abc import Container
from datetime import datetime, timezone

from app.agent.comparison_workflow import canonicalize_place_query, tags_for_feature_concept
from app.analytics.contracts import AnalysisGoal
from app.execution_memory.contracts import (
    AnalysisPattern,
    DatasetSource,
    ExecutionDatasetSnapshot,
    ExecutionSnapshot,
    FollowUpResolution,
    IncrementalComparisonPlan,
    IncrementalTarget,
    PatternReuseAssessment,
    ReuseAssessment,
    ReuseDecision,
    TargetDatasetAction,
)
from app.execution_memory.metric_revalidation import revalidate_metric


class ExecutionReuseValidator:
    """Deterministic reuse / refresh / replan decisions."""

    def __init__(self, *, osm_ttl_seconds: int) -> None:
        self._osm_ttl_seconds = osm_ttl_seconds

    def assess(
        self,
        snapshot: ExecutionSnapshot,
        follow_up: FollowUpResolution,
        *,
        now: datetime | None = None,
    ) -> ReuseAssessment:
        clock = now or datetime.now(tz=timezone.utc)
        intents = {follow_up.follow_up_type, *follow_up.extra_intents}
        reasons: list[str] = []

        feature_concept = follow_up.new_feature_concept or snapshot.feature_concept
        comparison_goal = snapshot.analysis_goal
        inferred_goal: AnalysisGoal = follow_up.new_goal or snapshot.inferred_goal or "abundance"
        radius_m = follow_up.new_radius_m or snapshot.radius_m or 2000
        re_ground = False
        grounding_tags = snapshot.grounding_tags

        if follow_up.follow_up_type == "NEW_ANALYSIS":
            return ReuseAssessment(
                decision="CANNOT_REUSE",
                reasons=("new_analysis",),
                metric_revalidation=revalidate_metric(
                    previous=snapshot.metric, new_goal=inferred_goal
                ),
                feature_concept=feature_concept,
                comparison_goal=comparison_goal,
                inferred_goal=inferred_goal,
                radius_m=radius_m,
                grounding_tags=grounding_tags,
                re_ground=True,
            )

        if "CHANGE_FEATURE_CONCEPT" in intents:
            re_ground = True
            tags = tags_for_feature_concept(feature_concept)
            grounding_tags = tuple(tags) if tags else ()
            reasons.append("feature_concept_incompatible")
        elif tuple(grounding_tags) != tuple(
            tags_for_feature_concept(feature_concept) or grounding_tags
        ):
            re_ground = True
            reasons.append("grounding_tags_incompatible")
        else:
            reasons.append("feature_concept_compatible")

        if snapshot.scope_kind != "point":
            reasons.append(
                "scope_type_compatible" if snapshot.scope_kind == "point" else "scope_type_reused"
            )
        if radius_m != snapshot.radius_m:
            reasons.append("radius_incompatible")
        else:
            reasons.append("radius_compatible")

        metric_rev = revalidate_metric(previous=snapshot.metric, new_goal=inferred_goal)
        if metric_rev.status == "accepted":
            reasons.append("metric_semantics_valid")
        else:
            reasons.append("metric_revalidation_rejected")
            comparison_goal = (
                "accessibility comparison"
                if inferred_goal == "accessibility"
                else snapshot.analysis_goal
            )

        final_metric = (
            snapshot.metric.metric
            if metric_rev.status == "accepted"
            else metric_rev.replacement_metric
        )

        radius_invalidates = radius_m != snapshot.radius_m
        force_refresh = (
            follow_up.refresh_requested
            or "REFRESH_DATA" in intents
            or radius_invalidates
            or re_ground
        )
        keep_stale = follow_up.keep_previous_data and not radius_invalidates and not re_ground

        # The target set always comes from the current request. Previous
        # targets are only inherited through explicit reference resolution
        # (``requested_labels`` already carries them for anaphoric messages),
        # never because they happen to sit in execution history.
        final_labels = list(follow_up.requested_labels)
        if not final_labels:
            final_labels = list(follow_up.preserved_labels) or [t.label for t in snapshot.targets]
            for label in follow_up.added_labels:
                if not any(_same_label(label, existing) for existing in final_labels):
                    final_labels.append(label)

        actions: list[TargetDatasetAction] = []
        for index, label in enumerate(final_labels, start=1):
            prior = next((t for t in snapshot.targets if _same_label(t.label, label)), None)
            if prior is None:
                actions.append(
                    TargetDatasetAction(
                        stable_id=f"t{index}",
                        label=label,
                        source="new",
                        reasons=("new_target",),
                    )
                )
                continue
            dataset = _dataset_for_target(snapshot, prior.stable_id)
            source, target_reasons = self._dataset_action(
                dataset,
                force_refresh=force_refresh,
                keep_stale=keep_stale,
                inferred_goal=inferred_goal,
                now=clock,
            )
            actions.append(
                TargetDatasetAction(
                    stable_id=f"t{index}",
                    label=prior.label,
                    source=source,
                    reasons=tuple(target_reasons),
                    persistent_dataset_id=None
                    if source != "reused"
                    else (dataset.execution_dataset_id if dataset else None),
                )
            )

        decision = self._decision(
            intents=intents,
            actions=actions,
            metric_rev_status=metric_rev.status,
            radius_invalidates=radius_invalidates,
            re_ground=re_ground,
            truncated_blocked=any("truncated_limit_hit" in item.reasons for item in actions),
        )
        if metric_rev.status == "rejected" and decision == "FULL_REUSE":
            decision = "REVALIDATE_METRIC"
            reasons.append("metric_must_be_replaced")

        return ReuseAssessment(
            decision=decision,
            reasons=tuple(dict.fromkeys(reasons)),
            metric_revalidation=metric_rev,
            target_actions=tuple(actions),
            feature_concept=feature_concept,
            comparison_goal=comparison_goal,
            inferred_goal=inferred_goal,
            radius_m=radius_m,
            grounding_tags=grounding_tags,
            re_ground=re_ground,
            final_metric=final_metric,
        )

    def build_incremental_plan(
        self,
        snapshot: ExecutionSnapshot,
        follow_up: FollowUpResolution,
        assessment: ReuseAssessment,
        user_message: str = "",
        *,
        pattern: AnalysisPattern | None = None,
        pattern_reuse: PatternReuseAssessment | None = None,
    ) -> IncrementalComparisonPlan:
        targets: list[IncrementalTarget] = []
        for action in assessment.target_actions:
            prior = next((t for t in snapshot.targets if _same_label(t.label, action.label)), None)
            if action.source == "new" or prior is None:
                targets.append(
                    IncrementalTarget(
                        stable_id=action.stable_id,
                        label=action.label,
                        place_query=_place_query_for_new(action.label, snapshot, user_message),
                        source="new",
                    )
                )
                continue
            dataset_id = action.persistent_dataset_id
            targets.append(
                IncrementalTarget(
                    stable_id=action.stable_id,
                    label=prior.label,
                    place_query=prior.place.query,
                    source=action.source,
                    place=prior.place,
                    persistent_dataset_id=dataset_id,
                )
            )
        return IncrementalComparisonPlan(
            parent_execution_id=snapshot.execution_id,
            follow_up=follow_up,
            reuse=assessment,
            targets=tuple(targets),
            pattern=pattern,
            pattern_reuse=pattern_reuse,
            user_facing_preamble=_preamble(follow_up, assessment, snapshot),
        )

    def _dataset_action(
        self,
        dataset: ExecutionDatasetSnapshot | None,
        *,
        force_refresh: bool,
        keep_stale: bool,
        inferred_goal: AnalysisGoal,
        now: datetime,
    ) -> tuple[DatasetSource, list[str]]:
        if dataset is None:
            return "new", ["execution_memory_dataset_missing"]
        reasons: list[str] = []
        if force_refresh:
            return "refreshed", ["refresh_required"]
        age = (now - dataset.retrieved_at).total_seconds()
        if age > self._osm_ttl_seconds and not keep_stale:
            return "refreshed", ["dataset_stale", f"age_seconds={int(age)}"]
        if dataset.truncated and inferred_goal == "abundance":
            return "refreshed", ["truncated_limit_hit", "dataset_comparability_invalid"]
        reasons.append("dataset_fresh")
        reasons.append("dataset_complete")
        return "reused", reasons

    def _decision(
        self,
        *,
        intents: Container[str],
        actions: list[TargetDatasetAction],
        metric_rev_status: str,
        radius_invalidates: bool,
        re_ground: bool,
        truncated_blocked: bool,
    ) -> ReuseDecision:
        if re_ground:
            return "FULL_REPLAN"
        if radius_invalidates:
            return "REFRESH_DATA"
        sources = {item.source for item in actions}
        if (
            sources == {"reused"}
            and "ADD_TARGET" not in intents
            and "REPLACE_TARGETS" not in intents
        ):
            if truncated_blocked:
                return "REFRESH_DATA"
            return "FULL_REUSE"
        if "new" in sources or "refreshed" in sources:
            if sources == {"refreshed"}:
                return "REFRESH_DATA"
            return "PARTIAL_REUSE"
        if metric_rev_status == "rejected":
            return "REVALIDATE_METRIC"
        return "PARTIAL_REUSE"


def _dataset_for_target(
    snapshot: ExecutionSnapshot, stable_id: str
) -> ExecutionDatasetSnapshot | None:
    for dataset in snapshot.datasets:
        if dataset.target_stable_id == stable_id:
            return dataset
    return None


def _same_label(left: str, right: str) -> bool:
    return left.strip().lower() == right.strip().lower() or (
        left.lower() in right.lower() or right.lower() in left.lower()
    )


def _place_query_for_new(label: str, snapshot: ExecutionSnapshot, user_message: str = "") -> str:
    if user_message.strip():
        return canonicalize_place_query(label, user_message=user_message)
    in_tehran = any("tehran" in (t.place.query or "").lower() for t in snapshot.targets)
    if in_tehran and "tehran" not in label.lower() and "iran" not in label.lower():
        return f"{label}, Tehran, Iran"
    return label


def _preamble(
    follow_up: FollowUpResolution,
    assessment: ReuseAssessment,
    snapshot: ExecutionSnapshot,
) -> str:
    added = follow_up.added_labels
    metric_rev = assessment.metric_revalidation
    if follow_up.follow_up_type == "REPLACE_TARGETS":
        names = " and ".join(item.label for item in assessment.target_actions)
        return (
            f"The comparison now uses {names}. "
            "The previous dataset definition, radius, and metric were reused "
            "where they remained valid; only targets missing from execution "
            "memory were queried."
        )
    if added and metric_rev.status == "accepted":
        names = " and ".join(added)
        return (
            f"{names} was added to the existing {assessment.radius_m // 1000} km "
            f"{assessment.feature_concept} comparison. The previous metric remained "
            "suitable, so earlier valid datasets were reused and only the new "
            "target was queried."
        )
    if metric_rev.status == "rejected":
        previous = metric_rev.previous_metric or "the previous indicator"
        replacement = metric_rev.replacement_metric or "a replacement metric"
        return (
            f"The previous comparison used {previous} for "
            f"{snapshot.inferred_goal or 'the prior goal'}. The new request asks "
            f"about {assessment.inferred_goal}, so the analysis was replanned "
            f"using {replacement}."
        )
    if follow_up.follow_up_type == "CHANGE_RADIUS":
        return (
            f"The comparison radius changed to {assessment.radius_m} m, so previous "
            "spatial datasets were not reused."
        )
    return ""
