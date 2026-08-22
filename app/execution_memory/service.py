"""Execution Memory service: load, resolve, restore, persist."""

from __future__ import annotations

import logging
from uuid import UUID

from app.agent.contracts import GeoAgentRequest, GeoAgentResponse
from app.agent.prompts import PROMPT_VERSION
from app.analytics.contracts import AnalysisGoal
from app.core.config import Settings
from app.execution_memory.analysis_memory import (
    classify_target_category,
    derive_analysis_pattern,
    select_pattern,
    validate_pattern_reuse,
)
from app.execution_memory.contracts import (
    AnalysisPattern,
    ExecutionMemoryTrace,
    ExecutionSnapshot,
    IncrementalComparisonPlan,
)
from app.execution_memory.follow_up import FollowUpResolver
from app.execution_memory.repository import ExecutionMemoryRepository
from app.execution_memory.reuse import ExecutionReuseValidator
from app.execution_memory.snapshot import assert_no_hidden_reasoning, build_snapshot
from app.llm.contracts import LLMProvider
from app.llm.tool_schema import TOOL_SCHEMA_VERSION
from app.tools.context import ToolContext

logger = logging.getLogger(__name__)


class ExecutionMemoryService:
    """Optional, non-catastrophic analytical memory."""

    def __init__(
        self,
        repository: ExecutionMemoryRepository,
        settings: Settings,
        llm: LLMProvider | None = None,
    ) -> None:
        self._repository = repository
        self._settings = settings
        self._resolver = FollowUpResolver(llm)
        self._validator = ExecutionReuseValidator(
            osm_ttl_seconds=settings.execution_memory_osm_ttl_seconds
        )

    @property
    def enabled(self) -> bool:
        return self._settings.execution_memory_enabled

    async def latest(self, conversation_id: str) -> ExecutionSnapshot | None:
        try:
            return await self._repository.latest_for_conversation(conversation_id)
        except Exception:
            logger.warning("execution_memory_unavailable: failed to load conversation memory")
            return None

    async def history(self, conversation_id: str) -> list[ExecutionSnapshot]:
        """Immutable audit records, newest first. Never replayed as an answer."""
        try:
            return await self._repository.history_for_conversation(
                conversation_id,
                limit=self._settings.execution_memory_max_per_conversation,
            )
        except Exception:
            logger.warning("execution_memory_unavailable: failed to load conversation history")
            return []

    def patterns_from(self, history: list[ExecutionSnapshot]) -> list[AnalysisPattern]:
        """Project audit history into target-free reusable methodology."""
        patterns: list[AnalysisPattern] = []
        for snapshot in history:
            try:
                pattern = derive_analysis_pattern(snapshot)
            except ValueError:
                logger.warning("execution_memory: discarded pattern that leaked target identity")
                continue
            if pattern is not None:
                patterns.append(pattern)
        return patterns

    async def prepare_incremental(
        self,
        conversation_id: str,
        user_message: str,
    ) -> IncrementalComparisonPlan | None:
        """Plan a follow-up from reusable methodology, or None for a fresh run.

        Step 1 resolves the targets the *current* message asks for. Step 2
        retrieves a compatible pattern. Step 3 validates reuse. Step 4 builds a
        plan whose target set comes only from step 1.
        """
        if not self.enabled or not conversation_id:
            return None
        history = await self.history(conversation_id)
        if not history:
            return None
        snapshot = history[0]

        follow_up = await self._resolver.resolve(user_message, snapshot)
        if follow_up.follow_up_type in {"NEW_ANALYSIS", "AMBIGUOUS_FOLLOW_UP"}:
            return None
        requested = follow_up.requested_labels or tuple(target.label for target in snapshot.targets)

        feature_concept = follow_up.new_feature_concept or snapshot.feature_concept
        inferred_goal = follow_up.new_goal or snapshot.inferred_goal or "abundance"
        pattern = select_pattern(
            self.patterns_from(history),
            feature_concept=feature_concept,
            target_category=classify_target_category(requested),
            analysis_type=snapshot.analysis_type,
        )
        pattern_reuse = validate_pattern_reuse(
            pattern,
            requested_labels=requested,
            feature_concept=feature_concept,
            inferred_goal=inferred_goal,
            radius_m=follow_up.new_radius_m or snapshot.radius_m,
        )
        if pattern_reuse.blocks_reuse:
            # Methodology cannot be transferred: plan the analysis from scratch.
            return None

        assessment = self._validator.assess(snapshot, follow_up)
        if assessment.decision == "CANNOT_REUSE":
            return None
        return self._validator.build_incremental_plan(
            snapshot,
            follow_up,
            assessment,
            user_message,
            pattern=pattern,
            pattern_reuse=pattern_reuse,
        )

    def trace_for(
        self,
        *,
        conversation_id: str,
        plan: IncrementalComparisonPlan | None,
        snapshot: ExecutionSnapshot | None = None,
    ) -> ExecutionMemoryTrace:
        if plan is None:
            return ExecutionMemoryTrace(
                conversation_id=conversation_id,
                memory_reuse_attempted=False,
                parent_execution_id=str(snapshot.execution_id) if snapshot else None,
            )
        reuse = plan.reuse
        return ExecutionMemoryTrace(
            conversation_id=conversation_id,
            memory_execution_id=str(plan.parent_execution_id),
            parent_execution_id=str(plan.parent_execution_id),
            memory_reuse_attempted=True,
            memory_reuse_decision=reuse.decision,
            memory_reuse_reasons=reuse.reasons,
            follow_up_type=plan.follow_up.follow_up_type,
            reused_targets=tuple(
                item.label for item in reuse.target_actions if item.source == "reused"
            ),
            refreshed_targets=tuple(
                item.label for item in reuse.target_actions if item.source == "refreshed"
            ),
            new_targets=tuple(item.label for item in reuse.target_actions if item.source == "new"),
            previous_metric=reuse.metric_revalidation.previous_metric,
            metric_revalidation_status=reuse.metric_revalidation.status,
            metric_revalidation_reason=reuse.metric_revalidation.reason,
            final_metric=reuse.final_metric,
            previous_scope=_scope_label(snapshot) if snapshot else None,
            final_scope=f"{reuse.radius_m}m {reuse.feature_concept}",
            dataset_freshness_checks=tuple(
                f"{item.label}:{item.source}:{','.join(item.reasons)}"
                for item in reuse.target_actions
            ),
            analysis_pattern=plan.pattern.analysis_pattern if plan.pattern else None,
            pattern_key=plan.pattern.pattern_key if plan.pattern else None,
            pattern_reusable=bool(plan.pattern_reuse and plan.pattern_reuse.reusable),
            pattern_checks=(
                tuple(
                    f"{item.name}:{'pass' if item.passed else 'fail'}:{item.detail}"
                    for item in plan.pattern_reuse.checks
                )
                if plan.pattern_reuse
                else ()
            ),
            reused_components=(
                tuple(plan.pattern_reuse.reused_components) if plan.pattern_reuse else ()
            ),
            recomputed_components=(
                tuple(plan.pattern_reuse.recomputed_components) if plan.pattern_reuse else ()
            ),
            requested_targets=tuple(item.label for item in plan.targets),
            documentation_sources_restored=bool(
                snapshot and snapshot.documentation_sources and not reuse.re_ground
            ),
            documentation_source_titles=(
                tuple(item.title for item in snapshot.documentation_sources)
                if snapshot and not reuse.re_ground
                else ()
            ),
        )

    async def persist(
        self,
        *,
        request: GeoAgentRequest,
        response: GeoAgentResponse,
        tool_context: ToolContext,
        conversation_id: str,
        parent_execution_id: UUID | None,
        provider: str | None,
        radius_m: int | None = None,
        feature_concept: str | None = None,
        analysis_goal: str | None = None,
        inferred_goal: AnalysisGoal | None = None,
    ) -> ExecutionSnapshot | None:
        if not self.enabled:
            return None
        try:
            snapshot = build_snapshot(
                request=request,
                response=response,
                tool_context=tool_context,
                conversation_id=conversation_id,
                parent_execution_id=parent_execution_id,
                provider=provider,
                prompt_version=PROMPT_VERSION,
                tool_schema_version=TOOL_SCHEMA_VERSION,
                radius_m=radius_m,
                feature_concept=feature_concept,
                analysis_goal=analysis_goal,
                inferred_goal=inferred_goal,
            )
            if snapshot is None:
                return None
            assert_no_hidden_reasoning(snapshot)
            saved = await self._repository.save(snapshot)
            await self._repository.enforce_retention(
                conversation_id,
                max_per_conversation=self._settings.execution_memory_max_per_conversation,
                max_age_seconds=self._settings.execution_memory_max_age_seconds,
            )
            return saved
        except Exception:
            logger.warning(
                "execution_memory_unavailable: failed to persist execution snapshot",
                exc_info=True,
            )
            return None


def _scope_label(snapshot: ExecutionSnapshot | None) -> str | None:
    if snapshot is None:
        return None
    radius = f"{snapshot.radius_m}m" if snapshot.radius_m else snapshot.scope_kind
    return f"{radius} {snapshot.feature_concept}".strip()
