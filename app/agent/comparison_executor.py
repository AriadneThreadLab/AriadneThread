"""Deterministic multi-target comparison execution after a compact plan.

LLM stages: comparison plan → metric selection → final report.
Tool Registry stages: grounding, resolve_place, query_osm, analyze_features.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, cast

from app.agent.accumulator import ResultAccumulator
from app.agent.comparison_workflow import (
    COMPARISON_PLANNER_SYSTEM,
    FINAL_REPORT_SYSTEM,
    METRIC_PLANNER_SYSTEM,
    ComparisonTargetDraft,
    MetricSelectionDraft,
    MultiTargetComparisonPlan,
    is_multi_target_landmark_comparison,
    normalize_plan_with_user_context,
    parse_comparison_plan_payload,
    seed_plan_from_user_message,
    tags_for_feature_concept,
)
from app.agent.contracts import GeoAgentResponse, ListTraceRecorder, StopReason, describe_payload
from app.agent.planning_diagnostics import diagnostics_log_fields, measure_planning_messages
from app.analytics.catalog import METRIC_CATALOG
from app.analytics.contracts import MetricType
from app.core.errors import LLMError, LLMTimeoutError
from app.execution_memory.contracts import (
    ExecutionMemoryTrace,
    ExecutionSnapshot,
    IncrementalComparisonPlan,
)
from app.execution_memory.restore import (
    restore_dataset,
    restore_documentation_passages,
    restore_place,
)
from app.llm.contracts import ChatMessage, LLMOptions, LLMProvider, ToolCall
from app.llm.tool_protocol import FINAL_ANSWER_KEY, normalize_prompted_text
from app.tools.analyze_features import TOOL_NAME as ANALYZE_FEATURES
from app.tools.context import ToolContext
from app.tools.query_osm import TOOL_NAME as QUERY_OSM
from app.tools.registry import ToolRegistry
from app.tools.resolve_place import TOOL_NAME as RESOLVE_PLACE
from app.tools.search_osm_knowledge import TOOL_NAME as SEARCH_OSM_KNOWLEDGE

logger = logging.getLogger(__name__)

_ALLOWED_METRICS = frozenset(
    {
        "count",
        "density",
        "total_area",
        "coverage_percentage",
        "median_area",
        "mean_area",
        "standard_deviation",
        "nearest_distance",
        "mean_nearest_distance",
        "median_nearest_distance",
    }
)
_DEFAULT_FEATURE_LIMIT = 50


class MultiTargetComparisonRunner:
    """Backend-scheduled comparison workflow using Tool Registry only."""

    def __init__(self, llm: LLMProvider, registry: ToolRegistry) -> None:
        self._llm = llm
        self._registry = registry

    async def run(
        self,
        *,
        user_message: str,
        state: ResultAccumulator,
        tool_context: ToolContext,
        recorder: ListTraceRecorder,
    ) -> GeoAgentResponse:
        plan = await self._plan(user_message, recorder)
        if plan is None:
            state.note_error(
                "llm_protocol_error",
                "comparison planner did not return a valid multi-target plan",
            )
            return self._response(
                state,
                recorder,
                answer=(
                    "The comparison could not be planned from the request. "
                    "No live OSM query was executed."
                ),
                stop_reason="llm_error",
            )

        recorder.record(
            "llm_turn",
            "comparison plan accepted",
            round_index=1,
            details={
                "status": "completed",
                "target_count": len(plan.targets),
                "radius_meters": plan.radius_m,
                "feature_concept": plan.feature_concept,
                # Structured plan echo (labels and place queries only). Downstream
                # active-learning selection reads this instead of the planner
                # prompt or the raw model reply.
                "comparison_plan": plan.model_dump(mode="json"),
            },
        )

        # STEP 1 — semantic grounding (Tool Registry)
        if not await self._ground(plan, state, tool_context, recorder):
            return self._response(
                state,
                recorder,
                answer=(
                    "OSM documentation grounding failed before live retrieval. "
                    "No comparison was completed."
                ),
                stop_reason="llm_error",
            )

        # STEP 2 — resolve ALL targets before any query_osm
        place_refs = await self._resolve_all(plan, state, tool_context, recorder)
        if place_refs is None:
            return self._response(
                state,
                recorder,
                answer=(
                    "One or more comparison locations could not be resolved reliably. "
                    "No live OSM comparison query was executed."
                ),
                stop_reason="llm_error",
            )

        # STEPS 3-5 — equal scopes + one query_osm per target + dataset refs
        dataset_refs = await self._query_all(
            plan,
            place_refs,
            state,
            tool_context,
            recorder,
        )
        if dataset_refs is None:
            return self._response(
                state,
                recorder,
                answer=(
                    "The live OpenStreetMap query timed out."
                    if state.live_error_code == "overpass_timeout"
                    else "Live OpenStreetMap retrieval failed for one or more comparison targets."
                ),
                stop_reason="llm_error",
            )

        # STEP 6-7 — metric selection (LLM) + analyze_features (deterministic)
        analysis_ok = await self._analyze(
            plan,
            dataset_refs,
            state,
            tool_context,
            recorder,
            place_refs=place_refs,
        )
        if not analysis_ok:
            return self._response(
                state,
                recorder,
                answer="Deterministic comparison analysis could not be completed.",
                stop_reason="llm_error",
            )

        # STEP 8 — final report (LLM; numbers from analysis only)
        answer = await self._final_report(plan, state, recorder)
        return self._response(state, recorder, answer=answer, stop_reason="final_answer")

    async def run_incremental(
        self,
        *,
        user_message: str,
        state: ResultAccumulator,
        tool_context: ToolContext,
        recorder: ListTraceRecorder,
        incremental: IncrementalComparisonPlan,
        snapshot: ExecutionSnapshot,
        memory_trace: ExecutionMemoryTrace,
    ) -> GeoAgentResponse:
        """Follow-up comparison using Execution Memory + Tool Registry."""
        assessment = incremental.reuse
        recorder.record(
            "llm_turn",
            "execution memory follow-up accepted",
            round_index=1,
            details={
                "status": "completed",
                "follow_up_type": incremental.follow_up.follow_up_type,
                "parent_execution_id": str(incremental.parent_execution_id),
                "reuse_decision": assessment.decision,
                "memory_reuse_reasons": list(assessment.reasons),
                "reused_targets": list(memory_trace.reused_targets),
                "refreshed_targets": list(memory_trace.refreshed_targets),
                "new_targets": list(memory_trace.new_targets),
                "previous_metric": memory_trace.previous_metric,
                "metric_revalidation_status": memory_trace.metric_revalidation_status,
                "final_metric": memory_trace.final_metric,
            },
        )
        _record_memory_steps(recorder, incremental, memory_trace, snapshot)
        draft = MultiTargetComparisonPlan(
            feature_concept=assessment.feature_concept[:120],
            targets=_draft_targets(incremental),
            radius_m=assessment.radius_m,
            comparison_goal=(assessment.comparison_goal or assessment.inferred_goal)[:240],
        )

        if assessment.re_ground:
            if not await self._ground(draft, state, tool_context, recorder):
                return self._response(
                    state,
                    recorder,
                    answer="OSM documentation grounding failed before the follow-up comparison.",
                    stop_reason="llm_error",
                    memory_trace=memory_trace,
                )
        else:
            tags = list(assessment.grounding_tags or snapshot.grounding_tags)
            if not tags:
                tags = tags_for_feature_concept(assessment.feature_concept) or []
            restored_docs = restore_documentation_passages(snapshot.documentation_sources)
            if restored_docs and tags:
                tool_context.grounding.tags = list(tags)
                state.validated_tags = list(tags)
                state.absorb_documentation_passages(restored_docs)
                memory_trace = memory_trace.model_copy(
                    update={
                        "documentation_sources_restored": True,
                        "documentation_source_titles": tuple(
                            item.title for item in snapshot.documentation_sources
                        ),
                    }
                )
            elif not await self._ground(draft, state, tool_context, recorder):
                return self._response(
                    state,
                    recorder,
                    answer="OSM documentation grounding failed before the follow-up comparison.",
                    stop_reason="llm_error",
                    memory_trace=memory_trace,
                )
            elif tags:
                # Live search recovered citations; keep the validated tag definition.
                tool_context.grounding.tags = list(tags)
                state.validated_tags = list(tags)

        place_refs: list[str] = []
        dataset_refs: list[str] = []
        for target in incremental.targets:
            if target.place is not None:
                record = restore_place(tool_context.places, target.place)
                place_refs.append(record.place_ref)
            else:
                before = set(tool_context.places.refs())
                call = ToolCall(
                    id=f"cmp-{uuid.uuid4().hex[:10]}",
                    name=RESOLVE_PLACE,
                    arguments={
                        "query": target.place_query,
                        "label": target.label,
                        "limit": 5,
                    },
                )
                ok = await self._invoke(call, state, tool_context, recorder, round_index=2)
                after = [ref for ref in tool_context.places.refs() if ref not in before]
                if not ok or not after:
                    return self._response(
                        state,
                        recorder,
                        answer=(
                            "The new comparison location could not be resolved reliably. "
                            "No incorrect reused result was returned."
                        ),
                        stop_reason="llm_error",
                        memory_trace=memory_trace,
                    )
                place_refs.append(after[-1])

            if target.source == "reused" and target.persistent_dataset_id is not None:
                stored = next(
                    (
                        item
                        for item in snapshot.datasets
                        if item.execution_dataset_id == target.persistent_dataset_id
                    ),
                    None,
                )
                if stored is None:
                    state.note_warning("execution_memory_dataset_missing")
                else:
                    dataset_refs.append(restore_dataset(state.datasets, stored))
                    state.absorb_osm_target_collection(
                        analysis_target=target.label,
                        geojson=stored.feature_collection,
                        feature_count=stored.feature_count,
                        effective_limit=stored.effective_limit,
                        scope_summary=stored.scope.summary if stored.scope else None,
                    )
                    continue
            # refresh or new → live query_osm for this place only
            created = await self._query_all(
                draft,
                [place_refs[-1]],
                state,
                tool_context,
                recorder,
            )
            if created is None:
                return self._response(
                    state,
                    recorder,
                    answer="Live OpenStreetMap retrieval failed for a follow-up target.",
                    stop_reason="llm_error",
                    memory_trace=memory_trace,
                )
            dataset_refs.append(created[0])

        if len(dataset_refs) != len(incremental.targets):
            return self._response(
                state,
                recorder,
                answer="Follow-up comparison could not bind datasets for every target.",
                stop_reason="llm_error",
                memory_trace=memory_trace,
            )

        analysis_ok = await self._analyze(
            draft,
            dataset_refs,
            state,
            tool_context,
            recorder,
            forced_metric=assessment.final_metric,
            inferred_goal=assessment.inferred_goal,
            place_refs=place_refs,
        )
        if not analysis_ok:
            return self._response(
                state,
                recorder,
                answer="Deterministic comparison analysis could not be completed.",
                stop_reason="llm_error",
                memory_trace=memory_trace,
            )
        recorder.record(
            "memory_reuse",
            f"New comparison generated for {len(incremental.targets)} target(s)",
            round_index=5,
            details={"status": "completed", "memory_step": "comparison_recomputed"},
        )
        answer = await self._final_report(
            draft,
            state,
            recorder,
            preamble=incremental.user_facing_preamble,
        )
        return self._response(
            state,
            recorder,
            answer=answer,
            stop_reason="final_answer",
            memory_trace=memory_trace,
        )

    async def _plan(
        self,
        user_message: str,
        recorder: ListTraceRecorder,
    ) -> MultiTargetComparisonPlan | None:
        seed = seed_plan_from_user_message(user_message)
        messages = [
            ChatMessage(role="system", content=COMPARISON_PLANNER_SYSTEM),
            ChatMessage(role="user", content=user_message),
        ]
        diag = measure_planning_messages(
            messages,
            tools=None,
            tool_catalog_text="",
            model=self._llm.model_name,
            temperature=0.1,
        )
        logger.info(
            "comparison_planning_diagnostics %s",
            " ".join(f"{k}={v}" for k, v in diagnostics_log_fields(diag).items()),
        )
        recorder.record(
            "llm_turn",
            "comparison planner (compact, no tools)",
            round_index=1,
            details={"status": "info", **diagnostics_log_fields(diag)},
        )
        started = time.perf_counter()
        try:
            response = await self._llm.chat(
                messages,
                tools=None,
                options=LLMOptions(json_mode=True, temperature=0.1, max_tokens=400),
            )
        except LLMTimeoutError as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            recorder.record(
                "stopped",
                f"LLM timed out during comparison planning ({duration_ms} ms)",
                round_index=1,
                error_code=exc.code,
                details={"status": "failed", "duration_ms": duration_ms},
            )
            raise
        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "comparison_planning_complete duration_ms=%s content_chars=%s",
            duration_ms,
            len(response.content or ""),
        )
        raw = normalize_prompted_text(response.content or "")
        try:
            plan = parse_comparison_plan_payload(raw)
            plan = normalize_plan_with_user_context(plan, user_message)
            return plan
        except (ValueError, json.JSONDecodeError, TypeError) as exc:
            logger.info("comparison_plan_parse_failed error=%s", type(exc).__name__)
            if seed is not None:
                recorder.record(
                    "llm_turn",
                    "comparison planner fell back to deterministic seed plan",
                    round_index=1,
                    details={"status": "warning"},
                )
                return seed
            return None

    async def _ground(
        self,
        plan: MultiTargetComparisonPlan,
        state: ResultAccumulator,
        tool_context: ToolContext,
        recorder: ListTraceRecorder,
    ) -> bool:
        if not self._registry.has(SEARCH_OSM_KNOWLEDGE):
            # Exact-tag style concepts can proceed without RAG.
            concept_tags = tags_for_feature_concept(plan.feature_concept)
            if concept_tags:
                tool_context.grounding.tags = list(concept_tags)
                state.validated_tags = list(concept_tags)
                return True
            return False
        concept_tags = tags_for_feature_concept(plan.feature_concept)
        if concept_tags:
            tool_context.grounding.tags = list(concept_tags)
            state.validated_tags = list(concept_tags)
        query = plan.feature_concept
        if "tag" not in query.lower() and ("park" in query.lower() or concept_tags):
            query = f"{query} OpenStreetMap tag"
        call = ToolCall(
            id=f"cmp-{uuid.uuid4().hex[:10]}",
            name=SEARCH_OSM_KNOWLEDGE,
            arguments={"query": query, "top_k": 5},
        )
        ok = await self._invoke(call, state, tool_context, recorder, round_index=2)
        if concept_tags:
            # Built-in concept tags win over a single RAG hint (e.g. leisure=park).
            tool_context.grounding.tags = list(concept_tags)
            state.validated_tags = list(concept_tags)
        return bool(tool_context.grounding.tags) or ok

    async def _resolve_all(
        self,
        plan: MultiTargetComparisonPlan,
        state: ResultAccumulator,
        tool_context: ToolContext,
        recorder: ListTraceRecorder,
    ) -> list[str] | None:
        if not self._registry.has(RESOLVE_PLACE):
            state.note_error("place_resolution_error", "resolve_place is not registered")
            return None
        refs: list[str] = []
        for target in plan.targets:
            before = set(tool_context.places.refs())
            call = ToolCall(
                id=f"cmp-{uuid.uuid4().hex[:10]}",
                name=RESOLVE_PLACE,
                arguments={
                    "query": target.place_query,
                    "label": target.label,
                    "limit": 5,
                },
            )
            ok = await self._invoke(call, state, tool_context, recorder, round_index=2)
            if not ok:
                return None
            after = [ref for ref in tool_context.places.refs() if ref not in before]
            if not after:
                state.note_error(
                    "place_resolution_error",
                    f"no place_ref registered for {target.label}",
                )
                return None
            refs.append(after[-1])
        if len(refs) != len(plan.targets):
            return None
        return refs

    async def _query_all(
        self,
        plan: MultiTargetComparisonPlan,
        place_refs: list[str],
        state: ResultAccumulator,
        tool_context: ToolContext,
        recorder: ListTraceRecorder,
    ) -> list[str] | None:
        if not self._registry.has(QUERY_OSM):
            state.note_error("tool_execution_error", "query_osm is not registered")
            return None
        tags = tool_context.grounding.tags or state.validated_tags or ["leisure=park"]
        tag_args = []
        for tag in tags:
            if "=" in tag:
                key, value = tag.split("=", 1)
                tag_args.append({"key": key, "value": value})
            else:
                tag_args.append({"key": tag})
        refs_before = set(state.datasets.refs())
        for place_ref in place_refs:
            if not tool_context.places.has(place_ref):
                state.note_error(
                    "tool_argument_error",
                    f"query_osm blocked: missing place_ref {place_ref}",
                )
                return None
            arguments: dict[str, Any] = {
                "place_ref_scope": {
                    "place_ref": place_ref,
                    "radius_m": plan.radius_m,
                },
                "tags": tag_args,
                "limit": _DEFAULT_FEATURE_LIMIT,
                "include_geometry": True,
            }
            if len(tag_args) > 1:
                arguments["tag_match"] = "any"
            call = ToolCall(
                id=f"cmp-{uuid.uuid4().hex[:10]}",
                name=QUERY_OSM,
                arguments=arguments,
            )
            ok = await self._invoke(call, state, tool_context, recorder, round_index=3)
            if not ok:
                return None
        created = [ref for ref in state.datasets.refs() if ref not in refs_before]
        if len(created) != len(place_refs):
            return None
        return created

    async def _analyze(
        self,
        plan: MultiTargetComparisonPlan,
        dataset_refs: list[str],
        state: ResultAccumulator,
        tool_context: ToolContext,
        recorder: ListTraceRecorder,
        *,
        forced_metric: str | None = None,
        inferred_goal: str | None = None,
        place_refs: list[str] | None = None,
    ) -> bool:
        if forced_metric and forced_metric in _ALLOWED_METRICS:
            primary = forced_metric
            inferred = inferred_goal or _map_inferred_goal("", primary)
            recorder.record(
                "llm_turn",
                f"metric revalidated: {primary}",
                round_index=4,
                details={
                    "status": "completed",
                    "primary_metric": primary,
                    "metric_revalidation": "accepted" if inferred_goal else "forced",
                },
            )
        else:
            metric_draft = await self._select_metric(plan, state, dataset_refs, recorder)
            primary = metric_draft.primary_metric
            if primary not in _ALLOWED_METRICS:
                primary = "count"
            inferred = inferred_goal or _map_inferred_goal(metric_draft.inferred_goal, primary)
        metrics: list[dict[str, Any]] = [
            {
                "metric": primary,
                "role": "primary",
                "inferred_goal": inferred,
            }
        ]
        targets = []
        for index, (target, dataset_ref) in enumerate(
            zip(plan.targets, dataset_refs, strict=True),
            start=1,
        ):
            item: dict[str, Any] = {
                "target_id": f"t{index}",
                "label": target.label[:80],
                "dataset_ref": dataset_ref,
            }
            catalog = (
                METRIC_CATALOG[cast(MetricType, primary)] if primary in METRIC_CATALOG else None
            )
            if (
                catalog is not None
                and catalog.requires_reference_points
                and place_refs is not None
                and index <= len(place_refs)
            ):
                place = tool_context.places.get(place_refs[index - 1])
                if place is not None:
                    item["reference_points"] = [{"lat": place.latitude, "lon": place.longitude}]
            targets.append(item)
        call = ToolCall(
            id=f"cmp-{uuid.uuid4().hex[:10]}",
            name=ANALYZE_FEATURES,
            arguments={
                "analysis_type": "comparison",
                "feature_concept": plan.feature_concept[:80],
                "comparison_goal": plan.comparison_goal[:200],
                "targets": targets,
                "metrics": metrics,
            },
        )
        return await self._invoke(call, state, tool_context, recorder, round_index=4)

    async def _select_metric(
        self,
        plan: MultiTargetComparisonPlan,
        state: ResultAccumulator,
        dataset_refs: list[str],
        recorder: ListTraceRecorder,
    ) -> MetricSelectionDraft:
        dataset_lines = []
        for ref in dataset_refs:
            record = state.datasets.get(ref)
            dataset_lines.append(
                f"{ref}: features={record.feature_count} truncated={record.truncated} "
                f"limit={record.effective_limit} scope={record.scope.scope_kind}"
            )
        user = (
            f"comparison_goal={plan.comparison_goal}\n"
            f"feature_concept={plan.feature_concept}\n"
            f"radius_m={plan.radius_m}\n"
            "datasets:\n" + "\n".join(dataset_lines)
        )
        messages = [
            ChatMessage(role="system", content=METRIC_PLANNER_SYSTEM),
            ChatMessage(role="user", content=user),
        ]
        diag = measure_planning_messages(
            messages,
            tools=None,
            tool_catalog_text="",
            model=self._llm.model_name,
        )
        logger.info(
            "metric_planning_diagnostics %s",
            " ".join(f"{k}={v}" for k, v in diagnostics_log_fields(diag).items()),
        )
        try:
            response = await self._llm.chat(
                messages,
                tools=None,
                options=LLMOptions(json_mode=True, temperature=0.1, max_tokens=250),
            )
            raw = normalize_prompted_text(response.content or "")
            payload = json.loads(raw)
            draft = MetricSelectionDraft.model_validate(payload)
            recorder.record(
                "llm_turn",
                f"metric selected: {draft.primary_metric}",
                round_index=4,
                details={
                    "status": "completed",
                    "primary_metric": draft.primary_metric,
                },
            )
            return draft
        except (LLMError, ValueError, TypeError, json.JSONDecodeError):
            recorder.record(
                "llm_turn",
                "metric planner fallback to count",
                round_index=4,
                details={"status": "warning", "primary_metric": "count"},
            )
            return MetricSelectionDraft(
                primary_metric="count",
                inferred_goal="abundance",
                rationale="Fallback to count after metric planner failure.",
            )

    async def _final_report(
        self,
        plan: MultiTargetComparisonPlan,
        state: ResultAccumulator,
        recorder: ListTraceRecorder,
        *,
        preamble: str = "",
    ) -> str:
        analysis = state.analysis
        if analysis is None or analysis.result is None:
            base = "Comparison analysis completed without a reportable result."
            return f"{preamble} {base}".strip() if preamble else base
        payload = {
            "comparison_goal": plan.comparison_goal,
            "feature_concept": plan.feature_concept,
            "status": analysis.status,
            "targets": [
                {
                    "label": target.label,
                    "dataset_ref": target.dataset_ref,
                    "retrieved_feature_count": target.data_provenance.retrieved_feature_count,
                    "truncated": target.data_provenance.truncated,
                    "limit_reached": target.data_provenance.limit_reached,
                    "effective_limit": target.data_provenance.effective_limit,
                    "metrics": [
                        {
                            "metric": metric.metric,
                            "value": metric.value,
                            "unit": metric.unit,
                            "status": metric.status,
                            "notes": metric.notes,
                        }
                        for metric in target.metrics
                    ],
                }
                for target in analysis.result.targets
            ],
            "comparison_statement": (
                analysis.comparison.overall_statement if analysis.comparison else None
            ),
            "limitations": list(analysis.result.limitations),
            "validated_tags": list(state.validated_tags),
        }
        messages = [
            ChatMessage(role="system", content=FINAL_REPORT_SYSTEM),
            ChatMessage(
                role="user",
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
        ]
        try:
            response = await self._llm.chat(
                messages,
                tools=None,
                options=LLMOptions(json_mode=True, temperature=0.2, max_tokens=500),
            )
            raw = normalize_prompted_text(response.content or "")
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                answer = parsed.get(FINAL_ANSWER_KEY) or parsed.get("answer")
                if isinstance(answer, str) and answer.strip():
                    recorder.record(
                        "final_answer",
                        f"final answer ({len(answer)} characters)",
                        round_index=5,
                        details={"status": "completed"},
                    )
                    return _with_preamble(answer.strip(), preamble)
        except (LLMError, ValueError, TypeError, json.JSONDecodeError):
            pass
        # Deterministic fallback report (still uses computed values only).
        lines = [
            f"Comparison of {plan.feature_concept} within {plan.radius_m} m "
            f"({plan.comparison_goal}).",
        ]
        if analysis.comparison is not None:
            lines.append(analysis.comparison.overall_statement)
        for target in analysis.result.targets:
            primary = next((m for m in target.metrics if m.role == "primary"), None)
            if primary is None:
                continue
            trunc = " (truncated at limit)" if target.data_provenance.limit_reached else ""
            lines.append(f"{target.label}: {primary.metric}={primary.value}{trunc}")
        if analysis.result.limitations:
            lines.append("Limitations: " + ", ".join(analysis.result.limitations))
        answer = _with_preamble(" ".join(lines), preamble)
        recorder.record(
            "final_answer",
            f"final answer ({len(answer)} characters)",
            round_index=5,
            details={"status": "completed"},
        )
        return answer

    async def _invoke(
        self,
        call: ToolCall,
        state: ResultAccumulator,
        tool_context: ToolContext,
        recorder: ListTraceRecorder,
        *,
        round_index: int,
    ) -> bool:
        recorder.record(
            "tool_call",
            f"scheduled {call.name}",
            round_index=round_index,
            tool_name=call.name,
            details={"status": "info", "tool_name": call.name},
        )
        started = time.perf_counter()
        invocation = await self._registry.invoke(call, tool_context)
        duration_ms = int((time.perf_counter() - started) * 1000)
        if invocation.ok:
            recorder.record(
                "tool_result",
                describe_payload(invocation.payload),
                round_index=round_index,
                tool_name=call.name,
                details={
                    "status": "completed",
                    "duration_ms": duration_ms,
                    "tool_name": call.name,
                },
            )
            state.absorb(call.name, invocation.payload)
            if call.name == SEARCH_OSM_KNOWLEDGE and not tool_context.grounding.tags:
                hints = state.documented_tag_hints()
                preferred = [hint for hint in hints if hint.lower() == "leisure=park"]
                selected = preferred or hints[:1]
                if selected:
                    tool_context.grounding.tags = list(selected)
                    state.absorb_grounding_tags(list(selected))
            if call.name == QUERY_OSM:
                tags = call.arguments.get("tags")
                if isinstance(tags, list):
                    summarised = []
                    for item in tags:
                        if isinstance(item, dict) and isinstance(item.get("key"), str):
                            key = item["key"]
                            value = item.get("value")
                            summarised.append(key if value is None else f"{key}={value}")
                    if summarised:
                        state.validated_tags = summarised
                        if tool_context.grounding.tags is None:
                            tool_context.grounding.tags = list(summarised)
            return True

        code = invocation.error_code or "tool_execution_error"
        recorder.record(
            "tool_error",
            invocation.observation,
            round_index=round_index,
            tool_name=call.name,
            error_code=code,
            details={"status": "failed", "duration_ms": duration_ms},
        )
        public = invocation.public_error or invocation.observation
        if code.startswith("place_"):
            state.note_error(code, public)
        elif call.name == QUERY_OSM and invocation.failure_meta:
            state.note_error(code, public)
            state.note_live_query_failure(invocation.failure_meta)
        else:
            state.note_error(code, public)
        return False

    def _response(
        self,
        state: ResultAccumulator,
        recorder: ListTraceRecorder,
        *,
        answer: str,
        stop_reason: StopReason,
        memory_trace: ExecutionMemoryTrace | None = None,
        conversation_id: str | None = None,
    ) -> GeoAgentResponse:
        return GeoAgentResponse(
            answer=answer,
            sources=list(state.sources),
            trace=recorder.as_list(),
            geojson=state.geojson,
            feature_count=state.feature_count,
            passages=list(state.passages),
            overpass_query=state.overpass_query,
            warnings=list(state.warnings),
            errors=list(state.errors),
            stop_reason=stop_reason,
            model=self._llm.model_name,
            effective_limit=state.effective_limit,
            scope_summary=state.scope_summary,
            validated_tags=list(state.validated_tags),
            live_query_failed=state.live_query_failed,
            live_error_code=state.live_error_code,
            analysis=state.analysis,
            conversation_id=conversation_id,
            execution_memory=memory_trace,
        )


def should_use_multi_target_runner(message: str) -> bool:
    return is_multi_target_landmark_comparison(message)


def _map_inferred_goal(raw: str, metric: str) -> str:
    allowed = {
        "abundance",
        "concentration",
        "accessibility",
        "total_provision",
        "typical_value",
        "variability",
        "coverage",
        "relative_share",
    }
    text = (raw or "").strip().lower().replace(" ", "_")
    if text in allowed:
        return text
    defaults = {
        "count": "abundance",
        "density": "concentration",
        "total_area": "total_provision",
        "coverage_percentage": "coverage",
        "median_area": "typical_value",
        "mean_area": "typical_value",
        "standard_deviation": "variability",
    }
    return defaults.get(metric, "abundance")


def _draft_targets(incremental: IncrementalComparisonPlan) -> list[ComparisonTargetDraft]:
    return [
        ComparisonTargetDraft(label=target.label[:120], place_query=target.place_query[:200])
        for target in incremental.targets
    ]


def _record_memory_steps(
    recorder: ListTraceRecorder,
    incremental: IncrementalComparisonPlan,
    memory_trace: ExecutionMemoryTrace,
    snapshot: ExecutionSnapshot,
) -> None:
    """Make reuse auditable in the workflow: what was reused, what was redone."""
    reuse = incremental.reuse
    pattern = incremental.pattern

    def step(message: str, name: str, status: str = "completed") -> None:
        recorder.record(
            "memory_reuse",
            message,
            round_index=1,
            details={"status": status, "memory_step": name},
        )

    if pattern is not None:
        step(f"Previous analysis pattern found: {pattern.analysis_pattern}", "pattern_found")
    else:
        step("No reusable analysis pattern; methodology planned fresh", "pattern_missing", "info")

    revalidation = reuse.metric_revalidation
    if revalidation.status == "rejected":
        step(
            f"Metric replanned: {reuse.final_metric} replaces {revalidation.previous_metric}",
            "metric_replanned",
            "warning",
        )
    else:
        step(f"Metric validated: {reuse.final_metric}", "metric_validated")

    if snapshot.radius_m == reuse.radius_m:
        step(f"Radius reused: {reuse.radius_m} m", "radius_reused")
    else:
        step(f"Radius changed: {snapshot.radius_m} m to {reuse.radius_m} m", "radius_changed")

    tags = ",".join(reuse.grounding_tags or snapshot.grounding_tags)
    if tags and not reuse.re_ground:
        step(f"Dataset definition reused: {tags}", "dataset_definition_reused")
    if snapshot.documentation_sources and not reuse.re_ground:
        titles = ", ".join(item.title for item in snapshot.documentation_sources)
        step(
            f"OSM documentation sources restored: {titles}",
            "documentation_restored",
        )
    else:
        step(
            "OSM documentation evidence missing or concept changed; searching OSM knowledge",
            "documentation_researched",
            "info",
        )

    previous = [target.label for target in snapshot.targets]
    requested = [target.label for target in incremental.targets]
    dropped = [label for label in previous if label not in requested]
    if dropped or memory_trace.new_targets:
        step(
            "Targets changed: "
            + ", ".join(
                part
                for part in (
                    f"dropped {', '.join(dropped)}" if dropped else "",
                    (
                        f"added {', '.join(memory_trace.new_targets)}"
                        if memory_trace.new_targets
                        else ""
                    ),
                )
                if part
            ),
            "targets_changed",
        )
    else:
        step(f"Targets unchanged: {', '.join(requested)}", "targets_unchanged")

    recomputed = [*memory_trace.new_targets, *memory_trace.refreshed_targets]
    if recomputed:
        step(f"Spatial query recomputed for: {', '.join(recomputed)}", "osm_query_recomputed")
    if memory_trace.reused_targets:
        step(
            f"Stored dataset reused for: {', '.join(memory_trace.reused_targets)}",
            "dataset_reused",
        )


def _with_preamble(answer: str, preamble: str) -> str:
    text = answer.strip()
    prefix = (preamble or "").strip()
    if not prefix:
        return text
    if prefix in text:
        return text
    return f"{prefix} {text}".strip()
