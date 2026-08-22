"""Production agent query endpoint."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from app.active_learning.contracts import FeedbackSubmission
from app.agent.contracts import GeoAgentRequest, GeoAgentResponse
from app.agent.prompts import PROMPT_VERSION
from app.api.dependencies import ActiveLearningDep, GeoAgentDep, SettingsDep
from app.api.errors import error_body, status_for_agent_result_errors
from app.api.schemas import (
    AgentFeedbackRequest,
    AgentFeedbackResponse,
    AgentQueryRequest,
    AgentQueryResponse,
    to_agent_query_response,
)
from app.core.errors import AgentRequestTimeoutError, GeoAgentError
from app.llm.tool_schema import TOOL_SCHEMA_VERSION

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["agent"])

_THINK_RE = re.compile(r"</?think\b", re.IGNORECASE)


@router.post(
    "/agent/query",
    response_model=AgentQueryResponse,
    summary="Run Ariadne Thread on a natural-language GIS request",
    responses={
        200: {"description": "Agent completed (including valid empty OSM results)."},
        422: {"description": "Request validation failed."},
        429: {"description": "Upstream rate limited the request."},
        503: {"description": "A required dependency is unavailable."},
        504: {"description": "Bounded request or upstream timeout."},
        500: {"description": "Unexpected internal failure."},
    },
)
async def agent_query(
    body: AgentQueryRequest,
    request: Request,
    response: Response,
    agent: GeoAgentDep,
    settings: SettingsDep,
    active_learning: ActiveLearningDep,
) -> AgentQueryResponse | JSONResponse:
    """Execute the bounded planner-executor loop and return a typed result."""
    request_id = getattr(request.state, "request_id", None)
    conversation_id = (body.conversation_id or "").strip() or str(uuid.uuid4())
    domain_request = GeoAgentRequest(
        message=body.message,
        conversation_id=conversation_id,
        request_id=request_id,
    )

    try:
        result = await asyncio.wait_for(
            agent.run(domain_request),
            timeout=settings.agent_request_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise AgentRequestTimeoutError(
            f"Agent request exceeded {settings.agent_request_timeout_seconds:g}s"
        ) from exc
    except GeoAgentError:
        raise
    except Exception:
        logger.exception("unexpected agent failure request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content=error_body(
                code="internal_error",
                message="An unexpected internal error occurred.",
                request_id=request_id,
            ),
        )

    payload = to_agent_query_response(result, request_id=request_id)
    if _contains_hidden_reasoning(payload):
        logger.error("hidden reasoning leaked into agent response; refusing to return it")
        return JSONResponse(
            status_code=500,
            content=error_body(
                code="internal_error",
                message="An unexpected internal error occurred.",
                request_id=request_id,
            ),
        )

    await _offer_to_active_learning(
        active_learning,
        request_id=request_id,
        user_query=body.message,
        result=result,
    )

    tool_calls = sum(1 for step in payload.execution_trace if step.event == "tool_call")
    logger.info(
        "agent_query request_id=%s stop_reason=%s tool_calls=%s feature_count=%s "
        "warnings=%s errors=%s",
        request_id,
        payload.stop_reason,
        tool_calls,
        payload.feature_count,
        len(payload.warnings),
        [code.split(":", 1)[0] for code in payload.errors],
    )

    if result.stop_reason == "llm_error":
        status = status_for_agent_result_errors(result.errors) or 503
        return JSONResponse(
            status_code=status,
            content=payload.model_dump(mode="json"),
        )

    if not result.answer.strip() and result.errors:
        mapped_status = status_for_agent_result_errors(result.errors)
        if mapped_status is not None:
            return JSONResponse(
                status_code=mapped_status,
                content=payload.model_dump(mode="json"),
            )

    response.status_code = 200
    return payload


@router.post(
    "/agent/feedback",
    response_model=AgentFeedbackResponse,
    summary="Submit feedback about a completed run for human review",
    responses={
        200: {"description": "Feedback accepted (a candidate may or may not exist)."},
        422: {"description": "Request validation failed."},
    },
)
async def agent_feedback(
    body: AgentFeedbackRequest,
    active_learning: ActiveLearningDep,
) -> AgentFeedbackResponse:
    """Record end-user feedback.

    Feedback only ever *queues* a run for human review. It never approves a
    training example and never triggers training.
    """
    if active_learning is None:
        return AgentFeedbackResponse(accepted=False, detail="active learning is disabled")

    submission = FeedbackSubmission(
        request_id=body.request_id,
        sentiment=body.sentiment,
        failure_category=body.failure_category,
        corrected_output=body.corrected_output,
        note=body.note,
    )
    try:
        candidate = await active_learning.record_feedback(submission)
    except Exception:
        logger.warning("active_learning_feedback_failed request_id=%s", body.request_id)
        return AgentFeedbackResponse(accepted=False, detail="feedback could not be recorded")

    if candidate is None:
        return AgentFeedbackResponse(
            accepted=True,
            detail="feedback recorded; no reviewable candidate exists for this request",
        )
    return AgentFeedbackResponse(
        accepted=True,
        candidate_id=candidate.candidate_id,
        review_status=candidate.review_status.value,
        detail="feedback recorded and queued for human review",
    )


async def _offer_to_active_learning(
    service: ActiveLearningDep,
    *,
    request_id: str | None,
    user_query: str,
    result: GeoAgentResponse,
) -> None:
    """Offer a finished run for candidate selection.

    Best-effort by construction: the service swallows its own failures, and a
    missing correlation id simply means the run is not retained.
    """
    if service is None or not request_id:
        return
    await service.observe_run(
        request_id=request_id,
        user_query=user_query,
        response=result,
        prompt_version=PROMPT_VERSION,
        tool_schema_version=TOOL_SCHEMA_VERSION,
    )


def _contains_hidden_reasoning(payload: AgentQueryResponse) -> bool:
    """Defence-in-depth check for leaked chain-of-thought markers."""
    return _THINK_RE.search(payload.model_dump_json()) is not None
