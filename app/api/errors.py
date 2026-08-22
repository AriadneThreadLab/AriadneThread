"""Stable public HTTP error mapping.

Never include stack traces, credentials, filesystem paths or raw provider
bodies in the JSON returned to clients.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import (
    AgentRequestTimeoutError,
    AvalAIAuthenticationError,
    AvalAIBillingError,
    AvalAIRateLimitError,
    DependencyUnavailableError,
    EmbeddingError,
    GeoAgentError,
    LLMError,
    LLMProtocolError,
    LLMTimeoutError,
    OverpassError,
    OverpassRateLimitedError,
    OverpassTimeoutError,
    ToolTimeoutError,
)

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str | None:
    value = getattr(request.state, "request_id", None)
    return str(value) if value else None


def error_body(
    *,
    code: str,
    message: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "detail": message}
    if request_id:
        body["request_id"] = request_id
    return body


def status_for_geoagent_error(exc: GeoAgentError) -> int:
    """Map a domain error to an HTTP status code."""
    if isinstance(
        exc,
        AgentRequestTimeoutError | LLMTimeoutError | ToolTimeoutError | OverpassTimeoutError,
    ):
        return 504
    if isinstance(exc, AvalAIRateLimitError | OverpassRateLimitedError):
        return 429
    if isinstance(exc, AvalAIAuthenticationError):
        return 401
    if isinstance(exc, AvalAIBillingError):
        return 402
    if isinstance(exc, OverpassError):
        return 503
    if isinstance(exc, LLMProtocolError | LLMError | EmbeddingError | DependencyUnavailableError):
        return 503
    return 400


def status_for_agent_result_errors(errors: list[str]) -> int | None:
    """Optional HTTP status when the orchestrator finished with a hard failure."""
    joined = " | ".join(errors).lower()
    if not joined:
        return None
    if "llm_timeout" in joined or "agent_request_timeout" in joined:
        return 504
    if "overpass_timeout" in joined:
        return 504
    if (
        "overpass_rate_limited" in joined
        or "avalai_rate_limited" in joined
        or "429" in joined
        or "rate limit" in joined
    ):
        return 429
    if "tool_argument_error" in joined and "llm_protocol_error" not in joined:
        # Final stop after repeated invalid arguments — client/model mistake, not
        # an upstream dependency outage.
        return 422
    if (
        "llm_protocol_error" in joined
        or "llm_tool_protocol_error" in joined
        or "llm_error" in joined
        or "ollama_bad_request" in joined
        or "ollama_unreachable" in joined
        or "ollama_model_not_found" in joined
        or "ollama_protocol_error" in joined
        or "ollama_invalid_response" in joined
        or "avalai_authentication_error" in joined
        or "avalai_billing_error" in joined
        or "avalai_unavailable" in joined
        or "avalai_invalid_model" in joined
        or "avalai_invalid_response" in joined
        or "avalai_error" in joined
    ):
        return 503
    if "embedding_error" in joined or "knowledge search unavailable" in joined:
        return 503
    if "overpass_upstream_error" in joined or "overpass_bad_response" in joined:
        return 503
    if "overpass" in joined and ("timeout" in joined or "timed out" in joined):
        return 504
    if "overpass" in joined:
        return 503
    return None


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Keep FastAPI's structured detail for 422; add request_id.
        content: dict[str, Any] = {"code": "validation_error", "detail": exc.errors()}
        request_id = _request_id(request)
        if request_id:
            content["request_id"] = request_id
        return JSONResponse(status_code=422, content=content)

    @app.exception_handler(GeoAgentError)
    async def geoagent_error(request: Request, exc: GeoAgentError) -> JSONResponse:
        status = status_for_geoagent_error(exc)
        request_id = _request_id(request)
        logger.warning(
            "request failed code=%s status=%s request_id=%s",
            exc.code,
            status,
            request_id,
        )
        return JSONResponse(
            status_code=status,
            content=error_body(code=exc.code, message=exc.message, request_id=request_id),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = _request_id(request)
        detail = exc.detail if isinstance(exc.detail, str) else "request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(code="http_error", message=detail, request_id=request_id),
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        logger.exception("unexpected internal error request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content=error_body(
                code="internal_error",
                message="An unexpected internal error occurred.",
                request_id=request_id,
            ),
        )
