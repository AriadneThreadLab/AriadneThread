"""HTTP middleware for correlation IDs and safe request logging."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

CallNext = Callable[[Request], Awaitable[Response]]


async def request_context_middleware(request: Request, call_next: CallNext) -> Response:
    """Attach a request ID and log operational metadata only.

    Implemented as a plain ``@app.middleware("http")`` function so FastAPI
    exception handlers still run (unlike ``BaseHTTPMiddleware``).
    """
    incoming = request.headers.get(REQUEST_ID_HEADER)
    request_id = incoming.strip() if incoming and incoming.strip() else str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000.0
    response.headers[REQUEST_ID_HEADER] = request_id
    logger.info(
        "http method=%s path=%s status=%s duration_ms=%.1f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
    )
    return response
