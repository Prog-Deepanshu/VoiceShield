"""
VoiceGuard — API Middleware
============================
Security, observability, and CORS middleware for the FastAPI application.
"""

import logging
import time
import uuid
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config.settings import settings

logger = logging.getLogger(__name__)

# API Key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """
    Verify the API key if authentication is enabled.

    If settings.api_key is None/empty, auth is disabled and all
    requests are allowed through.
    """
    if not settings.api_key:
        return None  # Auth disabled

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide it via X-API-Key header.",
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return api_key


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs all requests with timing and assigns request IDs.

    Adds headers:
        - X-Request-ID: Unique request identifier for tracing
        - X-Process-Time: Server processing time in seconds
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        # Attach request ID to state for downstream access
        request.state.request_id = request_id

        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"client={request.client.host if request.client else 'unknown'}"
        )

        try:
            response = await call_next(request)
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"FAILED in {duration:.3f}s — {e}"
            )
            raise

        duration = time.perf_counter() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration:.4f}"

        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"status={response.status_code} duration={duration:.3f}s"
        )

        return response
