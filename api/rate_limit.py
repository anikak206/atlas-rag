"""Enterprise rate limiting middleware using slowapi."""

from __future__ import annotations

import logging

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# Create limiter with user/key-based identification
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])


def get_user_identifier(request: Request) -> str:
    """Get rate limit key from user or IP."""
    # Try to get user from request state if authenticated
    user = getattr(request.state, "user", None)
    if user and hasattr(user, "id"):
        return f"user:{user.id}"
    return get_remote_address(request)


# Per-endpoint limits
RATE_LIMITS = {
    "query": "30/minute",      # Expensive LLM operations
    "documents_upload": "10/minute",
    "documents_ingest": "20/minute",
    "auth_login": "5/minute",  # Prevent brute force
    "auth_signup": "3/minute",
    "default": "100/minute",
}


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Custom handler for rate limit exceeded."""
    logger.warning("Rate limit exceeded for %s on %s", get_remote_address(request), request.url.path)
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please try again later.",
            "error": "rate_limit_exceeded",
        },
    )
