"""FastAPI application factory and operational endpoints.

Run locally with::

    uvicorn main:app --reload

Startup guarantees: validates settings, configures logging, verifies database
connectivity and ensures the pgvector extension exists. Schema migrations stay
the responsibility of Alembic.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Final

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from api.routes import audit as audit_routes
from api.routes import auth as auth_routes
from api.routes import documents as documents_routes
from api.routes import query as query_routes
from config import Settings, get_settings
from database import REQUEST_ID_CTX, check_connection, dispose_engine, init_database

logger: Final[logging.Logger] = logging.getLogger(__name__)

_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging, prepare the database and tear the pool down."""
    settings: Settings = app.state.settings
    logging.basicConfig(level=settings.log_level, format=_LOG_FORMAT)
    logger.info(
        "Starting %s | env=%s debug=%s", settings.app_name, settings.environment, settings.debug
    )
    try:
        await init_database()
    except RuntimeError:
        logger.critical("Startup aborted: database unavailable", exc_info=True)
        raise
    try:
        yield
    finally:
        await dispose_engine()
        logger.info("Shutdown complete")


def create_application() -> FastAPI:
    """Build the FastAPI application with middleware and ops endpoints."""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Multi-tenant enterprise knowledge RAG engine with RBAC-aware "
            "retrieval and append-only audit logging."
        ),
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    application.state.settings = settings

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )

    @application.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Bind the correlation id to contextvars and echo it on the response."""
        request_id = request.headers.get("X-Request-ID", "").strip()[:64]
        if not request_id:
            request_id = str(uuid.uuid4())
        token = REQUEST_ID_CTX.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            REQUEST_ID_CTX.reset(token)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return response

    # Versioned business API; ops probes stay at the root for load balancers.
    api_prefix = settings.api_prefix
    application.include_router(auth_routes.router, prefix=api_prefix)
    application.include_router(documents_routes.router, prefix=api_prefix)
    application.include_router(query_routes.router, prefix=api_prefix)
    application.include_router(audit_routes.router, prefix=api_prefix)

    @application.get("/health", tags=["ops"], summary="Liveness probe")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/ready", tags=["ops"], summary="Readiness probe")
    async def ready(response: Response) -> dict[str, str]:
        if await check_connection():
            return {"status": "ready", "database": "ok"}
        response.status_code = 503
        return {"status": "unavailable", "database": "unreachable"}

    return application


app = create_application()
