"""Async SQLAlchemy 2.0 engine/session management with tenant-aware session GUCs.

Responsibilities
----------------
1. Own the process-wide async engine and session factory (lazily created).
2. Provide the declarative :class:`Base` with a deterministic naming convention
   so Alembic autogenerate output is stable.
3. Propagate request identity (tenant / actor / request id) into PostgreSQL
   session parameters (``app.tenant_id``, ``app.actor_id``, ``app.request_id``)
   at the start of **every** transaction. The row-level security policies
   created by the baseline migration read ``app.tenant_id``, which makes tenant
   isolation a database-level guarantee rather than an application convention.

Usage
-----
    from typing import Annotated
    from fastapi import Depends
    from sqlalchemy.ext.asyncio import AsyncSession

    async def get_session() -> AsyncIterator[AsyncSession]:
        ...

    SessionDep = Annotated[AsyncSession, Depends(get_session)]

    # In a request handler / background job:
    with session_context(tenant_id=tenant_id, actor_id=user_id, request_id=request_id):
        ...
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from functools import lru_cache
from typing import Any, Final
from uuid import UUID

from pgvector.sqlalchemy import Vector  # noqa: F401  (registers Vector with SQLAlchemy)
from sqlalchemy import MetaData, create_engine, event, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from config import Settings, get_settings

logger: Final[logging.Logger] = logging.getLogger(__name__)

#: Deterministic object names -> Alembic diffs stay readable across environments.
NAMING_CONVENTION: Final[dict[str, str]] = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(column_0_N_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# --------------------------------------------------------------------------------------
# Request identity contextvars
# --------------------------------------------------------------------------------------
TENANT_ID_CTX: ContextVar[str | None] = ContextVar("tenant_id", default=None)
ACTOR_ID_CTX: ContextVar[str | None] = ContextVar("actor_id", default=None)
REQUEST_ID_CTX: ContextVar[str | None] = ContextVar("request_id", default=None)

#: PostgreSQL session parameter <- contextvar. Values are bind-parameterised,
#: so no SQL injection is possible even though GUC values are dynamic.
_SESSION_GUCS: Final[tuple[tuple[str, ContextVar[str | None]], ...]] = (
    ("app.tenant_id", TENANT_ID_CTX),
    ("app.actor_id", ACTOR_ID_CTX),
    ("app.request_id", REQUEST_ID_CTX),
)


class Base(DeclarativeBase):
    """Declarative base shared by every mapped class in :mod:`models`."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def bind_identity(
    *,
    tenant_id: str | UUID | None = None,
    actor_id: str | UUID | None = None,
    request_id: str | UUID | None = None,
) -> list[tuple[ContextVar[str | None], Token[str | None]]]:
    """Bind identity contextvars and return reset tokens for :func:`unbind_identity`.

    Prefer :func:`session_context`; this lower-level form exists for ``yield``-style
    FastAPI dependencies that must stay bound across the endpoint call.

    **Timing rule:** GUCs are applied by the ``begin`` event listener, i.e. at
    transaction start. If the session already has an open transaction, roll back
    first so the next ``BEGIN`` re-reads the contextvars - otherwise queries run
    with no ``app.tenant_id`` and RLS fail-closes to zero rows.
    """
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = []
    for ctx_var, value in (
        (TENANT_ID_CTX, tenant_id),
        (ACTOR_ID_CTX, actor_id),
        (REQUEST_ID_CTX, request_id),
    ):
        if value is not None:
            tokens.append((ctx_var, ctx_var.set(str(value))))
    return tokens


def unbind_identity(tokens: list[tuple[ContextVar[str | None], Token[str | None]]]) -> None:
    """Reset contextvars bound by :func:`bind_identity` (reverse order)."""
    for ctx_var, token in reversed(tokens):
        ctx_var.reset(token)


@contextmanager
def session_context(
    *,
    tenant_id: str | UUID | None = None,
    actor_id: str | UUID | None = None,
    request_id: str | UUID | None = None,
) -> Iterator[None]:
    """Bind request identity for the duration of the block.

    Every transaction opened while the block is active receives the identity as
    PostgreSQL session parameters, which is what the RLS policies match on.
    Contextvars are task-local, so this is safe under asyncio concurrency.

    The bound values are applied when a transaction *begins* (``SET LOCAL``
    semantics): unbinding later only clears the contextvars - a transaction that
    already started keeps its GUCs until commit/rollback. Consequently the
    session must not already hold an open transaction when entering this block;
    roll back first if it does (see :func:`bind_identity`).

    Args:
        tenant_id: Tenant whose rows should be visible to this transaction.
        actor_id: Authenticated user id (audit trail / debugging).
        request_id: Correlation id shared with API logs and audit rows.

    Yields:
        None, once the identity has been bound.
    """
    tokens = bind_identity(tenant_id=tenant_id, actor_id=actor_id, request_id=request_id)
    try:
        yield
    finally:
        unbind_identity(tokens)


def _register_engine_listeners(engine: AsyncEngine) -> None:
    """Apply the identity contextvars as session parameters per transaction."""

    @event.listens_for(engine.sync_engine, "begin")
    def _apply_identity_session_parameters(conn: Connection) -> None:
        for guc_name, ctx_var in _SESSION_GUCS:
            value = ctx_var.get()
            if not value:
                continue
            try:
                conn.execute(
                    text("SELECT set_config(:key, :value, true)"),
                    {"key": guc_name, "value": value},
                )
            except SQLAlchemyError:
                # Fail loudly: silently losing app.tenant_id would make every
                # RLS-protected query return an empty result set.
                logger.exception(
                    "Failed to set session parameter %s; rolling back transaction", guc_name
                )
                raise


def _build_async_engine(settings: Settings) -> AsyncEngine:
    """Create the async engine from validated settings."""
    kwargs: dict[str, Any] = {
        "echo": settings.db_echo,
        "echo_pool": "debug" if settings.debug else False,
        "pool_pre_ping": True,
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_timeout": settings.db_pool_timeout,
        "pool_recycle": settings.db_pool_recycle,
        "future": True,
    }
    if settings.environment == "test":
        # Tests open short-lived connections; pooling them leaks server slots.
        kwargs["poolclass"] = NullPool
    return create_async_engine(settings.async_database_url, **kwargs)


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Return the process-wide async engine (created on first use)."""
    engine = _build_async_engine(get_settings())
    _register_engine_listeners(engine)
    logger.debug("Async engine created for %s", engine.url.render_as_string(hide_password=True))
    return engine


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide :class:`async_sessionmaker` factory."""
    return async_sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding an :class:`AsyncSession` with safe teardown.

    Rolls back any uncommitted work on error and always closes the session,
    returning its connection to the pool.
    """
    async with get_session_factory()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


#: Reads the RLS posture of the most sensitive table. World-readable catalogs:
#: no special grants required. ``document_chunks`` is the probe because it holds
#: every tenant's content - the row a bypass would actually leak.
_RLS_POSTURE_SQL: Final[str] = """
    SELECT current_user AS session_role,
           pg_get_userbyid(c.relowner) AS table_owner,
           c.relrowsecurity AS rls_enabled,
           c.relforcerowsecurity AS rls_forced,
           COALESCE(
               (SELECT rolbypassrls FROM pg_roles WHERE rolname = current_user),
               false
           ) AS bypasses_rls
      FROM pg_class c
      JOIN pg_namespace n ON n.oid = c.relnamespace
     WHERE c.relname = 'document_chunks'
       AND c.relkind = 'r'
       AND n.nspname NOT IN ('pg_catalog', 'information_schema')
"""


async def _verify_rls_enforcement() -> None:
    """Fail fast when the runtime role would bypass row-level security.

    ``ENABLE ROW LEVEL SECURITY`` does not constrain a table's owner (unless
    ``FORCE`` is set) or a ``BYPASSRLS`` role - either case would silently hand
    the application *every* tenant's rows while all queries still "succeed".
    Production raises; development/test logs a warning so the owner-credentials
    quickstart still boots (with the risk visible in the logs).
    """
    settings = get_settings()
    issues: list[str] = []
    async with get_engine().connect() as connection:
        row = (await connection.execute(text(_RLS_POSTURE_SQL))).first()
    if row is None:
        logger.debug(
            "document_chunks not found (pre-migration database); "
            "skipping RLS enforcement check"
        )
        return
    if not row.rls_enabled:
        issues.append("ROW LEVEL SECURITY is not enabled on document_chunks")
    if row.bypasses_rls:
        issues.append(f"role {row.session_role!r} has BYPASSRLS")
    if row.session_role == row.table_owner and not row.rls_forced:
        issues.append(
            f"role {row.session_role!r} owns document_chunks and RLS is not "
            "FORCED; table owners bypass RLS by default"
        )
    if not issues:
        logger.info(
            "RLS enforcement verified: session_role=%s owner=%s enabled=%s forced=%s",
            row.session_role,
            row.table_owner,
            row.rls_enabled,
            row.rls_forced,
        )
        return
    detail = (
        "Tenant isolation is NOT enforced for this connection: "
        + "; ".join(issues)
        + ". Connect as the non-owner application role (see README section 4: "
        "rag_owner/rag_app) or enable FORCE ROW LEVEL SECURITY."
    )
    if settings.is_production:
        raise RuntimeError(detail)
    logger.warning(detail)


async def init_database() -> None:
    """Verify connectivity, ensure pgvector, verify RLS enforcement.

    Runs on application startup so a fresh environment becomes usable without
    manual SQL. Table/index creation is owned by Alembic, never by the app.

    Raises:
        RuntimeError: When the database is unreachable, the role is not
            allowed to create the ``vector`` extension, or (in production) the
            runtime role would bypass row-level security.
    """
    try:
        async with get_engine().begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except SQLAlchemyError as exc:
        raise RuntimeError(
            "Database initialisation failed: could not reach PostgreSQL or create "
            "the 'vector' extension. Verify POSTGRES_* settings, that the "
            "pgvector extension is installed, and that the application role may "
            "create extensions (or pre-create it as a superuser)."
        ) from exc
    logger.info("Database reachable; pgvector extension ensured")
    await _verify_rls_enforcement()


async def check_connection() -> bool:
    """Return ``True`` when the database answers a trivial query."""
    try:
        async with get_engine().connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        logger.warning("Database health check failed", exc_info=True)
        return False


async def dispose_engine() -> None:
    """Dispose the engine pool and drop cached factories (idempotent)."""
    if get_engine.cache_info().currsize == 0:
        return
    await get_engine().dispose()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
    logger.info("Async engine disposed")


def create_sync_engine(url: str | None = None) -> Engine:
    """Create a standalone sync engine (used by Alembic and ops scripts).

    Args:
        url: Optional DSN override; defaults to ``settings.sync_database_url``.

    Returns:
        A configured sync :class:`Engine` using psycopg2.
    """
    settings = get_settings()
    target = url or settings.sync_database_url
    return create_engine(
        target,
        echo=settings.db_echo,
        pool_pre_ping=True,
        future=True,
    )
