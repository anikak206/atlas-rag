"""Append-only audit log construction and persistence.

``build_audit_log`` is a pure function (unit-testable without a database);
``record_audit`` flushes the row so callers receive the generated identity
before committing. Every RAG query - successful or not - must produce exactly
one audit row; the query route owns that guarantee.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from models import AuditLog, User


def coerce_request_id(value: uuid.UUID | str | None) -> uuid.UUID:
    """Accept the middleware's correlation id when it is a UUID, else mint one.

    The middleware echoes arbitrary client-supplied ``X-Request-ID`` strings;
    audit rows and API responses need a real UUID, so non-UUID correlation ids
    are replaced (they remain available in HTTP headers and application logs).
    """
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str):
        try:
            return uuid.UUID(value)
        except ValueError:
            pass
    return uuid.uuid4()


def _store_response(response_text: str | None) -> str | None:
    """Apply the response-retention policy (off by default, truncated when on)."""
    settings = get_settings()
    if response_text is None or not settings.audit_store_response_snippet:
        return None
    limit = settings.audit_response_snippet_chars
    if limit <= 0:
        return None
    if len(response_text) <= limit:
        return response_text
    return response_text[: limit - 1] + "\u2026"


def build_audit_log(
    *,
    tenant_id: uuid.UUID,
    query_text: str,
    latency_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    user: User | None = None,
    request_id: uuid.UUID | str | None = None,
    model_name: str | None = None,
    retrieved_chunk_count: int | None = None,
    success: bool = True,
    error_code: str | None = None,
    error_message: str | None = None,
    response_text: str | None = None,
    filters: dict[str, Any] | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    response_time: datetime | None = None,
) -> AuditLog:
    """Build (but do not persist) one audit row.

    Args:
        tenant_id: Tenant the query ran in.
        query_text: The user's question.
        latency_ms: End-to-end latency; negative values are clamped to 0.
        prompt_tokens: Prompt token count (provider-reported or estimated).
        completion_tokens: Completion token count (provider-reported or estimated).
        user: Actor, when authenticated - snapshots email and role at query time.
        request_id: Correlation id from the middleware; non-UUID values are replaced.
        model_name: LLM used for generation.
        retrieved_chunk_count: Candidates returned by vector search (pre-RBAC cut).
        success: False when the request ended in an error.
        error_code: Stable machine-readable error code.
        error_message: Human-readable failure detail (truncated for safety).
        response_text: Answer text, stored only when retention is enabled.
        filters: Retrieval filters that were applied (tenant/RBAC/metadata).
        ip_address: Client address (IPv4/IPv6).
        user_agent: Client user-agent header.
        response_time: Completion wall-clock time; defaults to now (UTC).

    Returns:
        An unsaved :class:`AuditLog` ready for :func:`record_audit`.
    """
    return AuditLog(
        request_id=coerce_request_id(request_id),
        tenant_id=tenant_id,
        user_id=user.id if user else None,
        actor_email=user.email if user else None,
        actor_role=user.role if user else None,
        query_text=query_text,
        response_time=response_time or datetime.now(UTC),
        latency=max(0.0, float(latency_ms)),
        prompt_tokens=max(0, int(prompt_tokens)),
        completion_tokens=max(0, int(completion_tokens)),
        tokens_used=max(0, int(prompt_tokens)) + max(0, int(completion_tokens)),
        model_name=model_name,
        retrieved_chunk_count=retrieved_chunk_count,
        success=success,
        error_code=error_code,
        error_message=error_message[:2000] if error_message else None,
        response_text=_store_response(response_text),
        filters=filters,
        ip_address=ip_address,
        user_agent=user_agent[:512] if user_agent else None,
    )


async def record_audit(session: AsyncSession, log: AuditLog) -> int:
    """Add *log* to the session and flush so its identity id is available.

    The caller owns the transaction: commit persists the audit row (and any
    preceding writes), rollback discards them.
    """
    session.add(log)
    await session.flush()
    if log.id is None:  # pragma: no cover - defensive
        raise RuntimeError("Audit row did not receive an id on flush")
    return int(log.id)
