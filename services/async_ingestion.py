"""Shared ingestion job runner for Celery and the in-process fallback worker.

Both backends must perform exactly the same work, so the job body lives here
rather than in the Celery task. The runner owns the transaction, binds the
tenant identity for RLS, reloads the uploading user, and hands the payload to
:func:`services.ingestion.ingest_text`.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from database import get_engine, session_context
from models import User, UserRole
from schemas.documents import DocumentIngestRequest
from services.ingestion import IngestResult, ingest_text

__all__ = ["REQUIRED_FIELDS", "TaskPayloadError", "build_ingest_request", "run_ingestion_task"]

logger = logging.getLogger(__name__)

#: Fields a queued job cannot run without. Everything else has a default.
REQUIRED_FIELDS = ("task_id", "tenant_id", "user_id", "title", "content")


class TaskPayloadError(ValueError):
    """A queued job is missing required data or references unknown rows."""


def build_ingest_request(task_data: dict[str, Any]) -> DocumentIngestRequest:
    """Convert a JSON-serialisable job payload into a validated ingest request.

    Raises:
        TaskPayloadError: A required field is absent or empty.
        pydantic.ValidationError: A field fails schema validation.
    """
    missing = [field for field in REQUIRED_FIELDS if not task_data.get(field)]
    if missing:
        raise TaskPayloadError(
            "Ingestion job is missing required fields: " + ", ".join(missing)
        )
    return DocumentIngestRequest(
        title=str(task_data["title"]),
        content=str(task_data["content"]),
        source_uri=task_data.get("source_uri"),
        source_system=task_data.get("source_system"),
        mime_type=str(task_data.get("mime_type") or "text/plain"),
        min_role_required=UserRole(str(task_data.get("min_role_required") or UserRole.INTERN.value)),
        metadata=dict(task_data.get("metadata") or {}),
    )


async def run_ingestion_task(
    task_data: dict[str, Any],
    *,
    engine: Any | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> dict[str, Any]:
    """Chunk, embed and index one queued document.

    Args:
        task_data: Job payload produced by the upload route.
        engine: Existing engine to reuse (the inline worker shares the app
            pool). When omitted a per-job engine is created and disposed, which
            is what Celery needs because each task runs in a fresh event loop.
        session_factory: Optional factory bound to *engine* (tests).

    Returns:
        JSON-serialisable result suitable for a task backend.

    Raises:
        TaskPayloadError: Payload is malformed or the user no longer exists.
        services.ingestion.IngestionError: Chunking/embedding failed.
    """
    tenant_id = UUID(str(task_data["tenant_id"]))
    user_id = UUID(str(task_data["user_id"]))
    payload = build_ingest_request(task_data)

    owned_engine = engine is None
    engine = engine if engine is not None else get_engine()
    factory = session_factory or async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            with session_context(
                tenant_id=tenant_id,
                actor_id=user_id,
                request_id=task_data.get("task_id"),
            ):
                actor = await session.get(User, user_id)
                if actor is None:
                    raise TaskPayloadError(
                        f"User {user_id} does not exist in tenant {tenant_id}"
                    )
                result: IngestResult = await ingest_text(
                    session, tenant_id=tenant_id, actor=actor, payload=payload
                )
                await session.commit()

        logger.info(
            "Indexed document %s (%d chunks) for task %s",
            result.document_id,
            result.chunks_created,
            task_data.get("task_id"),
        )
        return {
            "status": "completed",
            "document_id": str(result.document_id),
            "chunks_created": result.chunks_created,
            "elapsed_ms": round(result.elapsed_ms, 2),
            "content_hash": result.content_hash,
        }
    finally:
        if owned_engine:
            await engine.dispose()
