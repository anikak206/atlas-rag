"""Asynchronous document upload: parse now, embed on a worker.

Extraction is fast and must fail while the client is still connected, so the
request parses the file and only the expensive step (chunking + embedding) is
deferred. Celery is used when ``CELERY_BROKER_URL`` is configured; otherwise the
job goes to the in-process queue started by the application lifespan.

Every job is scoped to the uploader's tenant. Task ids are UUID4 values, but the
status endpoint still verifies tenant ownership instead of relying on
unguessability alone.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Annotated, Any, Final

from celery.exceptions import CeleryError
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from kombu.exceptions import OperationalError as BrokerError

from api.deps import CurrentUser
from celery_app import celery_app
from models import UserRole
from services import task_registry
from services.background_worker import ingestion_queue, start_inline_worker
from services.uploads import (
    MAX_UPLOAD_BYTES,
    MissingParserError,
    UnsupportedUploadError,
    UploadRejectedError,
    UploadTooLargeError,
    extract_upload,
)
from tasks.ingestion_tasks import ingest_document_async

logger: Final[logging.Logger] = logging.getLogger(__name__)

router = APIRouter(prefix="/documents/async", tags=["documents-async"])

#: Bounds the status endpoint's wait on a possibly unreachable result backend.
STATUS_POLL_TIMEOUT_SECONDS: Final[float] = 2.0

_CELERY_READY_STATES: Final[frozenset[str]] = frozenset(
    {"SUCCESS", "FAILURE", "REVOKED"}
)


def _upload_http_error(exc: UploadRejectedError) -> HTTPException:
    """Map an upload validation failure onto the matching status code."""
    if isinstance(exc, MissingParserError):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    if isinstance(exc, UnsupportedUploadError):
        return HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc))
    if isinstance(exc, UploadTooLargeError):
        return HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(exc))
    return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))


async def _dispatch(task_data: dict[str, Any]) -> str:
    """Hand *task_data* to Celery, falling back to the in-process queue.

    Returns the backend that accepted the job (``"celery"`` or ``"inline"``).
    """
    try:
        # retry=False: an unavailable broker must fail immediately so the
        # request can fall back instead of blocking on connection retries.
        ingest_document_async.apply_async((task_data,), retry=False)
        return "celery"
    except (BrokerError, CeleryError, OSError) as exc:
        logger.warning(
            "Celery broker unavailable (%s); falling back to the in-process queue", exc
        )

    task_registry.register(task_data["task_id"], task_data["tenant_id"])
    # The lifespan only starts this pool when no broker is configured, so a
    # broker that is configured but currently down needs it started here.
    await start_inline_worker()
    await ingestion_queue.enqueue(task_data)
    return "inline"


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a document for background ingestion",
)
async def upload_document_async(
    user: CurrentUser,
    file: Annotated[UploadFile, File(description="PDF, DOCX, PPTX, or text file")],
    min_role_required: Annotated[UserRole, Form()] = UserRole.INTERN,
    source_system: Annotated[str | None, Form(max_length=64)] = None,
    title: Annotated[str | None, Form(max_length=1000)] = None,
) -> dict[str, Any]:
    """Validate and parse the upload, then queue it; returns a task id."""
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        if len(raw) > MAX_UPLOAD_BYTES:
            raise UploadTooLargeError("File exceeds size limit")
        upload = await asyncio.to_thread(
            extract_upload, raw, file.content_type, file.filename, title=title
        )
    except UploadRejectedError as exc:
        raise _upload_http_error(exc) from exc

    task_id = str(uuid.uuid4())
    task_data = {
        "task_id": task_id,
        "tenant_id": str(user.tenant_id),
        "user_id": str(user.id),
        "title": upload.title,
        "content": upload.content,
        "mime_type": upload.mime_type,
        "min_role_required": min_role_required.value,
        "source_system": source_system,
        "metadata": upload.metadata,
    }
    backend = await _dispatch(task_data)
    logger.info(
        "Queued document %r for tenant %s (task %s via %s)",
        upload.title,
        user.tenant_id,
        task_id,
        backend,
    )
    return {
        "task_id": task_id,
        "status": "queued",
        "backend": backend,
        "filename": file.filename,
        "title": upload.title,
        "characters": len(upload.content),
        "message": "Document queued. Poll the status endpoint for progress.",
    }


async def _celery_state(task_id: str) -> dict[str, Any] | None:
    """Read task state from Celery, or ``None`` if the backend is unreachable."""
    try:
        from celery.result import AsyncResult

        result = AsyncResult(task_id, app=celery_app)
        state = await asyncio.wait_for(
            asyncio.to_thread(lambda: result.state),
            timeout=STATUS_POLL_TIMEOUT_SECONDS,
        )
        if state not in _CELERY_READY_STATES:
            return {"status": state.lower(), "ready": False, "result": None}
        payload = result.result
        body: dict[str, Any] = {
            "status": state.lower(),
            "ready": True,
            "result": payload if state == "SUCCESS" else None,
        }
        if state == "FAILURE":
            body["error"] = str(payload)
        return body
    except (BrokerError, CeleryError, OSError, TimeoutError) as exc:
        logger.warning("Result backend unavailable for task %s: %s", task_id, exc)
        return None


@router.get(
    "/status/{task_id}",
    summary="Check the status of a queued ingestion task",
)
async def get_task_status(task_id: str, user: CurrentUser) -> dict[str, Any]:
    """Return the state of a job belonging to the caller's tenant."""
    record = task_registry.get(task_id)
    if record is not None:
        if record.tenant_id != user.tenant_id:
            # Same response as an unknown id: do not confirm other tenants' jobs.
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown task")
        return record.as_response()

    celery_state = await _celery_state(task_id)
    if celery_state is None:
        return {
            "task_id": task_id,
            "status": "unknown",
            "ready": False,
            "result": None,
            "backend": "celery",
            "detail": (
                "This instance has no record of the task and the Celery result "
                "backend is unreachable."
            ),
        }
    return {"task_id": task_id, "backend": "celery", **celery_state}
