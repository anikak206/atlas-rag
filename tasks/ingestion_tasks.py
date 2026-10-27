"""Celery task for asynchronous document ingestion.

The job body lives in :mod:`services.async_ingestion` so the Celery worker and
the in-process fallback perform identical work. This module only maps failures
onto Celery semantics: bad input fails immediately, transient infrastructure
errors are retried with a delay.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from celery_app import celery_app
from database import get_engine
from services.async_ingestion import run_ingestion_task
from services.ingestion import DuplicateDocumentError, EmptyDocumentError
from services.providers import ProviderConfigurationError

logger = logging.getLogger(__name__)

#: Permanent failures: retrying cannot fix bad input, duplicate content or a
#: missing provider credential. ``TaskPayloadError`` is a ``ValueError``.
NON_RETRYABLE: tuple[type[BaseException], ...] = (
    DuplicateDocumentError,
    EmptyDocumentError,
    ProviderConfigurationError,
    ValueError,
)

#: Transient failures worth another attempt (database down, network blip).
RETRYABLE: tuple[type[BaseException], ...] = (
    SQLAlchemyError,
    OSError,
    TimeoutError,
)

RETRY_COUNTDOWN_SECONDS = 60
MAX_RETRIES = 3


@celery_app.task(bind=True, name="ingest_document_async", max_retries=MAX_RETRIES)
def ingest_document_async(self: Any, task_data: dict[str, Any]) -> dict[str, Any]:
    """Chunk, embed and index one uploaded document on a worker."""
    task_id = task_data.get("task_id")
    try:
        return asyncio.run(run_ingestion_task(task_data))
    except NON_RETRYABLE as exc:
        logger.warning("Rejecting ingestion task %s: %s", task_id, exc)
        return {"status": "failed", "error": str(exc), "retryable": False}
    except RETRYABLE as exc:
        logger.warning("Retrying ingestion task %s in %ss: %s", task_id, RETRY_COUNTDOWN_SECONDS, exc)
        raise self.retry(exc=exc, countdown=RETRY_COUNTDOWN_SECONDS) from exc
    finally:
        # Each Celery task runs in its own event loop, so pooled asyncpg
        # connections from the previous loop must never be reused.
        get_engine.cache_clear()
