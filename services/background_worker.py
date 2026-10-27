"""In-process ingestion queue used when no Celery broker is configured.

Keeping a fallback matters for local development and single-container
deployments: ``POST /documents/async/upload`` still returns immediately, jobs
still run in the background, and ``GET /documents/async/status/{task_id}`` still
reports progress - without requiring Redis. When ``CELERY_BROKER_URL`` is set,
the lifespan leaves this queue stopped and Celery does the work instead.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from database import get_engine
from services import task_registry
from services.async_ingestion import run_ingestion_task

__all__ = [
    "BackgroundIngestionQueue",
    "broker_configured",
    "ingestion_queue",
    "start_inline_worker",
    "start_inline_worker_if_needed",
    "stop_inline_worker",
]

logger = logging.getLogger(__name__)

#: Presence of this variable means a real broker is available.
BROKER_ENV_VAR = "CELERY_BROKER_URL"


def broker_configured() -> bool:
    """Return ``True`` when a Celery broker URL is configured."""
    return bool(os.getenv(BROKER_ENV_VAR))


class BackgroundIngestionQueue:
    """A small asyncio worker pool consuming serialised ingestion jobs."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._running = False

    @property
    def running(self) -> bool:
        """Whether the worker pool has been started in this process."""
        return self._running

    async def start(self, num_workers: int = 2) -> None:
        """Spawn *num_workers* consumers (idempotent)."""
        if self._running:
            return
        self._running = True
        for index in range(num_workers):
            self._workers.append(asyncio.create_task(self._worker(f"worker-{index}")))
        logger.info("Started %d in-process ingestion worker(s)", num_workers)

    async def stop(self) -> None:
        """Drain and stop every consumer."""
        if not self._running:
            return
        self._running = False
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Stopped in-process ingestion workers")

    async def enqueue(self, task: dict[str, Any]) -> None:
        """Queue *task* for the next free worker."""
        await self._queue.put(task)
        logger.info("Queued ingestion task %s", task.get("task_id"))

    async def _worker(self, worker_id: str) -> None:
        """Consume jobs until the queue is stopped."""
        logger.info("%s started", worker_id)
        while self._running:
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                continue
            if task is None:
                self._queue.task_done()
                break
            try:
                task_registry.mark_running(task["task_id"])
                result = await run_ingestion_task(task, engine=get_engine())
                task_registry.mark_completed(task["task_id"], result)
                logger.info(
                    "%s indexed document %s", worker_id, result.get("document_id")
                )
            except Exception as exc:
                task_registry.mark_failed(task["task_id"], str(exc))
                logger.exception("%s failed task %s", worker_id, task.get("task_id"))
            finally:
                self._queue.task_done()
        logger.info("%s stopped", worker_id)


#: Process-wide fallback queue.
ingestion_queue = BackgroundIngestionQueue()


async def start_inline_worker(num_workers: int = 2) -> bool:
    """Start the fallback pool if it is not already running."""
    if ingestion_queue.running:
        return False
    await ingestion_queue.start(num_workers)
    return True


async def start_inline_worker_if_needed(num_workers: int = 2) -> bool:
    """Start the fallback pool only when no Celery broker is configured."""
    if broker_configured():
        return False
    return await start_inline_worker(num_workers)


async def stop_inline_worker() -> None:
    """Stop the fallback pool (no-op when it was never started)."""
    await ingestion_queue.stop()
