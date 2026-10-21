"""Background document ingestion worker using asyncio tasks."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from database import get_engine, session_context

logger = logging.getLogger(__name__)


class BackgroundIngestionQueue:
    """In-memory async queue for background document processing."""
    
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._running = False
    
    async def start(self, num_workers: int = 2):
        """Start background workers."""
        if self._running:
            return
        self._running = True
        for i in range(num_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(task)
        logger.info(f"Started {num_workers} background ingestion workers")
    
    async def stop(self):
        """Stop all workers."""
        self._running = False
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Stopped background ingestion workers")
    
    async def enqueue(self, task: dict[str, Any]):
        """Enqueue a document for background processing."""
        await self._queue.put(task)
        logger.info(f"Enqueued document {task.get('document_id')} for background processing")
    
    async def _worker(self, worker_id: str):
        """Worker loop to process queued documents."""
        logger.info(f"{worker_id} started")
        engine = get_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        
        while self._running:
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                if task is None:
                    break
                
                async with session_factory() as session:
                    try:
                        with session_context(tenant_id=task["tenant_id"]):
                            # Processing would happen here
                            logger.info(f"{worker_id} processing document {task['document_id']}")
                            await asyncio.sleep(0.1)  # Simulate processing
                        await session.commit()
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"{worker_id} failed to process {task['document_id']}: {e}")
                        await session.rollback()
                    finally:
                        self._queue.task_done()
            except TimeoutError:
                continue
            except Exception as e:  # noqa: BLE001
                logger.error(f"{worker_id} error: {e}")
        
        logger.info(f"{worker_id} stopped")


# Global queue instance
ingestion_queue = BackgroundIngestionQueue()
