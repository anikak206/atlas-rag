"""In-process registry of ingestion jobs handled by the fallback worker.

Celery keeps task state in its result backend. When no broker is configured the
application processes jobs inline, so it needs somewhere to remember what
happened; that is this module. Entries are bounded and evicted oldest-first so a
long-running process cannot leak memory, and every record carries its tenant so
the status endpoint can refuse cross-tenant lookups.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

__all__ = [
    "MAX_ENTRIES",
    "TaskRecord",
    "clear",
    "get",
    "mark_completed",
    "mark_failed",
    "mark_running",
    "register",
]

MAX_ENTRIES = 1024


@dataclass(slots=True)
class TaskRecord:
    """Lifecycle of one inline ingestion job."""

    task_id: str
    tenant_id: UUID
    status: str = "queued"
    result: dict[str, Any] | None = None
    error: str | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_response(self) -> dict[str, Any]:
        """Project the record into the public status payload."""
        body: dict[str, Any] = {
            "task_id": self.task_id,
            "status": self.status,
            "ready": self.status in {"completed", "failed"},
            "result": self.result,
            "backend": "inline",
        }
        if self.error is not None:
            body["error"] = self.error
        return body


_records: OrderedDict[str, TaskRecord] = OrderedDict()
_lock = threading.Lock()


def _evict_locked() -> None:
    while len(_records) > MAX_ENTRIES:
        _records.popitem(last=False)


def register(task_id: str, tenant_id: UUID | str) -> TaskRecord:
    """Create (or reset) the record for *task_id*."""
    with _lock:
        record = TaskRecord(task_id=task_id, tenant_id=UUID(str(tenant_id)))
        _records[task_id] = record
        _records.move_to_end(task_id)
        _evict_locked()
        return record


def get(task_id: str) -> TaskRecord | None:
    """Return the record for *task_id*, if this process handled it."""
    with _lock:
        return _records.get(task_id)


def _update(task_id: str, **changes: Any) -> None:
    with _lock:
        record = _records.get(task_id)
        if record is None:
            return
        for attribute, value in changes.items():
            setattr(record, attribute, value)
        record.updated_at = datetime.now(UTC)


def mark_running(task_id: str) -> None:
    """Mark a queued job as actively being processed."""
    _update(task_id, status="processing")


def mark_completed(task_id: str, result: dict[str, Any]) -> None:
    """Store the successful ingestion result."""
    _update(task_id, status="completed", result=result, error=None)


def mark_failed(task_id: str, error: str) -> None:
    """Store a terminal failure message."""
    _update(task_id, status="failed", error=error)


def clear() -> None:
    """Drop every record (used by tests)."""
    with _lock:
        _records.clear()
