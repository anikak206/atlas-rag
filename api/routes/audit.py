"""Admin-only audit readback (``GET /audit``).

The ledger is append-only for the application; this endpoint is the supported
read path - filtered, paginated and tenant-scoped by RLS plus an explicit
``tenant_id`` predicate. 403 for non-Admins comes from the shared
:func:`api.deps.require_role` guard.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.sql.elements import ColumnElement

from api.deps import PageParamsDep, SessionDep, require_role
from models import AuditLog, User, UserRole
from schemas.common import Page
from schemas.query import AuditRead

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get(
    "",
    response_model=Page[AuditRead],
    summary="Query audit trail (Admin)",
)
async def list_audit_logs(
    admin: Annotated[
        User, Depends(require_role(UserRole.ADMIN, resource="audit logs"))
    ],
    session: SessionDep,
    page: PageParamsDep,
    user_id: Annotated[UUID | None, Query(description="Filter by actor id")] = None,
    success: Annotated[bool | None, Query(description="Filter by outcome")] = None,
    since: Annotated[
        datetime | None,
        Query(description="Only rows created at/after this instant (ISO-8601)"),
    ] = None,
) -> Page[AuditRead]:
    filters: list[ColumnElement[bool]] = [AuditLog.tenant_id == admin.tenant_id]
    if user_id is not None:
        filters.append(AuditLog.user_id == user_id)
    if success is not None:
        filters.append(AuditLog.success == success)
    if since is not None:
        filters.append(AuditLog.created_at >= since)

    total = (
        await session.scalar(
            select(func.count()).select_from(AuditLog).where(*filters)
        )
        or 0
    )
    rows = (
        (
            await session.execute(
                select(AuditLog)
                .where(*filters)
                .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
                .limit(page.limit)
                .offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    return Page.build(
        [AuditRead.model_validate(row) for row in rows],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )
