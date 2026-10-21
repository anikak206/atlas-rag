"""HTTP-layer glue that can be tested without a database: RBAC guards,
auth-error mapping, pagination parsing and audit request context.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from api.deps import (
    PageParams,
    auth_error_to_http,
    client_audit_context,
    get_page_params,
    require_role,
)
from models import UserRole
from services.auth import (
    AuthError,
    InactiveTenantError,
    InactiveUserError,
    InvalidCredentialsError,
    TenantAlreadyExistsError,
    UserNotFoundError,
)


def _fake_user(role: UserRole) -> Any:
    return SimpleNamespace(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        email=f"{role.value}@acme.test",
        role=role,
    )


async def _first(agen: AsyncIterator[Any]) -> Any:
    try:
        return await agen.__anext__()
    finally:
        await agen.aclose()


@pytest.mark.parametrize(
    ("held", "required", "allowed"),
    [
        (UserRole.ADMIN, UserRole.ADMIN, True),
        (UserRole.ADMIN, UserRole.INTERN, True),
        (UserRole.FINANCE, UserRole.INTERN, True),
        (UserRole.FINANCE, UserRole.ADMIN, False),
        (UserRole.INTERN, UserRole.FINANCE, False),
    ],
)
async def test_role_guard_matches_hierarchy(
    held: UserRole, required: UserRole, allowed: bool
) -> None:
    guard = require_role(required, resource="audit logs")
    user = _fake_user(held)
    if allowed:
        assert await _first(guard(user)) is user
        return
    with pytest.raises(HTTPException) as exc_info:
        await _first(guard(user))
    assert exc_info.value.status_code == 403
    assert "audit logs" in str(exc_info.value.detail)


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (InvalidCredentialsError("nope"), 401),
        (UserNotFoundError("gone"), 401),
        (InactiveTenantError("suspended"), 403),
        (InactiveUserError("disabled"), 403),
        (TenantAlreadyExistsError("taken"), 409),
        (AuthError("unexpected"), 500),
    ],
)
def test_auth_errors_map_to_http_statuses(error: AuthError, status_code: int) -> None:
    response = auth_error_to_http(error)
    assert response.status_code == status_code


async def test_page_params_are_returned_verbatim() -> None:
    assert await get_page_params(limit=5, offset=10) == PageParams(limit=5, offset=10)


def test_client_audit_context_reads_ip_and_user_agent() -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/query",
            "headers": [(b"user-agent", b"pytest-agent/1.0")],
            "client": ("203.0.113.7", 54321),
            "query_string": b"",
        }
    )
    assert client_audit_context(request) == ("203.0.113.7", "pytest-agent/1.0")


def test_client_audit_context_tolerates_missing_client_and_agent() -> None:
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/health",
            "headers": [],
            "query_string": b"",
        }
    )
    assert client_audit_context(request) == (None, None)
