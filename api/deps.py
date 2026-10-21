"""FastAPI dependencies: sessions, authenticated identity, RBAC guards, paging.

Request-scoped tenancy
----------------------
``get_current_user`` wraps identity loading in :func:`database.session_context`
around a ``yield``: the tenant GUC is bound before the session's first
transaction begins, stays effective for every query the endpoint issues, and is
reset on teardown. Authorization failures become 401/403 before any query runs;
the role itself is re-read from the database on every request (see
:func:`services.auth.load_user`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session, session_context
from models import User, UserRole
from rbac import AuthorizationError
from rbac import require_role as check_role
from schemas.common import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from services.auth import (
    AuthError,
    InactiveTenantError,
    InactiveUserError,
    InvalidCredentialsError,
    TenantAlreadyExistsError,
    UserNotFoundError,
    load_user,
)
from services.security import (
    InvalidTokenError,
    TokenClaims,
    TokenExpiredError,
    decode_token,
)

_bearer = HTTPBearer(auto_error=False, scheme_name="JWT")

__all__ = [
    "ClaimsDep",
    "CurrentUser",
    "PageParams",
    "PageParamsDep",
    "SessionDep",
    "auth_error_to_http",
    "client_audit_context",
    "get_claims",
    "get_current_user",
    "get_page_params",
    "require_role",
]


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_claims(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> TokenClaims:
    """Verify the ``Authorization: Bearer`` JWT and return its claims.

    Raises:
        HTTPException: 401 for missing, malformed, expired, tampered or
            wrong-type (refresh presented as access) tokens.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("Not authenticated")
    try:
        return decode_token(credentials.credentials, expected_type="access")
    except TokenExpiredError as exc:
        raise _unauthorized("Access token has expired; obtain a new one") from exc
    except InvalidTokenError as exc:
        raise _unauthorized("Invalid access token") from exc


async def get_current_user(
    claims: Annotated[TokenClaims, Depends(get_claims)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AsyncIterator[User]:
    """Yield the active user for the request with tenant identity bound.

    Tenant binding is active for the whole endpoint (it wraps the ``yield``);
    teardown resets the contextvars while the loaded transaction keeps its
    ``SET LOCAL`` GUCs until the session closes.
    """
    with session_context(tenant_id=claims.tenant_id, actor_id=claims.subject):
        try:
            user = await load_user(session, claims)
        except UserNotFoundError as exc:
            raise _unauthorized(str(exc)) from exc
        except InactiveUserError as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
        yield user


CurrentUser = Annotated[User, Depends(get_current_user)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
ClaimsDep = Annotated[TokenClaims, Depends(get_claims)]


def require_role(
    min_role: UserRole, *, resource: str
) -> Callable[..., AsyncIterator[User]]:
    """Dependency factory enforcing ``role >= min_role`` for *resource*.

    Usage::

        admin: Annotated[User, Depends(require_role(UserRole.ADMIN, resource="audit logs"))]

    The guard yields the same user object as :func:`get_current_user` (deps are
    cached per-request), so no extra database round-trip occurs.
    """

    async def _guard(user: CurrentUser) -> AsyncIterator[User]:
        try:
            check_role(user.role, min_role, resource=resource)
        except AuthorizationError as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
        yield user

    return _guard


def auth_error_to_http(exc: AuthError) -> HTTPException:
    """Map service-layer auth failures onto HTTP responses.

    Login deliberately returns the same 401 body for unknown tenant, unknown
    email and wrong password so probing cannot enumerate accounts.
    """
    if isinstance(exc, InvalidCredentialsError):
        return _unauthorized(str(exc))
    if isinstance(exc, UserNotFoundError):
        return _unauthorized(str(exc))
    if isinstance(exc, (InactiveTenantError, InactiveUserError)):
        return HTTPException(status.HTTP_403_FORBIDDEN, str(exc))
    if isinstance(exc, TenantAlreadyExistsError):
        return HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return HTTPException(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Authentication could not be completed",
    )


@dataclass(frozen=True, slots=True)
class PageParams:
    """Validated ``limit``/``offset`` pair for list endpoints."""

    limit: int
    offset: int


async def get_page_params(
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = DEFAULT_PAGE_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> PageParams:
    """Parse pagination query parameters within documented bounds."""
    return PageParams(limit=limit, offset=offset)


PageParamsDep = Annotated[PageParams, Depends(get_page_params)]


def client_audit_context(
    request: Request,
) -> tuple[str | None, str | None]:
    """Return ``(client_ip, user_agent)`` for audit rows."""
    client = request.client
    user_agent = request.headers.get("user-agent")
    return (client.host if client else None, user_agent)
