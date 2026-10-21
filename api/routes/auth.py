"""Auth routes: tenant signup, login, token refresh, current identity.

Signup creates the tenant and its first Admin in a single RLS-checked
transaction (client-generated tenant id, see ``services.auth.signup_tenant``);
login delegates credential verification to the SECURITY DEFINER lookup and
returns the same 401 body for every failure cause so accounts cannot be
enumerated.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from api.deps import CurrentUser, SessionDep, auth_error_to_http
from models import Tenant
from schemas.auth import (
    LoginRequest,
    LoginResponse,
    MeResponse,
    RefreshRequest,
    SignupResponse,
    TenantRead,
    TenantSignupRequest,
    TokenPair,
    UserRead,
)
from services.auth import AuthError, authenticate, issue_tokens, load_user, signup_tenant
from services.security import TokenValidationError, decode_token

router = APIRouter(prefix="/auth", tags=["auth"])

_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


@router.post(
    "/signup",
    response_model=SignupResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a tenant with its first Admin user",
)
async def signup(payload: TenantSignupRequest, session: SessionDep) -> SignupResponse:
    try:
        tenant, user = await signup_tenant(session, payload)
    except AuthError as exc:
        raise auth_error_to_http(exc) from exc
    await session.commit()  # persist before signing anything for the new identity
    return SignupResponse(
        tenant=TenantRead.model_validate(tenant),
        user=UserRead.model_validate(user),
        tokens=issue_tokens(user_id=user.id, tenant_id=user.tenant_id, role=user.role),
    )


@router.post("/login", response_model=LoginResponse, summary="Authenticate within a tenant")
async def login(payload: LoginRequest, session: SessionDep) -> LoginResponse:
    try:
        user = await authenticate(session, payload)
    except AuthError as exc:
        raise auth_error_to_http(exc) from exc
    tokens = issue_tokens(user_id=user.id, tenant_id=user.tenant_id, role=user.role)
    await session.commit()  # also persists users.last_login_at
    return LoginResponse(user=UserRead.model_validate(user), tokens=tokens)


@router.post("/refresh", response_model=TokenPair, summary="Exchange a refresh token for a new pair")
async def refresh(payload: RefreshRequest, session: SessionDep) -> TokenPair:
    try:
        claims = decode_token(payload.refresh_token, expected_type="refresh")
    except TokenValidationError as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Refresh token is invalid or has expired",
            headers=_UNAUTHORIZED_HEADERS,
        ) from exc
    try:
        # Role is re-read from the database, so refreshed tokens reflect any
        # privilege change made since the refresh token was issued.
        user = await load_user(session, claims)
    except AuthError as exc:
        raise auth_error_to_http(exc) from exc
    return issue_tokens(user_id=user.id, tenant_id=user.tenant_id, role=user.role)


@router.get("/me", response_model=MeResponse, summary="Current user and tenant")
async def me(user: CurrentUser, session: SessionDep) -> MeResponse:
    tenant = await session.get(Tenant, user.tenant_id)
    if tenant is None:  # pragma: no cover - RLS + GUC guarantee visibility
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Tenant not found")
    return MeResponse(
        user=UserRead.model_validate(user),
        tenant=TenantRead.model_validate(tenant),
    )
