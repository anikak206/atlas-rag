"""Tenant signup, login and identity loading - RLS-aware.

Tenancy discipline
------------------
Every query issued here runs in one of exactly two states:

1. **Pre-authentication** (login): the ``authenticate_user`` SECURITY DEFINER
   function (migration ``0002_auth_rls``) reads ``tenants``/``users`` with RLS
   bypassed on behalf of the unauthenticated caller - the only sanctioned
   bypass in the system.
2. **Post-authentication**: the tenant GUC (``app.tenant_id``) is bound via
   :func:`database.session_context` *before* the transaction begins, so RLS
   policies and ``WITH CHECK`` clauses evaluate against the right tenant for
   the whole transaction.

Password hashing/verification is dispatched to a thread because bcrypt costs
~200 ms per call at cost 12 - blocking the event loop would stall every
in-flight request.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import UTC, datetime
from typing import Final

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from database import session_context
from models import Tenant, TenantStatus, User, UserRole
from schemas.auth import LoginRequest, TenantSignupRequest, TokenPair
from services.security import (
    TokenClaims,
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)

_NON_WORD: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
_MAX_SLUG_LEN: Final[int] = 63

#: SECURITY DEFINER function created by migration 0002_auth_rls. Native enum
#: columns are cast to text: raw ``text()`` SQL gives asyncpg no Python type
#: for user-defined enum OIDs, while the ORM's own type map handles them.
AUTHENTICATE_SQL: Final[str] = """
    SELECT p_user_id,
           p_tenant_id,
           p_hashed_password,
           p_role::text AS p_role,
           p_is_active,
           p_tenant_status::text AS p_tenant_status
      FROM authenticate_user(:slug, :email)
"""

__all__ = [
    "AUTHENTICATE_SQL",
    "AuthError",
    "InactiveTenantError",
    "InactiveUserError",
    "InvalidCredentialsError",
    "TenantAlreadyExistsError",
    "UserNotFoundError",
    "authenticate",
    "issue_tokens",
    "load_user",
    "signup_tenant",
    "slugify",
]


class AuthError(Exception):
    """Base class for authentication/tenancy failures (routes map to HTTP)."""


class TenantAlreadyExistsError(AuthError):
    """The requested tenant slug is already registered."""


class InvalidCredentialsError(AuthError):
    """Login failed; deliberately indistinguishable between causes."""


class InactiveTenantError(AuthError):
    """The tenant is suspended or archived and may not authenticate."""


class InactiveUserError(AuthError):
    """The user exists but has been disabled."""


class UserNotFoundError(AuthError):
    """No user row matches the token claims (deleted or cross-tenant)."""


def slugify(name: str) -> str:
    """Derive a URL-safe tenant slug from a display name.

    Guarantees 1-63 lowercase alphanumerics with inner hyphens (the schema's
    slug contract); falls back to a random ``org-<hex>`` slug for names with no
    usable characters.
    """
    slug = _NON_WORD.sub("-", name.strip().lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)[:_MAX_SLUG_LEN].strip("-")
    return slug or f"org-{uuid.uuid4().hex[:8]}"


def _coerce_uuid(value: object, *, claim: str) -> uuid.UUID:
    """Accept ``uuid.UUID`` or its string form from a driver result row."""
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise AuthError(f"Database returned an invalid {claim}") from exc


async def signup_tenant(
    session: AsyncSession, payload: TenantSignupRequest
) -> tuple[Tenant, User]:
    """Create a tenant and its first Admin user in one RLS-checked transaction.

    The tenant id is generated client-side *first* so ``app.tenant_id`` can be
    bound before ``BEGIN`` - otherwise the ``tenants`` insert would trip its own
    ``WITH CHECK`` policy. Slug uniqueness is enforced by insert conflict rather
    than a pre-read (pre-auth reads of ``tenants`` are RLS-hidden by design):
    an explicit slug conflict raises :class:`TenantAlreadyExistsError`, while an
    auto-generated slug retries once with a random suffix.

    Raises:
        AuthError: If the session already has an open transaction (the GUC
            could not have been applied at ``BEGIN``).
        TenantAlreadyExistsError: The slug is taken.
    """
    if session.in_transaction():
        raise AuthError(
            "Session already has an open transaction, so app.tenant_id could not "
            "be bound at BEGIN; commit or roll back before signup."
        )

    explicit_slug = payload.tenant_slug
    base_slug = explicit_slug or slugify(payload.organization)
    candidates = (
        [base_slug]
        if explicit_slug
        else [base_slug, f"{base_slug[:48].strip('-')}-{uuid.uuid4().hex[:6]}"]
    )
    hashed_password = await asyncio.to_thread(hash_password, payload.password)

    last_conflict: IntegrityError | None = None
    for slug in candidates:
        tenant_id = uuid.uuid4()
        with session_context(tenant_id=tenant_id):
            tenant = Tenant(
                id=tenant_id,
                name=payload.organization,
                slug=slug,
                status=TenantStatus.ACTIVE,
            )
            user = User(
                tenant_id=tenant_id,
                email=payload.email,
                hashed_password=hashed_password,
                full_name=payload.full_name,
                role=UserRole.ADMIN,
            )
            session.add(tenant)
            session.add(user)
            try:
                await session.flush()
            except IntegrityError as exc:
                await session.rollback()
                last_conflict = exc
                continue
            return tenant, user

    raise TenantAlreadyExistsError(
        f"Tenant slug {candidates[-1]!r} is already registered; choose another."
    ) from last_conflict


async def authenticate(session: AsyncSession, payload: LoginRequest) -> User:
    """Verify tenant + email + password and return the authenticated user.

    Ordering hides information from unauthenticated callers: a wrong password
    yields the same error whether the tenant is suspended or the account is
    disabled; those specific errors only surface after the password verifies.

    On success the last transaction is closed and re-opened with the tenant GUC
    bound, so the ORM read and the caller's subsequent ``last_login_at`` update
    both run under RLS in one committed transaction.

    Raises:
        InvalidCredentialsError: Unknown tenant/email or wrong password.
        InactiveTenantError: Tenant suspended/archived.
        InactiveUserError: Account disabled.
    """
    row = (
        await session.execute(
            text(AUTHENTICATE_SQL),
            {"slug": payload.tenant_slug, "email": payload.email},
        )
    ).first()
    if row is None:
        raise InvalidCredentialsError("Invalid tenant, email or password")
    mapping = row._mapping

    if not await asyncio.to_thread(
        verify_password, payload.password, str(mapping["p_hashed_password"])
    ):
        raise InvalidCredentialsError("Invalid tenant, email or password")

    if TenantStatus(str(mapping["p_tenant_status"])) is not TenantStatus.ACTIVE:
        raise InactiveTenantError(
            "Tenant is not active; contact your administrator"
        )
    if not bool(mapping["p_is_active"]):
        raise InactiveUserError("Account is disabled; contact your administrator")

    tenant_id = _coerce_uuid(mapping["p_tenant_id"], claim="tenant_id")
    user_id = _coerce_uuid(mapping["p_user_id"], claim="user_id")

    # The credential lookup ran pre-authentication with no tenant GUC. Close
    # that transaction so the next BEGIN applies SET app.tenant_id and RLS
    # exposes exactly this tenant's rows for the ORM read below.
    await session.rollback()

    with session_context(tenant_id=tenant_id, actor_id=user_id):
        user = await session.get(User, user_id)
        if user is None or user.tenant_id != tenant_id:
            raise InvalidCredentialsError("Invalid tenant, email or password")
        user.last_login_at = datetime.now(UTC)
        return user


async def load_user(session: AsyncSession, claims: TokenClaims) -> User:
    """Load the active user behind verified token claims.

    The role is always read fresh from the database (never trusted from the
    token), so privilege revocation takes effect on the next request even
    while the JWT itself is still valid.

    Raises:
        UserNotFoundError: The user was deleted or the token's tenant does not
            match (defence in depth - signature already verified).
        InactiveUserError: The account has been disabled.
    """
    with session_context(tenant_id=claims.tenant_id, actor_id=claims.subject):
        user = await session.get(User, claims.subject)
    if user is None:
        raise UserNotFoundError("User no longer exists")
    if user.tenant_id != claims.tenant_id:
        raise UserNotFoundError("Token does not belong to this user")
    if not user.is_active:
        raise InactiveUserError("Account is disabled; contact your administrator")
    return user


def issue_tokens(*, user_id: uuid.UUID, tenant_id: uuid.UUID, role: UserRole) -> TokenPair:
    """Mint a fresh access/refresh pair for *user_id*.

    Refresh flows pass the role re-read from the database by :func:`load_user`
    so role changes are reflected immediately.
    """
    access_token, expires_in = create_access_token(
        user_id=user_id, tenant_id=tenant_id, role=role
    )
    refresh_token, _ = create_refresh_token(
        user_id=user_id, tenant_id=tenant_id, role=role
    )
    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
    )
