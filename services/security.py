"""Password hashing (bcrypt) and JWT issuance/validation (python-jose).

Tokens carry the claims required for RBAC + tenancy (``tenant_id``, ``role``)
so authorization decisions can be made before any database round-trip, while
:class:`services.auth.load_user` always re-reads the role from the database so
revocations take effect immediately.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import bcrypt
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError

from config import get_settings
from models import UserRole

BCRYPT_ROUNDS: Final[int] = 12
BCRYPT_MAX_BYTES: Final[int] = 72  # bcrypt silently truncates beyond this
TOKEN_ISSUER: Final[str] = "rag-engine"
ACCESS_TOKEN_TYPE: Final[str] = "access"
REFRESH_TOKEN_TYPE: Final[str] = "refresh"

_REQUIRED_CLAIMS: Final[tuple[str, ...]] = (
    "sub",
    "tenant_id",
    "role",
    "type",
    "jti",
    "iat",
    "exp",
    "iss",
)


class PasswordValidationError(ValueError):
    """Raised when a password cannot be hashed under platform policy."""


class TokenValidationError(Exception):
    """Base class for token rejection reasons (maps to HTTP 401)."""


class TokenExpiredError(TokenValidationError):
    """The presented token is past its ``exp``."""


class InvalidTokenError(TokenValidationError):
    """The token is malformed, untrusted or carries unexpected claims."""


@dataclass(frozen=True, slots=True)
class TokenClaims:
    """Validated claims extracted from a verified JWT."""

    subject: uuid.UUID
    tenant_id: uuid.UUID
    role: UserRole
    jti: uuid.UUID
    token_type: str
    issued_at: datetime
    expires_at: datetime


def hash_password(password: str) -> str:
    """Hash a password with bcrypt (cost factor from :data:`BCRYPT_ROUNDS`).

    Raises:
        PasswordValidationError: When the password is empty or exceeds
            bcrypt's 72-byte limit (where it would be silently truncated).
    """
    raw = password.encode("utf-8")
    if not raw:
        raise PasswordValidationError("Password must not be empty")
    if len(raw) > BCRYPT_MAX_BYTES:
        raise PasswordValidationError(
            f"Password is {len(raw)} bytes; bcrypt only hashes the first "
            f"{BCRYPT_MAX_BYTES} bytes"
        )
    return bcrypt.hashpw(raw, bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("ascii")


def verify_password(password: str, hashed_password: str) -> bool:
    """Constant-time password check.

    Returns ``False`` for wrong passwords *and* malformed/legacy hashes rather
    than raising, so callers cannot accidentally distinguish the two.
    """
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            hashed_password.encode("ascii", errors="replace"),
        )
    except (ValueError, TypeError):
        return False


def _create_token(
    *,
    user_id: uuid.UUID,
    tenant_id: uuid.UUID,
    role: UserRole,
    token_type: str,
    lifetime: timedelta,
) -> tuple[str, int]:
    settings = get_settings()
    now = datetime.now(UTC)
    expires_at = now + lifetime
    claims = {
        "sub": str(user_id),
        "tenant_id": str(tenant_id),
        "role": role.value,
        "type": token_type,
        "jti": str(uuid.uuid4()),
        "iss": TOKEN_ISSUER,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    token = jwt.encode(
        claims,
        settings.secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    return token, int(lifetime.total_seconds())


def create_access_token(
    *, user_id: uuid.UUID, tenant_id: uuid.UUID, role: UserRole
) -> tuple[str, int]:
    """Issue an access token; returns ``(token, expires_in_seconds)``."""
    settings = get_settings()
    return _create_token(
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        token_type=ACCESS_TOKEN_TYPE,
        lifetime=timedelta(minutes=settings.access_token_expire_minutes),
    )


def create_refresh_token(
    *, user_id: uuid.UUID, tenant_id: uuid.UUID, role: UserRole
) -> tuple[str, int]:
    """Issue a refresh token; returns ``(token, expires_in_seconds)``."""
    settings = get_settings()
    return _create_token(
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        token_type=REFRESH_TOKEN_TYPE,
        lifetime=timedelta(days=settings.refresh_token_expire_days),
    )


def _as_uuid(claim: str, name: str) -> uuid.UUID:
    try:
        return uuid.UUID(claim)
    except (ValueError, AttributeError, TypeError) as exc:
        raise InvalidTokenError(f"Claim '{name}' is not a valid UUID") from exc


def decode_token(token: str, *, expected_type: str) -> TokenClaims:
    """Verify and decode a JWT, rejecting anything unexpected.

    Enforces signature, algorithm (``HS*`` only - ``alg=none`` is rejected),
    issuer, expiry and the full claim set before parsing UUIDs and the role.

    Raises:
        TokenExpiredError: The token has expired.
        InvalidTokenError: Any other validation failure.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            issuer=TOKEN_ISSUER,
            options={"require": list(_REQUIRED_CLAIMS)},
        )
    except ExpiredSignatureError as exc:
        raise TokenExpiredError("Token has expired") from exc
    except JWTError as exc:
        raise InvalidTokenError(f"Token rejected: {exc}") from exc

    # jose's `require` option is not uniform across versions/claims, so claim
    # presence is enforced explicitly before anything is indexed below.
    missing = [claim for claim in _REQUIRED_CLAIMS if claim not in payload]
    if missing:
        raise InvalidTokenError(
            "Token is missing required claims: " + ", ".join(sorted(missing))
        )

    if payload.get("type") != expected_type:
        raise InvalidTokenError(
            f"Expected a {expected_type!r} token, got {payload.get('type')!r}"
        )

    try:
        role = UserRole(payload["role"])
    except ValueError as exc:
        raise InvalidTokenError(f"Unknown role claim {payload.get('role')!r}") from exc

    try:
        issued_at = datetime.fromtimestamp(int(payload["iat"]), tz=UTC)
        expires_at = datetime.fromtimestamp(int(payload["exp"]), tz=UTC)
    except (ValueError, TypeError, OverflowError) as exc:
        raise InvalidTokenError("Token timestamps are invalid") from exc

    return TokenClaims(
        subject=_as_uuid(payload["sub"], "sub"),
        tenant_id=_as_uuid(payload["tenant_id"], "tenant_id"),
        role=role,
        jti=_as_uuid(payload["jti"], "jti"),
        token_type=str(payload["type"]),
        issued_at=issued_at,
        expires_at=expires_at,
    )
