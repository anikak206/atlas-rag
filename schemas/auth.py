"""Auth contracts: tenant signup, login, token refresh and identity reads."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models import TenantStatus, UserRole
from services.security import BCRYPT_MAX_BYTES

_SLUG_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

#: Passwords rejected even when they satisfy the structural policy.
WEAK_PASSWORDS: Final[frozenset[str]] = frozenset(
    {
        "password1234",
        "passw0rd1234",
        "administrator",
        "letmein12345",
        "qwertyuiop12",
        "welcome12345",
        "changeme1234",
        "password12345",
    }
)

SLUG_PATTERN_DESCRIPTION = "Lowercase letters, digits and inner hyphens (max 63 chars)"


def enforce_password_policy(value: str) -> str:
    """Validate a candidate password against the platform policy.

    Policy: >= 12 characters, at least one letter and one digit, not a known
    weak password, <= 128 characters **and** <= 72 UTF-8 bytes (bcrypt hashes
    only the first 72 bytes, so longer values would be silently truncated).

    Raises:
        ValueError: When the password does not satisfy the policy.
    """
    password = value.strip()
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters long")
    if len(password) > 128:
        raise ValueError("Password must be at most 128 characters long")
    if len(password.encode("utf-8")) > BCRYPT_MAX_BYTES:
        raise ValueError(
            f"Password must be at most {BCRYPT_MAX_BYTES} UTF-8 bytes "
            "(bcrypt hashing limit; prefer shorter passwords)"
        )
    if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
        raise ValueError("Password must contain at least one letter and one digit")
    if password.lower() in WEAK_PASSWORDS:
        raise ValueError("Password is too common; choose a less predictable one")
    if password.isspace():
        raise ValueError("Password must not be blank")
    return password


class _EmailNormalised(BaseModel):
    """Mixin-style base applying normalisation to ``email`` fields."""

    @field_validator("email", check_fields=False)
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        normalised = value.strip().lower()
        if not _EMAIL_PATTERN.match(normalised):
            raise ValueError("Not a valid email address")
        return normalised


class TenantSignupRequest(_EmailNormalised):
    """Create a new tenant together with its first Admin user."""

    model_config = ConfigDict(extra="forbid")

    organization: str = Field(min_length=2, max_length=120, description="Tenant display name")
    tenant_slug: str | None = Field(
        default=None,
        pattern=_SLUG_PATTERN,
        description=f"Optional URL slug; auto-generated from the name when omitted. {SLUG_PATTERN_DESCRIPTION}",
    )
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=12, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)

    @field_validator("password")
    @classmethod
    def _check_password(cls, value: str) -> str:
        return enforce_password_policy(value)

    @field_validator("organization")
    @classmethod
    def _trim_organisation(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Organization name must not be blank")
        return cleaned


class LoginRequest(_EmailNormalised):
    """Authenticate within a specific tenant (tenants are isolated)."""

    model_config = ConfigDict(extra="forbid")

    tenant_slug: str = Field(min_length=1, max_length=63, pattern=_SLUG_PATTERN)
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("tenant_slug", mode="before")
    @classmethod
    def _normalise_slug(cls, value: Any) -> Any:
        # Runs before the pattern constraint so pasted values ("  ACME ") are
        # trimmed and lowercased instead of failing validation outright.
        return value.strip().lower() if isinstance(value, str) else value


class RefreshRequest(BaseModel):
    """Exchange a refresh token for a fresh token pair."""

    model_config = ConfigDict(extra="forbid")

    refresh_token: str = Field(min_length=20, max_length=4096)


class TokenPair(BaseModel):
    """Issued JWT pair."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(ge=0, description="Access token lifetime in seconds")


class TenantRead(BaseModel):
    """Tenant projection returned to authenticated members of that tenant."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    slug: str
    status: TenantStatus
    created_at: datetime


class UserRead(BaseModel):
    """Authenticated user projection (never exposes password material)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    tenant_id: UUID
    email: str
    full_name: str | None = None
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None = None


class SignupResponse(BaseModel):
    """Result of tenant signup: identity plus immediately usable tokens."""

    tenant: TenantRead
    user: UserRead
    tokens: TokenPair


class LoginResponse(BaseModel):
    """Result of a successful login."""

    user: UserRead
    tokens: TokenPair


class MeResponse(BaseModel):
    """Current identity for ``GET /auth/me``."""

    user: UserRead
    tenant: TenantRead
