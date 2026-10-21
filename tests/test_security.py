"""Password hashing and JWT validation security properties."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from jose import jwt

from config import get_settings
from models import UserRole
from services.security import (
    ACCESS_TOKEN_TYPE,
    BCRYPT_MAX_BYTES,
    TOKEN_ISSUER,
    InvalidTokenError,
    PasswordValidationError,
    TokenExpiredError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

VALID_PASSWORD = "Sunrise-Cobalt-2026"


def _claims(**overrides: Any) -> dict[str, Any]:
    """A fully valid access-token claim set, with optional overrides."""
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "sub": str(uuid.uuid4()),
        "tenant_id": str(uuid.uuid4()),
        "role": UserRole.INTERN.value,
        "type": ACCESS_TOKEN_TYPE,
        "jti": str(uuid.uuid4()),
        "iss": TOKEN_ISSUER,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
    }
    claims.update(overrides)
    return claims


def _encode(claims: dict[str, Any]) -> str:
    settings = get_settings()
    token: str = jwt.encode(
        claims,
        settings.secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    return token


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------
def test_hash_is_bcrypt_with_configured_cost_and_salted() -> None:
    first = hash_password(VALID_PASSWORD)
    second = hash_password(VALID_PASSWORD)
    assert first.startswith("$2b$12$")
    assert first != second  # unique salt per hash
    assert verify_password(VALID_PASSWORD, first) is True
    assert verify_password(VALID_PASSWORD, second) is True


def test_wrong_password_is_rejected() -> None:
    hashed = hash_password(VALID_PASSWORD)
    assert verify_password("Sunrise-Cobalt-2025", hashed) is False


@pytest.mark.parametrize("password", ["", "a" * (BCRYPT_MAX_BYTES + 1)])
def test_hash_rejects_empty_and_over_limit_passwords(password: str) -> None:
    with pytest.raises(PasswordValidationError):
        hash_password(password)


def test_verify_treats_malformed_hash_as_failure() -> None:
    assert verify_password(VALID_PASSWORD, "not-a-bcrypt-hash") is False
    assert verify_password(VALID_PASSWORD, "") is False


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
def test_access_token_roundtrip_preserves_identity() -> None:
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    token, expires_in = create_access_token(
        user_id=user_id, tenant_id=tenant_id, role=UserRole.FINANCE
    )
    claims = decode_token(token, expected_type=ACCESS_TOKEN_TYPE)
    assert claims.subject == user_id
    assert claims.tenant_id == tenant_id
    assert claims.role is UserRole.FINANCE
    assert claims.token_type == ACCESS_TOKEN_TYPE
    assert claims.expires_at > claims.issued_at
    assert expires_in > 0


def test_refresh_token_roundtrip_and_type_separation() -> None:
    token, _ = create_refresh_token(
        user_id=uuid.uuid4(), tenant_id=uuid.uuid4(), role=UserRole.ADMIN
    )
    claims = decode_token(token, expected_type="refresh")
    assert claims.token_type == "refresh"
    with pytest.raises(InvalidTokenError, match="Expected"):
        decode_token(token, expected_type=ACCESS_TOKEN_TYPE)


def test_expired_token_raises_specific_error() -> None:
    now = datetime.now(UTC)
    token = _encode(
        _claims(
            iat=int((now - timedelta(minutes=10)).timestamp()),
            exp=int((now - timedelta(minutes=1)).timestamp()),
        )
    )
    with pytest.raises(TokenExpiredError):
        decode_token(token, expected_type=ACCESS_TOKEN_TYPE)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"iss": "another-issuer"}, "rejected"),
        ({"role": "Wizard"}, "Unknown role"),
        ({"sub": "not-a-uuid"}, "not a valid UUID"),
        ({"tenant_id": "not-a-uuid"}, "not a valid UUID"),
    ],
)
def test_invalid_claims_are_rejected(
    overrides: dict[str, Any], match: str
) -> None:
    token = _encode(_claims(**overrides))
    with pytest.raises(InvalidTokenError, match=match):
        decode_token(token, expected_type=ACCESS_TOKEN_TYPE)


def test_missing_required_claim_is_rejected() -> None:
    claims = _claims()
    del claims["jti"]
    with pytest.raises(InvalidTokenError):
        decode_token(_encode(claims), expected_type=ACCESS_TOKEN_TYPE)


def test_tampered_signature_is_rejected() -> None:
    token, _ = create_access_token(
        user_id=uuid.uuid4(), tenant_id=uuid.uuid4(), role=UserRole.INTERN
    )
    replacement = "A" if token[-1] != "A" else "B"
    with pytest.raises(InvalidTokenError):
        decode_token(token[:-1] + replacement, expected_type=ACCESS_TOKEN_TYPE)


def test_garbage_token_is_rejected() -> None:
    with pytest.raises(InvalidTokenError):
        decode_token("clearly-not-a-jwt", expected_type=ACCESS_TOKEN_TYPE)
