"""Request/response contract validation (policies, aliases, pagination)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from models import DocumentStatus, TenantStatus, UserRole
from schemas.auth import (
    LoginRequest,
    TenantSignupRequest,
    UserRead,
    enforce_password_policy,
)
from schemas.common import Page
from schemas.documents import (
    MAX_METADATA_BYTES,
    DocumentIngestRequest,
    DocumentRead,
)
from schemas.query import (
    MAX_QUESTION_CHARS,
    MAX_SNIPPET_CHARS,
    QueryRequest,
    SourceChunk,
    TokenUsage,
)

VALID_PASSWORD = "Sunrise-Cobalt-2026"


def _signup(**overrides: object) -> TenantSignupRequest:
    payload: dict[str, object] = {
        "organization": "Acme Corporation",
        "email": "admin@acme.test",
        "password": VALID_PASSWORD,
    }
    payload.update(overrides)
    return TenantSignupRequest(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Password policy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "password",
    ["Shorty-1", "NoDigitsAllowed-Here", "password1234", "   ", "a" * 200],
)
def test_password_policy_rejects_weak_candidates(password: str) -> None:
    with pytest.raises(ValueError):
        enforce_password_policy(password)


def test_password_policy_rejects_more_than_bcrypt_byte_limit() -> None:
    # 40 chars, 80 UTF-8 bytes: structurally fine but unhashable by bcrypt.
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        enforce_password_policy("é" * 40)


def test_password_policy_accepts_exactly_72_bytes() -> None:
    candidate = "A1" + "é" * 35  # 2 + 70 = 72 bytes
    assert enforce_password_policy(candidate) == candidate


def test_password_policy_accepts_strong_password() -> None:
    assert enforce_password_policy(VALID_PASSWORD) == VALID_PASSWORD


# ---------------------------------------------------------------------------
# Signup / login normalisation
# ---------------------------------------------------------------------------
def test_signup_normalises_email_and_organisation() -> None:
    request = _signup(organization="  Acme   Corporation ", email=" Admin@ACME.test ")
    assert request.email == "admin@acme.test"
    assert request.organization == "Acme Corporation"
    assert request.tenant_slug is None


@pytest.mark.parametrize("slug", ["Acme", "acme_corp", "-acme", "acme-", "a" * 64])
def test_signup_rejects_invalid_slugs(slug: str) -> None:
    with pytest.raises(ValidationError):
        _signup(tenant_slug=slug)


def test_signup_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _signup(is_superuser=True)


def test_login_normalises_slug_and_email() -> None:
    request = LoginRequest(
        tenant_slug="  ACME ", email="FINANCE@Acme.test ", password="whatever"
    )
    assert request.tenant_slug == "acme"
    assert request.email == "finance@acme.test"


def test_user_read_never_contains_password_material() -> None:
    assert "hashed_password" not in UserRead.model_fields


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def test_ingest_defaults_to_intern_visibility() -> None:
    request = DocumentIngestRequest(title="Handbook", content="Body text")
    assert request.min_role_required is UserRole.INTERN
    assert request.metadata == {}
    assert request.mime_type == "text/plain"


def test_ingest_trims_title_and_rejects_blank() -> None:
    assert DocumentIngestRequest(title="  Handbook  ", content="x").title == "Handbook"
    with pytest.raises(ValidationError):
        DocumentIngestRequest(title="   ", content="x")


def test_ingest_rejects_oversized_metadata() -> None:
    oversized = {"blob": "x" * (MAX_METADATA_BYTES + 1)}
    with pytest.raises(ValidationError, match="bytes"):
        DocumentIngestRequest(title="T", content="x", metadata=oversized)


def test_document_read_maps_orm_metadata_column() -> None:
    now = datetime.now(UTC)
    row = SimpleNamespace(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        title="Budget",
        status=DocumentStatus.INDEXED,
        min_role_required=UserRole.FINANCE,
        source_uri=None,
        source_system=None,
        mime_type="text/plain",
        file_size_bytes=10,
        chunk_count=2,
        created_by=None,
        created_at=now,
        updated_at=now,
        indexed_at=now,
        deleted_at=None,
        doc_metadata={"department": "Finance"},
    )
    document = DocumentRead.model_validate(row)
    assert document.metadata == {"department": "Finance"}


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def test_query_trims_question_and_rejects_blank() -> None:
    assert QueryRequest(question="  What is the leave policy?  ").question == (
        "What is the leave policy?"
    )
    with pytest.raises(ValidationError):
        QueryRequest(question="   ")


def test_query_rejects_overlong_question() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="x" * (MAX_QUESTION_CHARS + 1))


@pytest.mark.parametrize("top_k", [0, 51])
def test_query_top_k_bounds(top_k: int) -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="Q", top_k=top_k)


def test_query_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="Q", tenant_id=str(uuid.uuid4()))


def test_source_snippet_length_is_capped() -> None:
    base = {
        "chunk_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "document_title": "Doc",
        "chunk_index": 0,
        "score": 0.9,
        "distance": 0.1,
        "min_role_required": UserRole.INTERN,
    }
    assert len(SourceChunk(**base, snippet="x" * MAX_SNIPPET_CHARS).snippet) == (
        MAX_SNIPPET_CHARS
    )
    with pytest.raises(ValidationError):
        SourceChunk(**base, snippet="x" * (MAX_SNIPPET_CHARS + 1))


def test_token_usage_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        TokenUsage(prompt_tokens=-1, completion_tokens=0, total_tokens=0)


# ---------------------------------------------------------------------------
# Pagination envelope
# ---------------------------------------------------------------------------
def test_page_has_more_is_derived_from_offset_and_total() -> None:
    assert Page.build(["a", "b"], total=3, limit=2, offset=0).has_more is True
    assert Page.build(["a", "b"], total=2, limit=2, offset=0).has_more is False
    assert Page.build(["c"], total=3, limit=2, offset=2).has_more is False


def test_page_rejects_limit_above_ceiling() -> None:
    with pytest.raises(ValidationError):
        Page.build([], total=0, limit=101, offset=0)


def test_tenant_status_enum_roundtrip() -> None:
    assert TenantStatus("active") is TenantStatus.ACTIVE
