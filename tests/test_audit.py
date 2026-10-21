"""Audit row construction: clamping, truncation, privacy and correlation."""

from __future__ import annotations

import uuid
from datetime import UTC
from types import SimpleNamespace

import pytest

from config import get_settings
from models import UserRole
from services.audit import build_audit_log, coerce_request_id

TENANT_ID = uuid.uuid4()
USER = SimpleNamespace(
    id=uuid.uuid4(), email="finance@acme.test", role=UserRole.FINANCE
)


def test_builds_a_complete_success_row() -> None:
    log = build_audit_log(
        tenant_id=TENANT_ID,
        query_text="What is the leave policy?",
        latency_ms=42.5,
        prompt_tokens=120,
        completion_tokens=30,
        user=USER,
        model_name="gpt-4o-mini",
        retrieved_chunk_count=3,
        filters={"allowed_roles": ["intern", "finance"], "top_k": 8},
        ip_address="203.0.113.10",
        user_agent="pytest/1.0",
    )
    assert log.tenant_id == TENANT_ID
    assert log.user_id == USER.id
    assert log.actor_email == USER.email
    assert log.actor_role is UserRole.FINANCE
    assert log.query_text == "What is the leave policy?"
    assert log.latency == pytest.approx(42.5)
    assert log.prompt_tokens == 120
    assert log.completion_tokens == 30
    assert log.tokens_used == 150
    assert log.success is True
    assert log.response_time.tzinfo is not None
    assert log.response_time.tzinfo is not None
    assert log.filters == {"allowed_roles": ["intern", "finance"], "top_k": 8}


def test_anonymous_row_has_no_actor() -> None:
    log = build_audit_log(
        tenant_id=TENANT_ID, query_text="q", latency_ms=1.0, success=False,
        error_code="rag_pipeline_failed", error_message="provider down",
    )
    assert log.user_id is None
    assert log.actor_email is None
    assert log.actor_role is None
    assert log.success is False
    assert log.error_code == "rag_pipeline_failed"


def test_negative_metrics_are_clamped_for_check_constraints() -> None:
    log = build_audit_log(
        tenant_id=TENANT_ID,
        query_text="q",
        latency_ms=-5.0,
        prompt_tokens=-10,
        completion_tokens=-4,
    )
    assert log.latency == 0.0
    assert log.prompt_tokens == 0
    assert log.completion_tokens == 0
    assert log.tokens_used == 0


def test_error_message_and_user_agent_are_truncated() -> None:
    log = build_audit_log(
        tenant_id=TENANT_ID,
        query_text="q",
        latency_ms=1.0,
        success=False,
        error_message="e" * 5000,
        user_agent="u" * 900,
    )
    assert len(log.error_message or "") == 2000
    assert len(log.user_agent or "") == 512


def test_response_text_is_not_stored_by_default() -> None:
    assert get_settings().audit_store_response_snippet is False
    log = build_audit_log(
        tenant_id=TENANT_ID,
        query_text="q",
        latency_ms=1.0,
        response_text="confidential answer",
    )
    assert log.response_text is None


def test_response_snippet_policy_truncates_when_enabled(
    settings_env: object,
) -> None:
    settings_env(  # type: ignore[operator]
        AUDIT_STORE_RESPONSE_SNIPPET="true",
        AUDIT_RESPONSE_SNIPPET_CHARS="10",
    )
    log = build_audit_log(
        tenant_id=TENANT_ID,
        query_text="q",
        latency_ms=1.0,
        response_text="abcdefghijklmno",
    )
    assert log.response_text == "abcdefghi\u2026"
    assert len(log.response_text or "") == 10


def test_response_snippet_shorter_than_limit_is_kept(
    settings_env: object,
) -> None:
    settings_env(AUDIT_STORE_RESPONSE_SNIPPET="true")  # type: ignore[operator]
    log = build_audit_log(
        tenant_id=TENANT_ID,
        query_text="q",
        latency_ms=1.0,
        response_text="short answer",
    )
    assert log.response_text == "short answer"


@pytest.mark.parametrize("value", [None, "", "not-a-uuid", "12345"])
def test_request_id_is_coerced_to_a_uuid(value: str | None) -> None:
    assert isinstance(coerce_request_id(value), uuid.UUID)


def test_request_id_uuid_is_preserved() -> None:
    correlation_id = uuid.uuid4()
    assert coerce_request_id(correlation_id) == correlation_id
    assert coerce_request_id(str(correlation_id)) == correlation_id


def test_explicit_response_time_is_respected() -> None:
    from datetime import datetime

    moment = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    log = build_audit_log(
        tenant_id=TENANT_ID,
        query_text="q",
        latency_ms=1.0,
        response_time=moment,
    )
    assert log.response_time == moment
