"""RAG query contracts and audit readback."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models import UserRole
from schemas.common import MAX_PAGE_LIMIT

MAX_QUESTION_CHARS: Final[int] = 4000
MAX_SNIPPET_CHARS: Final[int] = 480


class QueryRequest(BaseModel):
    """A natural-language question to answer from tenant knowledge."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Overrides RETRIEVAL_TOP_K for this request",
    )
    metadata_filters: dict[str, Any] | None = Field(
        default=None,
        description="JSONB containment filters applied to documents.metadata, e.g. "
        '{"department": "Finance"}',
    )
    include_sources: bool = Field(
        default=True,
        description="Return the supporting chunks alongside the answer",
    )

    @field_validator("question")
    @classmethod
    def _trim_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question must not be blank")
        return cleaned


class SourceChunk(BaseModel):
    """One retrieved chunk backing the generated answer."""

    chunk_id: UUID
    document_id: UUID
    document_title: str
    chunk_index: int
    score: float = Field(description="Cosine similarity in [0, 1]")
    distance: float = Field(description="Cosine distance in [0, 2]")
    snippet: str = Field(max_length=480)
    page_number: int | None = None
    heading_path: str | None = None
    min_role_required: UserRole


class TokenUsage(BaseModel):
    """Token accounting reported by the provider (or estimated)."""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated: bool = Field(
        default=False,
        description="True when the provider did not report usage and a ~4 chars/token "
        "estimate was used instead",
    )


class QueryResponse(BaseModel):
    """Answer plus provenance, cost and audit correlation."""

    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    applied_role: UserRole = Field(
        description="Caller's role, as used for the min_role_required filter",
    )
    top_k: int = Field(ge=1)
    retrieved_chunk_count: int = Field(ge=0)
    tokens: TokenUsage
    latency_ms: float = Field(ge=0.0)
    model: str | None = None
    request_id: UUID
    audit_id: int | None = Field(default=None, ge=0)
    generated_at: datetime


class AuditRead(BaseModel):
    """Audit row projection for ``GET /audit`` (Admin only)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    request_id: UUID
    tenant_id: UUID
    user_id: UUID | None = None
    actor_email: str | None = None
    actor_role: UserRole | None = None
    query_text: str
    response_time: datetime
    latency: float
    tokens_used: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model_name: str | None = None
    retrieved_chunk_count: int | None = None
    success: bool
    error_code: str | None = None
    created_at: datetime


class AuditListParams(BaseModel):
    """Filters accepted by ``GET /audit``."""

    user_id: UUID | None = None
    success: bool | None = None
    since: datetime | None = None
    limit: int = Field(default=50, ge=1, le=MAX_PAGE_LIMIT)
    offset: int = Field(default=0, ge=0)
