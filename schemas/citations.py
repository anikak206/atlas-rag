"""Citation and provenance schemas for Phase-3 enhanced RAG.

Provides structured citation payloads with document provenance,
page/section info, and confidence/similarity metrics.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from models import UserRole


class CitationSource(BaseModel):
    """Structured citation with full provenance information."""

    model_config = ConfigDict(from_attributes=True)

    source_document_id: UUID = Field(description="ID of the source document")
    document_title: str = Field(description="Title of the source document")
    chunk_id: UUID = Field(description="ID of the specific chunk cited")
    chunk_index: int = Field(description="Index of chunk within document")
    page_number: int | None = Field(default=None, description="Page number if available")
    section_title: str | None = Field(default=None, description="Section/heading path")
    heading_path: str | None = Field(default=None, description="Full heading breadcrumb")
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence/similarity score (1 - cosine distance)",
    )
    similarity_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Additional similarity metrics (cosine_similarity, distance, etc.)",
    )
    snippet: str = Field(max_length=1000, description="Relevant snippet from source")
    min_role_required: UserRole = Field(description="RBAC level required to access this source")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provenance metadata"
    )


class EnhancedQueryResponse(BaseModel):
    """Enhanced query response with citations and provenance."""

    answer: str = Field(description="Generated answer")
    citations: list[CitationSource] = Field(
        default_factory=list,
        description="Structured citations with full provenance",
    )
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Legacy sources format for backward compatibility",
    )
    applied_role: UserRole = Field(description="Caller's role used for filtering")
    top_k: int = Field(ge=1)
    retrieved_chunk_count: int = Field(ge=0)
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall answer confidence based on retrieval scores",
    )
    provenance_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of sources used"
    )
    tokens: dict[str, Any] = Field(description="Token usage information")
    latency_ms: float = Field(ge=0.0)
    model: str | None = None
    request_id: UUID
    audit_id: int | None = Field(default=None, ge=0)
    generated_at: Any
