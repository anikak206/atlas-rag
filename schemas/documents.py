"""Document ingestion and read contracts."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Final
from uuid import UUID

from pydantic import AliasPath, BaseModel, ConfigDict, Field, field_validator

from models import DocumentStatus, UserRole
from schemas.common import MAX_PAGE_LIMIT

#: Upper bound on document text accepted by the JSON ingest endpoint.
MAX_CONTENT_CHARS: Final[int] = 2_000_000
MAX_METADATA_BYTES: Final[int] = 65_536


class DocumentIngestRequest(BaseModel):
    """Text document submitted for chunking, embedding and indexing."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=1000)
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    source_uri: str | None = Field(default=None, max_length=2048)
    source_system: str | None = Field(default=None, max_length=64)
    mime_type: str = Field(default="text/plain", max_length=127)
    min_role_required: UserRole = Field(
        default=UserRole.INTERN,
        description="RBAC metadata: least privileged role allowed to retrieve this document",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form JSONB metadata (department, classification, tags)",
    )

    @field_validator("title")
    @classmethod
    def _trim_title(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Title must not be blank")
        return cleaned

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Keep metadata JSONB small and keyed by strings."""
        if not value:
            return value
        if any(not isinstance(key, str) for key in value):
            raise ValueError("metadata keys must be strings")
        try:
            encoded = json.dumps(value, default=str)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"metadata must be JSON-serialisable: {exc}") from exc
        if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
            raise ValueError(f"metadata must be at most {MAX_METADATA_BYTES} bytes")
        return value


class DocumentRead(BaseModel):
    """Document projection returned by list/detail endpoints."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    tenant_id: UUID
    title: str
    status: DocumentStatus
    min_role_required: UserRole
    source_uri: str | None = None
    source_system: str | None = None
    mime_type: str | None = None
    file_size_bytes: int | None = None
    chunk_count: int
    created_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    indexed_at: datetime | None = None
    deleted_at: datetime | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasPath("doc_metadata"),
    )


class IngestResponse(BaseModel):
    """Result of a successful ingestion."""

    document: DocumentRead
    chunks_created: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0.0)


class DeleteResponse(BaseModel):
    """Result of a soft delete."""

    id: UUID
    deleted: bool = True


class DocumentListParams(BaseModel):
    """Optional server-side filters for ``GET /documents``."""

    status: DocumentStatus | None = None
    min_role: UserRole | None = None
    limit: int = Field(default=20, ge=1, le=MAX_PAGE_LIMIT)
    offset: int = Field(default=0, ge=0)
