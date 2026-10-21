"""Pydantic request/response contracts for the RAG API."""

from __future__ import annotations

from schemas.auth import (
    LoginRequest,
    LoginResponse,
    MeResponse,
    RefreshRequest,
    SignupResponse,
    TenantRead,
    TokenPair,
    UserRead,
)
from schemas.common import ErrorResponse, Page
from schemas.documents import (
    DeleteResponse,
    DocumentIngestRequest,
    DocumentRead,
    IngestResponse,
)
from schemas.query import (
    AuditRead,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    TokenUsage,
)

__all__ = [
    "AuditRead",
    "DeleteResponse",
    "DocumentIngestRequest",
    "DocumentRead",
    "ErrorResponse",
    "IngestResponse",
    "LoginRequest",
    "LoginResponse",
    "MeResponse",
    "Page",
    "QueryRequest",
    "QueryResponse",
    "RefreshRequest",
    "SignupResponse",
    "SourceChunk",
    "TenantRead",
    "TokenPair",
    "TokenUsage",
    "UserRead",
]
