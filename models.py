"""SQLAlchemy 2.0 models: Tenants, Users, Documents, DocumentChunks, AuditLogs.

Schema decisions
----------------
* **Tenancy** - every tenant-scoped table carries a non-nullable ``tenant_id``
  FK to ``tenants`` (ON DELETE CASCADE) so isolation is enforced by the
  database and by the RLS policies from the baseline migration.
* **RBAC metadata** - ``documents.min_role_required`` stores the least
  privileged role (``Intern`` < ``Finance`` < ``Admin``) allowed to retrieve a
  document; queries must filter on it in addition to RLS.
* **Vectors** - ``document_chunks.embedding`` is a pgvector ``Vector`` column
  dimensioned by ``EMBEDDING_DIMENSION`` and indexed with HNSW using the
  operator class matching ``DISTANCE_STRATEGY``.
* **Audit** - ``audit_logs`` is an append-only ledger (no ORM relationships, no
  cascading FKs) so records outlive the users/tenants they reference.
* **Loading policy** - relationships default to ``lazy="raise"`` (or
  ``selectin`` for the small ``tenant.users`` collection) so request handlers
  must state their graph explicitly (``selectinload``/``lazyload``); accidental
  N+1 queries become test failures instead of production latency.

Requires PostgreSQL 14+ (``gen_random_uuid()`` built in) and pgvector >= 0.5
(HNSW indexes).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Final

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy import (
    Enum as SQLAlchemyEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from config import get_settings
from database import Base

_settings = get_settings()

#: ``DISTANCE_STRATEGY`` -> pgvector operator class used by the ANN index.
_DISTANCE_OPS: Final[dict[str, str]] = {
    "cosine": "vector_cosine_ops",
    "l2": "vector_l2_ops",
    "inner_product": "vector_ip_ops",
}
VECTOR_DISTANCE_OPS: Final[str] = _DISTANCE_OPS[_settings.distance_strategy]
EMBEDDING_DIMENSION: Final[int] = _settings.embedding_dimension


class UserRole(StrEnum):
    """RBAC roles, ordered least to most privileged."""

    INTERN = "Intern"
    FINANCE = "Finance"
    ADMIN = "Admin"


class TenantStatus(StrEnum):
    """Lifecycle state of a tenant."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


class DocumentStatus(StrEnum):
    """Ingestion/indexing lifecycle of a document."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    ARCHIVED = "archived"


def _pg_enum(enum_cls: type[StrEnum], *, name: str) -> SQLAlchemyEnum:
    """Build a native PostgreSQL enum bound to a Python :class:`StrEnum`.

    ``values_callable`` makes the database store the member *values*
    (``'Admin'``) rather than the member *names* (``'ADMIN'``).
    """
    return SQLAlchemyEnum(
        enum_cls,
        name=name,
        native_enum=True,
        validate_strings=True,
        values_callable=lambda members: [member.value for member in members],
    )


#: Single enum instances reused across tables so ``create_all``/``drop_all``
#: emit exactly one ``CREATE TYPE``/``DROP TYPE`` per name.
USER_ROLE_ENUM: Final[SQLAlchemyEnum] = _pg_enum(UserRole, name="user_role")
TENANT_STATUS_ENUM: Final[SQLAlchemyEnum] = _pg_enum(TenantStatus, name="tenant_status")
DOCUMENT_STATUS_ENUM: Final[SQLAlchemyEnum] = _pg_enum(DocumentStatus, name="document_status")


class TimestampMixin:
    """``created_at``/``updated_at`` maintained by the ORM *and* by a
    ``BEFORE UPDATE`` trigger (see the baseline migration), so bulk/raw SQL
    updates keep ``updated_at`` accurate too."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="UTC row creation time",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="UTC row last modification time",
    )


class Tenant(Base, TimestampMixin):
    """Enterprise customer; the root isolation boundary for all data."""

    __tablename__ = "tenants"
    __table_args__ = (
        Index("ix_tenants_status", "status"),
        {
            "comment": "Tenant isolation boundary. Every tenant-scoped row references tenants.id."
        },
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, comment="Display name")
    slug: Mapped[str] = mapped_column(
        String(63),
        nullable=False,
        unique=True,
        comment="URL/subject-safe identifier, e.g. 'acme-corp'",
    )
    status: Mapped[TenantStatus] = mapped_column(
        TENANT_STATUS_ENUM,
        nullable=False,
        server_default=text("'active'"),
        comment="active | suspended | archived",
    )
    settings: Mapped[dict[str, Any]] = mapped_column(
        "settings",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        comment="Tenant-scoped feature flags and RAG preferences",
    )

    users: Mapped[list[User]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )
    documents: Mapped[list[Document]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )

    def __repr__(self) -> str:
        return f"<Tenant {self.slug!r} status={self.status.value if self.status else None}>"


class User(Base, TimestampMixin):
    """Authenticatable member of a tenant, holding exactly one RBAC role."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "email",
            name="uq_users_tenant_id_email",
        ),
        CheckConstraint("length(email) > 0", name="ck_users_email_non_empty"),
        Index("ix_users_tenant_id_role", "tenant_id", "role"),
        {"comment": "Tenant member with an RBAC role (Admin | Finance | Intern)."},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment=" Owning tenant",
    )
    email: Mapped[str] = mapped_column(
        String(320),
        nullable=False,
        comment="Lowercased address; unique together with tenant_id",
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="bcrypt hash; never logged",
    )
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[UserRole] = mapped_column(
        USER_ROLE_ENUM,
        nullable=False,
        server_default=text("'Intern'"),
        comment="RBAC role used for document access decisions",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
        comment="Disabled users are rejected at token validation",
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
        comment="Platform operator; bypasses tenant-scoped UI restrictions only",
    )
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    tenant: Mapped[Tenant] = relationship(back_populates="users")
    documents_created: Mapped[list[Document]] = relationship(
        back_populates="creator",
        foreign_keys="[Document.created_by]",
        lazy="raise",
        passive_deletes=True,
    )

    @validates("email")
    def _normalise_email(self, key: str, value: str) -> str:
        """Lowercase/trim and reject obviously malformed addresses early."""
        if not isinstance(value, str):
            raise TypeError(f"{key} must be a string")
        normalised = value.strip().lower()
        if len(normalised) < 3 or normalised.count("@") != 1 or normalised.startswith("@"):
            raise ValueError(f"{key} {value!r} is not a valid email address")
        return normalised

    def __repr__(self) -> str:
        return f"<User {self.email!r} role={self.role.value if self.role else None}>"


class Document(Base, TimestampMixin):
    """Tenant-scoped source document plus its RBAC and ingestion metadata."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "content_hash",
            name="uq_documents_tenant_id_content_hash",
        ),
        CheckConstraint("chunk_count >= 0", name="ck_documents_chunk_count_non_negative"),
        Index("ix_documents_tenant_id_status", "tenant_id", "status"),
        Index("ix_documents_tenant_id_min_role_required", "tenant_id", "min_role_required"),
        Index(
            "ix_documents_doc_metadata_gin",
            "metadata",
            postgresql_using="gin",
            postgresql_ops={"metadata": "jsonb_path_ops"},
        ),
        {"comment": "Source documents; min_role_required drives RBAC filtering at query time."},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Owning tenant",
    )
    title: Mapped[str] = mapped_column(String(1000), nullable=False, comment="Human-readable title")
    source_uri: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Original location (S3 key, URL, SharePoint path)",
    )
    source_system: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="Connector that produced the document, e.g. 'confluence'",
    )
    mime_type: Mapped[str | None] = mapped_column(String(127), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="SHA-256 hex digest of raw content; deduplicated per tenant",
    )
    status: Mapped[DocumentStatus] = mapped_column(
        DOCUMENT_STATUS_ENUM,
        nullable=False,
        server_default=text("'pending'"),
        comment="pending | processing | indexed | failed | archived",
    )
    min_role_required: Mapped[UserRole] = mapped_column(
        USER_ROLE_ENUM,
        nullable=False,
        server_default=text("'Intern'"),
        comment="RBAC metadata: least privileged role allowed to retrieve this document",
    )
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        comment="Free-form metadata (department, classification, language, tags)",
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        comment="Number of rows in document_chunks",
    )
    indexed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Set once embedding/indexing completed",
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Soft delete marker; rows are purged by the retention job",
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="User who ingested the document (kept as a value, not a join)",
    )

    tenant: Mapped[Tenant] = relationship(back_populates="documents")
    creator: Mapped[User | None] = relationship(
        back_populates="documents_created",
        foreign_keys=[created_by],
        lazy="raise",
    )
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )

    def __repr__(self) -> str:
        status = self.status.value if self.status else None
        return f"<Document {self.title!r} status={status} min_role={self.min_role_required}>"


class DocumentChunk(Base, TimestampMixin):
    """Text chunk with its pgvector embedding; the unit of retrieval.

    Phase-3 enhancement: RBAC metadata is embedded directly into chunk metadata
    for metadata pre-filtering so access control is enforced at the vector
    query level in addition to SQL filters.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_id_chunk_index",
        ),
        CheckConstraint("chunk_index >= 0", name="ck_document_chunks_chunk_index_non_negative"),
        CheckConstraint("token_count >= 0", name="ck_document_chunks_token_count_non_negative"),
        Index("ix_document_chunks_tenant_id_document_id", "tenant_id", "document_id"),
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": VECTOR_DISTANCE_OPS},
            postgresql_with={
                "m": _settings.hnsw_m,
                "ef_construction": _settings.hnsw_ef_construction,
            },
        ),
        # GIN index for metadata pre-filtering (RBAC + document attributes)
        Index(
            "ix_document_chunks_metadata_gin",
            "metadata",
            postgresql_using="gin",
            postgresql_ops={"metadata": "jsonb_path_ops"},
        ),
        {
            "comment": "Chunk store: content + pgvector embedding, isolated by tenant and document with RBAC metadata."
        },
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        comment="Denormalised tenant id: matches the parent document, used by RLS",
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="Parent document; deleting it deletes its chunks",
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="0-based position of the chunk within the document",
    )
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="Raw chunk text")
    embedding: Mapped[Any] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=False,
        comment=(
            f"pgvector vector({EMBEDDING_DIMENSION}) produced by "
            f"{_settings.embedding_model}; changing dimension requires a migration"
        ),
    )
    embedding_model: Mapped[str] = mapped_column(
        String(127),
        nullable=False,
        default=lambda: _settings.embedding_model,
        comment="Model that produced `embedding`; enables selective re-indexing",
    )
    token_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        comment="Tokenizer count for cost/quota accounting",
    )
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    heading_path: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="Breadcrumb of headings, e.g. 'Q3 Report > Risks'",
    )
    chunk_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
        comment="LlamaIndex node metadata with RBAC (allowed_roles, tenant_id, document_id, owner, min_role_required)",
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return f"<DocumentChunk doc={self.document_id} idx={self.chunk_index}>"


class AuditLog(Base):
    """Append-only record of every RAG query answered by the platform.

    Intentionally has **no** FK relationships: audit records must survive the
    deletion/archival of the user or tenant they reference (compliance), and
    the table is a candidate for monthly RANGE partitioning on ``created_at``.
    """

    __tablename__ = "audit_logs"
    __table_args__ = (
        CheckConstraint("latency >= 0", name="ck_audit_logs_latency_non_negative"),
        CheckConstraint("tokens_used >= 0", name="ck_audit_logs_tokens_used_non_negative"),
        Index("ix_audit_logs_tenant_id_created_at", "tenant_id", "created_at"),
        Index("ix_audit_logs_user_id_created_at", "user_id", "created_at"),
        Index("ix_audit_logs_request_id", "request_id"),
        Index("ix_audit_logs_created_at_brin", "created_at", postgresql_using="brin"),
        {"comment": "Append-only query audit trail; never UPDATEd or DELETEd by the application."},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        comment="Monotonic surrogate key",
    )
    request_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
        comment="Correlation id shared with API logs (X-Request-ID)",
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
        comment="Tenant the query ran in (denormalised; no FK so records outlive tenants)",
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
        comment="Actor id at query time; nullable so history survives user deletion",
    )
    actor_email: Mapped[str | None] = mapped_column(
        String(320),
        nullable=True,
        comment="Snapshot of the actor's email at query time",
    )
    actor_role: Mapped[UserRole | None] = mapped_column(
        USER_ROLE_ENUM,
        nullable=True,
        comment="Snapshot of the actor's RBAC role at query time",
    )
    query_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="User question / prompt submitted to the RAG engine",
    )
    response_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="UTC timestamp at which the response was completed (wall clock)",
    )
    latency: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default=text("0"),
        comment="End-to-end request latency in milliseconds (retrieval + generation)",
    )
    tokens_used: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        comment="Total tokens consumed (prompt + completion)",
    )
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str | None] = mapped_column(
        String(127),
        nullable=True,
        comment="LLM used for generation, e.g. 'gpt-4o-mini'",
    )
    retrieved_chunk_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of chunks returned by vector search before RBAC filtering",
    )
    success: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
        comment="False when the request ended in an error",
    )
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Truncated/redacted answer snippet; only stored when "
        "AUDIT_STORE_RESPONSE_SNIPPET=true",
    )
    filters: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Tenant/RBAC/metadata filters applied to the retrieval",
    )
    ip_address: Mapped[str | None] = mapped_column(
        String(45),
        nullable=True,
        comment="Client IPv4/IPv6 address",
    )
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="UTC append time; retention/partition key",
    )

    # No relationships by design: see class docstring.

    def __repr__(self) -> str:
        # Deliberately excludes query_text/response_text (PII / prompt content).
        return (
            f"<AuditLog id={self.id} tenant={self.tenant_id} user={self.user_id} "
            f"latency_ms={self.latency} tokens={self.tokens_used} success={self.success}>"
        )


class Conversation(Base, TimestampMixin):
    """Multi-turn chat conversation per user/tenant."""

    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_tenant_id_user_id", "tenant_id", "user_id"),
        Index("ix_conversations_tenant_id_updated_at", "tenant_id", "updated_at"),
        {"comment": "Multi-turn chat conversations for contextual RAG."},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New Chat")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=text("true"))

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )


class Message(Base):
    """Individual message in a conversation."""

    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
        Index("ix_messages_tenant_id_conversation_id", "tenant_id", "conversation_id"),
        {"comment": "Chat messages with role (user/assistant) and citations."},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
