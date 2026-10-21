"""Document ingestion: digest -> chunk -> embed -> persist.

Atomicity contract
------------------
``ingest_text`` writes the ``documents`` row and all ``document_chunks`` rows in
the caller's transaction and does **not** commit. On success the route commits
everything; on any exception the route rolls back, so a failed ingestion never
leaves a half-indexed document (``DocumentStatus.FEILED`` remains available for
a future async ingestion pipeline that must record failures durably).

All CPU/network work (splitting, embedding) runs in worker threads via
``asyncio.to_thread`` so a large document cannot stall the event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from models import Document, DocumentChunk, DocumentStatus, User
from schemas.documents import DocumentIngestRequest
from services.providers import create_embed_model, validate_embedding_dimension

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from llama_index.core.embeddings import BaseEmbedding

__all__ = [
    "EMBED_BATCH_SIZE",
    "DuplicateDocumentError",
    "EmptyDocumentError",
    "IngestResult",
    "IngestionError",
    "content_digest",
    "estimate_tokens",
    "ingest_text",
    "split_text",
]

#: Embeddings are requested in bounded batches to stay under provider request
#: size limits on documents that split into hundreds of chunks.
EMBED_BATCH_SIZE: Final[int] = 64
CHARS_PER_TOKEN_ESTIMATE: Final[int] = 4


class IngestionError(RuntimeError):
    """Ingestion failed before the document could be indexed."""


class DuplicateDocumentError(IngestionError):
    """A document with identical content already exists in this tenant."""


class EmptyDocumentError(IngestionError):
    """The submitted content produced no indexable text."""


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of one successful ingestion."""

    document_id: uuid.UUID
    chunks_created: int
    elapsed_ms: float
    content_hash: str


def content_digest(content: str) -> str:
    """SHA-256 hex digest of the UTF-8 content (per-tenant dedup key)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def estimate_tokens(chunk_text: str) -> int:
    """Approximate token count (~4 chars/token) for quota accounting.

    A tiktoken-backed counter can replace this for billing-grade accuracy;
    the value is advisory and never used for access decisions.
    """
    return max(1, len(chunk_text) // CHARS_PER_TOKEN_ESTIMATE)


def split_text(content: str) -> list[str]:
    """Split *content* into overlapping chunks with llama-index.

    Raises:
        IngestionError: llama-index is unavailable, or the content produced no
            non-empty chunks.
    """
    try:
        from llama_index.core.node_parser import SentenceSplitter
    except ImportError as exc:
        raise IngestionError(
            "llama-index is required for chunking: pip install llama-index"
        ) from exc

    settings = get_settings()
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = [part.strip() for part in splitter.split_text(content) if part.strip()]
    if not chunks:
        raise EmptyDocumentError("Document produced no text chunks after splitting")
    return chunks


async def _embed_texts(
    model: BaseEmbedding, texts: list[str]
) -> list[list[float]]:
    """Embed *texts* in batches off the event loop, validating every vector."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        batch_vectors = await asyncio.to_thread(
            model.get_text_embedding_batch, batch, show_progress=False
        )
        if len(batch_vectors) != len(batch):
            raise IngestionError(
                f"Embedding provider returned {len(batch_vectors)} vectors for "
                f"{len(batch)} texts; ingestion aborted"
            )
        for vector in batch_vectors:
            validate_embedding_dimension(vector)
        vectors.extend(batch_vectors)
    return vectors


async def ingest_text(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    actor: User,
    payload: DocumentIngestRequest,
    embed_model: BaseEmbedding | None = None,
) -> IngestResult:
    """Chunk, embed and persist one text document in a single transaction.

    Args:
        session: Session whose transaction the caller owns (commit/rollback).
        tenant_id: Owning tenant; must equal ``actor.tenant_id`` and the
            bound ``app.tenant_id`` GUC (RLS ``WITH CHECK`` enforces both).
        actor: Uploading user, recorded as ``documents.created_by``.
        payload: Validated ingest request (title, content, RBAC metadata...).
        embed_model: Optional pre-built embedding model (tests, warm caches).

    Returns:
        Document id, chunk count, elapsed milliseconds and content hash.

    Raises:
        DuplicateDocumentError: Identical content exists (or was ingested
            concurrently, detected via the unique constraint on flush).
        IngestionError / EmptyDocumentError / ProviderConfigurationError:
            Ingestion could not complete; all writes roll back with it.
    """
    if actor.tenant_id != tenant_id:
        raise IngestionError(
            f"Actor {actor.id} does not belong to tenant {tenant_id}"
        )

    settings = get_settings()
    started = time.perf_counter()
    digest = content_digest(payload.content)

    existing_id = await session.scalar(
        select(Document.id).where(
            Document.tenant_id == tenant_id,
            Document.content_hash == digest,
            Document.deleted_at.is_(None),
        )
    )
    if existing_id is not None:
        raise DuplicateDocumentError(
            f"Identical content is already indexed as document {existing_id}"
        )

    document = Document(
        tenant_id=tenant_id,
        title=payload.title,
        source_uri=payload.source_uri,
        source_system=payload.source_system,
        mime_type=payload.mime_type,
        file_size_bytes=len(payload.content.encode("utf-8")),
        content_hash=digest,
        status=DocumentStatus.PROCESSING,
        min_role_required=payload.min_role_required,
        doc_metadata=payload.metadata,
        created_by=actor.id,
    )
    session.add(document)
    try:
        await session.flush()  # assigns document.id; unique race -> duplicate
    except IntegrityError as exc:
        raise DuplicateDocumentError(
            "Identical content was ingested concurrently"
        ) from exc

    texts = await asyncio.to_thread(split_text, payload.content)
    model = embed_model if embed_model is not None else create_embed_model()
    vectors = await _embed_texts(model, texts)

    for index, (chunk_text, vector) in enumerate(
        zip(texts, vectors, strict=True)
    ):
        session.add(
            DocumentChunk(
                tenant_id=tenant_id,
                document_id=document.id,
                chunk_index=index,
                content=chunk_text,
                embedding=vector,
                embedding_model=settings.embedding_model,
                token_count=estimate_tokens(chunk_text),
                page_number=None,
                heading_path=None,
                chunk_metadata={
                    "tenant_id": str(tenant_id),
                    "document_id": str(document.id),
                    "document_title": payload.title,
                    "min_role_required": payload.min_role_required.value,
                    "allowed_roles": [r.value for r in [payload.min_role_required]],
                    "owner_id": str(actor.id),
                    "source_system": payload.source_system,
                    "source_uri": payload.source_uri,
                    **(payload.metadata or {}),
                },
            )
        )
    document.chunk_count = len(texts)
    document.status = DocumentStatus.INDEXED
    document.indexed_at = datetime.now(UTC)
    await session.flush()

    return IngestResult(
        document_id=document.id,
        chunks_created=len(texts),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        content_hash=digest,
    )
