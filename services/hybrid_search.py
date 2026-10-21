"""Hybrid Search: Dense (pgvector) + Sparse (PostgreSQL Full-Text Search) with RRF reranking.

Combines semantic similarity and keyword matching for better retrieval,
with RBAC/Tenant filtering applied at every stage.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from models import Document, DocumentChunk, DocumentStatus, UserRole
from services.retrieval import RetrievedChunk, get_query_embedding


@dataclass(frozen=True, slots=True)
class HybridResult:
    """Result from hybrid search before final filtering."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    content: str
    page_number: int | None
    heading_path: str | None
    min_role_required: UserRole
    dense_score: float
    sparse_score: float
    rrf_score: float


async def sparse_search(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    query_text: str,
    top_k: int,
    allowed_roles: Iterable[UserRole],
    metadata_filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """PostgreSQL Full-Text Search using tsvector."""
    allowed_roles_list = list(allowed_roles)
    
    # Use plainto_tsquery for natural language queries
    stmt = (
        select(
            DocumentChunk.id.label("chunk_id"),
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
            DocumentChunk.page_number,
            DocumentChunk.heading_path,
            DocumentChunk.chunk_metadata,
            Document.title.label("document_title"),
            Document.min_role_required,
            func.ts_rank_cd(
                func.to_tsvector('english', DocumentChunk.content),
                func.plainto_tsquery('english', query_text)
            ).label("sparse_rank"),
        )
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(
            DocumentChunk.tenant_id == tenant_id,
            Document.status == DocumentStatus.INDEXED,
            Document.deleted_at.is_(None),
            Document.min_role_required.in_(allowed_roles_list),
            func.to_tsvector('english', DocumentChunk.content).op('@@')(
                func.plainto_tsquery('english', query_text)
            ),
        )
        .order_by(text("sparse_rank DESC"))
        .limit(top_k * 2)
    )
    
    if metadata_filters:
        stmt = stmt.where(Document.doc_metadata.contains(metadata_filters))
    
    rows = (await session.execute(stmt)).all()
    return [
        {
            "chunk_id": row.chunk_id,
            "document_id": row.document_id,
            "document_title": row.document_title,
            "chunk_index": row.chunk_index,
            "content": row.content,
            "page_number": row.page_number,
            "heading_path": row.heading_path,
            "min_role_required": row.min_role_required,
            "chunk_metadata": row.chunk_metadata,
            "sparse_score": float(row.sparse_rank or 0.0),
        }
        for row in rows
    ]


async def dense_search(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    query_embedding: list[float],
    top_k: int,
    allowed_roles: Iterable[UserRole],
    metadata_filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Dense vector search using pgvector."""
    allowed_roles_list = list(allowed_roles)
    distance_expr = DocumentChunk.embedding.cosine_distance(query_embedding)
    
    stmt = (
        select(
            DocumentChunk.id.label("chunk_id"),
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
            DocumentChunk.page_number,
            DocumentChunk.heading_path,
            DocumentChunk.chunk_metadata,
            Document.title.label("document_title"),
            Document.min_role_required,
            distance_expr.label("distance"),
        )
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(
            DocumentChunk.tenant_id == tenant_id,
            Document.status == DocumentStatus.INDEXED,
            Document.deleted_at.is_(None),
            Document.min_role_required.in_(allowed_roles_list),
            DocumentChunk.chunk_metadata.contains({"tenant_id": str(tenant_id)}),
        )
        .order_by(distance_expr)
        .limit(top_k * 2)
    )
    
    if metadata_filters:
        stmt = stmt.where(Document.doc_metadata.contains(metadata_filters))
    
    rows = (await session.execute(stmt)).all()
    return [
        {
            "chunk_id": row.chunk_id,
            "document_id": row.document_id,
            "document_title": row.document_title,
            "chunk_index": row.chunk_index,
            "content": row.content,
            "page_number": row.page_number,
            "heading_path": row.heading_path,
            "min_role_required": row.min_role_required,
            "chunk_metadata": row.chunk_metadata,
            "dense_score": 1.0 - float(row.distance),
        }
        for row in rows
    ]


async def hybrid_retrieve(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    query_text: str,
    query_embedding: list[float] | None = None,
    top_k: int = 10,
    allowed_roles: Iterable[UserRole],
    metadata_filters: dict[str, Any] | None = None,
    alpha: float = 0.5,  # Weight for dense vs sparse
) -> list[RetrievedChunk]:
    """Hybrid search with Reciprocal Rank Fusion (RRF)."""
    allowed_roles_list = list(allowed_roles)
    allowed_role_values = [r.value for r in allowed_roles_list]
    
    # Get embedding if not provided
    if query_embedding is None:
        query_embedding = await get_query_embedding(query_text)
    
    # Run both searches in parallel
    dense_results, sparse_results = await asyncio.gather(
        dense_search(
            session,
            tenant_id=tenant_id,
            query_embedding=query_embedding,
            top_k=top_k,
            allowed_roles=allowed_roles_list,
            metadata_filters=metadata_filters,
        ),
        sparse_search(
            session,
            tenant_id=tenant_id,
            query_text=query_text,
            top_k=top_k,
            allowed_roles=allowed_roles_list,
            metadata_filters=metadata_filters,
        ),
    )
    
    # Build rank maps
    k = 60  # RRF constant
    rrf_scores = {}
    
    for rank, result in enumerate(dense_results):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = {
            **result,
            "rrf_score": rrf_scores.get(chunk_id, {}).get("rrf_score", 0) + alpha / (k + rank + 1),
            "dense_score": result["dense_score"],
            "sparse_score": rrf_scores.get(chunk_id, {}).get("sparse_score", 0.0),
        }
    
    for rank, result in enumerate(sparse_results):
        chunk_id = result["chunk_id"]
        if chunk_id in rrf_scores:
            rrf_scores[chunk_id]["rrf_score"] += (1 - alpha) / (k + rank + 1)
            rrf_scores[chunk_id]["sparse_score"] = result["sparse_score"]
        else:
            rrf_scores[chunk_id] = {
                **result,
                "rrf_score": (1 - alpha) / (k + rank + 1),
                "dense_score": 0.0,
                "sparse_score": result["sparse_score"],
            }
    
    # Sort by RRF score and apply RBAC filtering
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    
    chunks = []
    for result in sorted_results:
        chunk_meta = result.get("chunk_metadata") or {}
        chunk_min_role = chunk_meta.get("min_role_required") or result["min_role_required"].value
        if chunk_min_role not in allowed_role_values and result["min_role_required"].value not in allowed_role_values:
            continue
        chunks.append(
            RetrievedChunk(
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                document_title=result["document_title"],
                chunk_index=result["chunk_index"],
                content=result["content"],
                distance=1.0 - result.get("dense_score", 0.5),
                score=max(result.get("dense_score", 0.5), result.get("sparse_score", 0.5)),
                page_number=result["page_number"],
                heading_path=result["heading_path"],
                min_role_required=result["min_role_required"],
            )
        )
        if len(chunks) >= top_k:
            break
    
    return chunks
