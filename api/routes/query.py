"""RAG query route: embed -> retrieve -> generate -> **always** audit.

The audit contract is the point of this handler: exactly one ``audit_logs`` row
is written per request - on success with tokens and answer, on failure with the
error code - and the row is committed together with the decision to answer. If
the audit write itself fails, a *successful* answer is discarded (policy: no
unaudited answers); failures still return their original error so a broken
audit path cannot mask real outages.

Latency semantics: ``latency_ms`` covers embedding + retrieval + generation
(the pipeline), and is identical in the HTTP response and the audit row so the
two records correlate exactly.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any, Final

from fastapi import APIRouter, HTTPException, Request, status

from api.deps import CurrentUser, SessionDep, client_audit_context
from config import get_settings
from database import REQUEST_ID_CTX
from rbac import accessible_min_roles
from schemas.query import MAX_SNIPPET_CHARS, QueryRequest, QueryResponse, SourceChunk, TokenUsage
from services.agentic_rag import agentic_query
from services.audit import build_audit_log, coerce_request_id, record_audit
from services.providers import ProviderConfigurationError
from services.retrieval import (
    GenerationResult,
    RetrievalError,
    RetrievedChunk,
    generate_answer,
    get_query_embedding,
    retrieve_chunks,
)

logger: Final[logging.Logger] = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Answer a question from tenant knowledge (RBAC + audit logged)",
)
async def run_query(
    payload: QueryRequest,
    request: Request,
    user: CurrentUser,
    session: SessionDep,
) -> QueryResponse:
    settings = get_settings()
    started = time.perf_counter()
    request_id = REQUEST_ID_CTX.get()
    ip_address, user_agent = client_audit_context(request)
    top_k = payload.top_k if payload.top_k is not None else settings.retrieval_top_k
    allowed_roles = accessible_min_roles(user.role)

    chunks: list[RetrievedChunk] = []
    generation: GenerationResult | None = None
    agentic_result: dict[str, Any] | None = None
    success = True
    error_code: str | None = None
    error_message: str | None = None
    failure_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    failure_detail = "Question could not be answered"

    try:
        if payload.agentic:
            agentic_result = await agentic_query(
                session,
                tenant_id=user.tenant_id,
                user=user,
                question=payload.question,
                top_k=top_k,
                allowed_roles=allowed_roles,
                metadata_filters=payload.metadata_filters,
            )
            chunks = list(agentic_result.get("chunks") or [])
        else:
            query_embedding = await get_query_embedding(payload.question)
            chunks = await retrieve_chunks(
                session,
                tenant_id=user.tenant_id,
                query_embedding=query_embedding,
                top_k=top_k,
                allowed_roles=allowed_roles,
                metadata_filters=payload.metadata_filters,
            )
            generation = await generate_answer(
                question=payload.question, contexts=chunks
            )
    except ProviderConfigurationError as exc:
        success = False
        error_code, error_message = "provider_not_configured", str(exc)
        failure_status = status.HTTP_503_SERVICE_UNAVAILABLE
        failure_detail = "Retrieval/generation provider is not configured"
        logger.error("Query aborted: %s", exc)
    except RetrievalError as exc:
        success = False
        error_code, error_message = "rag_pipeline_failed", str(exc)
        failure_status = status.HTTP_502_BAD_GATEWAY
        failure_detail = "Retrieval or generation failed upstream"
        logger.error("Query pipeline failed: %s", exc)
    except Exception as exc:
        success = False
        error_code, error_message = "internal_error", str(exc)
        failure_detail = "Internal error while answering the question"
        logger.exception("Unexpected query failure")

    pipeline_ms = (time.perf_counter() - started) * 1000.0

    agentic_tokens = (agentic_result or {}).get("tokens") or {}
    answer_text = (
        generation.text if generation is not None else str((agentic_result or {}).get("answer") or "")
    )
    prompt_tokens = (
        generation.prompt_tokens if generation is not None else int(agentic_tokens.get("prompt_tokens", 0))
    )
    completion_tokens = (
        generation.completion_tokens
        if generation is not None
        else int(agentic_tokens.get("completion_tokens", 0))
    )
    tokens_estimated = (
        generation.estimated if generation is not None else bool(agentic_tokens.get("estimated", False))
    )
    model_name = generation.model if generation is not None else (agentic_result or {}).get("model")

    audit_log = build_audit_log(
        tenant_id=user.tenant_id,
        query_text=payload.question,
        latency_ms=pipeline_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        user=user,
        request_id=request_id,
        model_name=model_name,
        retrieved_chunk_count=len(chunks),
        success=success,
        error_code=error_code,
        error_message=error_message,
        response_text=answer_text or None,
        filters={
            "allowed_roles": sorted(role.value for role in allowed_roles),
            "top_k": top_k,
            "metadata_filters": payload.metadata_filters,
            "phase3_rbac_vector_filtering": True,
            "metadata_prefiltering_enabled": True,
            "agentic": payload.agentic,
            "source_type": (agentic_result or {}).get("source_type"),
            "sub_questions": (agentic_result or {}).get("sub_questions"),
        },
        ip_address=ip_address,
        user_agent=user_agent,
    )
    try:
        audit_id = await record_audit(session, audit_log)
        await session.commit()
    except Exception:
        logger.exception("Audit persistence failed (request_id=%s)", request_id)
        await session.rollback()
        if success:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Audit trail unavailable; request rejected",
            ) from None
        audit_id = None

    if not success:
        raise HTTPException(failure_status, failure_detail)
    if not answer_text:  # pragma: no cover - success implies an answer
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Question could not be answered"
        )

    sources = (
        [
            SourceChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
                distance=chunk.distance,
                snippet=chunk.content[:MAX_SNIPPET_CHARS],
                page_number=chunk.page_number,
                heading_path=chunk.heading_path,
                min_role_required=chunk.min_role_required,
            )
            for chunk in chunks
        ]
        if payload.include_sources
        else []
    )

    return QueryResponse(
        answer=answer_text,
        sources=sources,
        applied_role=user.role,
        top_k=top_k,
        retrieved_chunk_count=len(chunks),
        tokens=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated=tokens_estimated,
        ),
        latency_ms=pipeline_ms,
        model=model_name,
        request_id=coerce_request_id(request_id),
        audit_id=audit_id,
        generated_at=datetime.now(UTC),
        source_type=(agentic_result or {}).get("source_type"),
        confidence=(agentic_result or {}).get("confidence"),
        sub_questions=(agentic_result or {}).get("sub_questions"),
        web_sources=(agentic_result or {}).get("sources") or None,
    )
