"""Agentic (multi-hop) RAG: query decomposition, retrieval and web fallback.

The pipeline is best-effort by design: decomposition failures fall back to the
original question, a failing sub-query does not abort the others, and low
confidence degrades to web search. Only retrieval and generation errors that
make an answer impossible are allowed to propagate.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from config import get_settings
from services.hybrid_search import hybrid_retrieve
from services.providers import create_llm
from services.retrieval import RetrievedChunk, generate_answer, get_query_embedding

__all__ = [
    "CONFIDENCE_THRESHOLD",
    "agentic_query",
    "decompose_query",
    "web_search_fallback",
]

logger = logging.getLogger(__name__)

#: Chunks score on the ``0..1`` similarity scale (dense cosine similarity, or
#: ``ts_rank_cd`` for sparse hits). When the best chunk is below this, the
#: knowledge base is treated as unable to answer.
CONFIDENCE_THRESHOLD = 0.3

TAVILY_ENDPOINT = "https://api.tavily.com/search"
WEB_SEARCH_TIMEOUT_SECONDS = 10.0
WEB_SNIPPET_CHARS = 500

_DECOMPOSITION_PROMPT = """Decompose the following question into simple, independent \
sub-questions that can each be answered separately.
If the question is already simple, return it unchanged.

Question: {question}

Respond with a JSON array of strings only."""


async def decompose_query(question: str, llm: Any | None = None) -> list[str]:
    """Split a multi-part *question* into sub-questions.

    Falls back to ``[question]`` whenever decomposition is unavailable or the
    model returns something that is not a JSON array of strings.
    """
    try:
        model = llm if llm is not None else create_llm()
        response = await asyncio.to_thread(
            model.complete, _DECOMPOSITION_PROMPT.format(question=question)
        )
        text = (response.text or "").strip()
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            return [question]
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, list):
            return [question]
        sub_questions = [str(item).strip() for item in parsed if str(item).strip()]
        return sub_questions or [question]
    except Exception as exc:  # noqa: BLE001 - decomposition must never fail a query
        logger.warning("Query decomposition unavailable (%s); using original query", exc)
        return [question]


async def web_search_fallback(query: str) -> dict[str, Any]:
    """Answer from the public web when the knowledge base has no confident match.

    Uses Tavily when ``TAVILY_API_KEY`` is configured. Without a key - or if the
    provider fails - the response says so explicitly instead of inventing an
    answer, and ``sources`` stays empty so nothing is attributed to a fake URL.
    """
    api_key = get_settings().tavily_api_key
    if api_key is not None:
        try:
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    TAVILY_ENDPOINT,
                    json={
                        "api_key": api_key.get_secret_value(),
                        "query": query,
                        "search_depth": "basic",
                        "max_results": 5,
                        "include_answer": True,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Web search failed (%s); returning unconfigured fallback", exc)
        else:
            results = [
                item for item in payload.get("results", []) if item.get("content")
            ]
            if results:
                answer = payload.get("answer") or "\n\n".join(
                    f"{item.get('title') or 'Result'}: {item['content']}"
                    for item in results
                )
                return {
                    "answer": answer,
                    "sources": [
                        {
                            "type": "web",
                            "title": item.get("title") or "Web result",
                            "url": item.get("url"),
                            "snippet": str(item["content"])[:WEB_SNIPPET_CHARS],
                        }
                        for item in results
                    ],
                    "source_type": "web_fallback",
                }

    return {
        "answer": (
            f"No internal document matched '{query}' closely enough to answer "
            "confidently, and external web search is not configured. Set "
            "TAVILY_API_KEY to enable it."
        ),
        "sources": [],
        "source_type": "web_fallback",
    }


async def _retrieve(
    session: Any,
    *,
    tenant_id: Any,
    question: str,
    top_k: int,
    allowed_roles: Any,
    metadata_filters: dict[str, Any] | None,
) -> list[RetrievedChunk]:
    """Run one hybrid retrieval for *question*."""
    query_embedding = await get_query_embedding(question)
    return await hybrid_retrieve(
        session,
        tenant_id=tenant_id,
        query_text=question,
        query_embedding=query_embedding,
        top_k=top_k,
        allowed_roles=allowed_roles,
        metadata_filters=metadata_filters,
    )


async def agentic_query(
    session: Any,
    *,
    tenant_id: Any,
    user: Any,
    question: str,
    top_k: int = 10,
    allowed_roles: Any,
    metadata_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer *question*, fanning out over sub-questions when needed.

    Returns a mapping with ``answer``, ``sources``/``chunks``, ``source_type``
    (``"internal"`` or ``"web_fallback"``), ``confidence`` and, when
    decomposition happened, ``sub_questions``.
    """
    sub_questions = await decompose_query(question)

    if len(sub_questions) > 1:
        per_question_k = max(1, top_k // len(sub_questions) + 2)
        collected: list[RetrievedChunk] = []
        for sub_question in sub_questions:
            try:
                collected.extend(
                    await _retrieve(
                        session,
                        tenant_id=tenant_id,
                        question=sub_question,
                        top_k=per_question_k,
                        allowed_roles=allowed_roles,
                        metadata_filters=metadata_filters,
                    )
                )
            except (RuntimeError, ValueError, OSError) as exc:
                logger.warning(
                    "Sub-question retrieval failed for %r: %s", sub_question, exc
                )

        seen: set[Any] = set()
        chunks: list[RetrievedChunk] = []
        for chunk in collected:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                chunks.append(chunk)
        chunks = chunks[:top_k]
    else:
        chunks = await _retrieve(
            session,
            tenant_id=tenant_id,
            question=question,
            top_k=top_k,
            allowed_roles=allowed_roles,
            metadata_filters=metadata_filters,
        )

    # The best chunk decides: averaging would let one strong hit be diluted by
    # weaker ones that merely passed the similarity floor.
    confidence = max((chunk.score for chunk in chunks), default=0.0)

    if not chunks or confidence < CONFIDENCE_THRESHOLD:
        logger.info("Low confidence (%.3f); using web fallback", confidence)
        web_result = await web_search_fallback(question)
        return {
            "answer": web_result["answer"],
            "sources": web_result["sources"],
            "chunks": [],
            "source_type": "web_fallback",
            "confidence": confidence,
            "sub_questions": sub_questions if len(sub_questions) > 1 else None,
        }

    generation = await generate_answer(question=question, contexts=chunks)
    return {
        "answer": generation.text,
        "sources": [],
        "chunks": chunks,
        "source_type": "internal",
        "confidence": confidence,
        "sub_questions": sub_questions if len(sub_questions) > 1 else None,
        "tokens": {
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "total_tokens": generation.prompt_tokens + generation.completion_tokens,
            "estimated": generation.estimated,
        },
        "model": generation.model,
    }
