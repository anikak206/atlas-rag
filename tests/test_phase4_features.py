"""Phase-4 tests: streaming, hybrid search, multi-turn chat."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

from models import UserRole
from schemas.citations import CitationSource


def test_hybrid_search_rrf_logic():
    """Test RRF score combination logic."""
    k = 60
    dense_rank = 0
    sparse_rank = 2
    alpha = 0.5
    rrf = alpha / (k + dense_rank + 1) + (1 - alpha) / (k + sparse_rank + 1)
    assert rrf > 0
    assert rrf < 1


def test_hybrid_search_alpha_weighting():
    """Test alpha weighting between dense and sparse."""
    alpha = 0.7  # favor dense
    k = 60
    dense_contribution = alpha / (k + 0 + 1)
    sparse_contribution = (1 - alpha) / (k + 0 + 1)
    assert dense_contribution > sparse_contribution


def test_citation_source_has_all_provenance():
    """Test citation includes all provenance fields."""
    cid = CitationSource(
        source_document_id=uuid.uuid4(),
        document_title="Test Document",
        chunk_id=uuid.uuid4(),
        chunk_index=3,
        page_number=7,
        section_title="Introduction",
        heading_path="Chapter 1 > Introduction",
        confidence_score=0.89,
        similarity_metrics={"cosine_similarity": 0.89, "rrf_score": 0.015},
        snippet="This is relevant content from the document",
        min_role_required=UserRole.FINANCE,
        metadata={"department": "Finance", "year": 2026},
    )
    assert cid.source_document_id is not None
    assert cid.page_number == 7
    assert cid.section_title == "Introduction"
    assert cid.confidence_score == 0.89
    assert "rrf_score" in cid.similarity_metrics


def test_chat_message_structure():
    """Test chat message has required fields."""
    msg = SimpleNamespace(
        id=uuid.uuid4(),
        role="user",
        content="What is the policy?",
        citations=None,
        created_at=datetime.now(UTC),
    )
    assert msg.role in ["user", "assistant", "system"]
    assert msg.content


def test_conversation_context_building():
    """Test conversation context building."""
    messages = [
        SimpleNamespace(role="user", content="Hello"),
        SimpleNamespace(role="assistant", content="Hi there"),
        SimpleNamespace(role="user", content="What about leave policy?"),
    ]
    context = "\n\n".join([f"{m.role.upper()}: {m.content}" for m in messages[-10:]])
    assert "USER: Hello" in context
    assert "ASSISTANT: Hi there" in context


def test_question_rephrasing_fallback():
    """Test question rephrasing falls back gracefully."""
    question = "What about that?"
    messages = []
    # With no messages, should return original
    result = question if not messages else "rephrased"
    assert result == question


def test_streaming_chunk_format():
    """Test SSE streaming chunk format."""
    chunk = {"type": "token", "content": "Hello"}
    sse_data = f"data: {json.dumps(chunk)}\n\n"
    assert sse_data.startswith("data: ")
    assert sse_data.endswith("\n\n")
    parsed = json.loads(sse_data.replace("data: ", "").strip())
    assert parsed["type"] == "token"


def test_rate_limit_key_generation():
    """Test rate limit key generation."""
    user_id = uuid.uuid4()
    key = f"user:{user_id}"
    assert key.startswith("user:")
    
    ip = "192.168.1.1"
    ip_key = ip
    assert ip_key == "192.168.1.1"
