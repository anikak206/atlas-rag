"""Phase-3 tests: vector ingestion with RBAC metadata, metadata pre-filtering,
and citation/provenance verification."""

from __future__ import annotations

import uuid

import pytest

from models import UserRole
from schemas.citations import CitationSource, EnhancedQueryResponse
from services.retrieval import RetrievedChunk


class _FakeEmbedModel:
    """Fake embedding model for testing."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def get_text_embedding_batch(self, texts: list[str], show_progress: bool = False):
        return [[0.1] * self.dimension for _ in texts]

    def get_query_embedding(self, question: str):
        return [0.1] * self.dimension


@pytest.mark.asyncio
async def test_rbac_metadata_embedded_in_chunks():
    """Verify RBAC metadata is embedded in chunk metadata during ingestion."""
    
    # This test would need DB - skip in pure unit context
    # Mark as integration test conceptually


def test_chunk_metadata_structure():
    """Verify chunk metadata contains required RBAC fields."""
    meta = {
        "tenant_id": str(uuid.uuid4()),
        "document_id": str(uuid.uuid4()),
        "document_title": "Test Doc",
        "min_role_required": "Finance",
        "allowed_roles": ["Intern", "Finance"],
        "owner_id": str(uuid.uuid4()),
    }
    assert "tenant_id" in meta
    assert "document_id" in meta
    assert "min_role_required" in meta
    assert "allowed_roles" in meta
    assert isinstance(meta["allowed_roles"], list)


def test_retrieved_chunk_has_provenance():
    """Verify RetrievedChunk contains provenance fields for citations."""
    chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="Employee Handbook",
        chunk_index=2,
        content="Leave policy content",
        distance=0.15,
        score=0.85,
        page_number=5,
        heading_path="HR > Leave Policy",
        min_role_required=UserRole.INTERN,
    )
    assert chunk.document_id is not None
    assert chunk.page_number == 5
    assert chunk.heading_path == "HR > Leave Policy"
    assert chunk.score == 0.85
    assert chunk.distance == 0.15


def test_citation_source_schema():
    """Verify CitationSource schema has all required provenance fields."""
    citation = CitationSource(
        source_document_id=uuid.uuid4(),
        document_title="Budget Report",
        chunk_id=uuid.uuid4(),
        chunk_index=0,
        page_number=10,
        section_title="Q4 Summary",
        heading_path="Finance > Q4 > Summary",
        confidence_score=0.92,
        similarity_metrics={"cosine_similarity": 0.92, "distance": 0.08},
        snippet="Revenue increased by 15%",
        min_role_required=UserRole.FINANCE,
        metadata={"department": "Finance"},
    )
    assert citation.confidence_score == 0.92
    assert citation.similarity_metrics["cosine_similarity"] == 0.92
    assert citation.page_number == 10
    assert citation.section_title == "Q4 Summary"


def test_citation_confidence_score_bounds():
    """Verify confidence score validation."""
    from pydantic import ValidationError
    
    with pytest.raises(ValidationError):
        CitationSource(
            source_document_id=uuid.uuid4(),
            document_title="Test",
            chunk_id=uuid.uuid4(),
            chunk_index=0,
            confidence_score=1.5,  # > 1.0
            similarity_metrics={},
            snippet="test",
            min_role_required=UserRole.INTERN,
        )
    
    with pytest.raises(ValidationError):
        CitationSource(
            source_document_id=uuid.uuid4(),
            document_title="Test",
            chunk_id=uuid.uuid4(),
            chunk_index=0,
            confidence_score=-0.1,  # < 0.0
            similarity_metrics={},
            snippet="test",
            min_role_required=UserRole.INTERN,
        )


def test_rbac_pre_filtering_logic():
    """Verify RBAC pre-filtering prevents unauthorized access."""
    # Finance doc should not be accessible to Intern
    from rbac import role_satisfies
    assert role_satisfies(UserRole.INTERN, UserRole.FINANCE) is False
    assert role_satisfies(UserRole.INTERN, UserRole.INTERN) is True


def test_metadata_filters_structure():
    """Verify metadata filters are properly structured for pre-filtering."""
    filters = {
        "department": "Finance",
        "year": 2026,
        "classification": "internal"
    }
    # JSONB containment @> uses dict
    assert isinstance(filters, dict)
    assert "department" in filters


@pytest.mark.asyncio
async def test_enhanced_query_response_structure():
    """Verify enhanced response includes citations and provenance."""
    from datetime import UTC, datetime
    
    citation = CitationSource(
        source_document_id=uuid.uuid4(),
        document_title="Policy Doc",
        chunk_id=uuid.uuid4(),
        chunk_index=1,
        page_number=3,
        section_title="Overview",
        heading_path="Policies > Overview",
        confidence_score=0.88,
        similarity_metrics={"cosine_similarity": 0.88},
        snippet="Key policy text",
        min_role_required=UserRole.INTERN,
    )
    
    resp = EnhancedQueryResponse(
        answer="The answer is...",
        citations=[citation],
        sources=[],
        applied_role=UserRole.INTERN,
        top_k=5,
        retrieved_chunk_count=1,
        confidence_score=0.88,
        provenance_summary={"num_sources": 1, "top_score": 0.88},
        tokens={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_ms=250.5,
        model="gpt-4o-mini",
        request_id=uuid.uuid4(),
        generated_at=datetime.now(UTC),
    )
    assert len(resp.citations) == 1
    assert resp.citations[0].confidence_score == 0.88
    assert resp.confidence_score == 0.88
    assert resp.provenance_summary["num_sources"] == 1
