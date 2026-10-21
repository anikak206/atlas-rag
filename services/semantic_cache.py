"""Semantic caching layer for RAG queries.

Caches query results based on semantic similarity of questions
to avoid redundant LLM calls and vector searches.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    """Cached query result."""
    question: str
    question_hash: str
    answer: str
    sources: list[dict[str, Any]]
    tokens: dict[str, Any]
    model: str | None
    timestamp: float
    hits: int = 0
    embedding: list[float] | None = None


class SemanticCache:
    """In-memory semantic cache with similarity-based lookup."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600, similarity_threshold: float = 0.95):
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._similarity_threshold = similarity_threshold
    
    def _hash_question(self, question: str) -> str:
        """Create hash of normalized question."""
        normalized = question.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    async def get(self, question: str, question_embedding: list[float] | None = None) -> CacheEntry | None:
        """Get cached result if similar question found."""
        now = time.time()
        qhash = self._hash_question(question)
        
        # Exact match first
        if qhash in self._cache:
            entry = self._cache[qhash]
            if now - entry.timestamp < self._ttl_seconds:
                entry.hits += 1
                return entry
            else:
                del self._cache[qhash]
        
        # Semantic similarity search
        if question_embedding:
            best_match = None
            best_score = 0.0
            for entry in self._cache.values():
                if now - entry.timestamp >= self._ttl_seconds:
                    continue
                if entry.embedding:
                    score = self._cosine_similarity(question_embedding, entry.embedding)
                    if score > best_score and score >= self._similarity_threshold:
                        best_score = score
                        best_match = entry
            if best_match:
                best_match.hits += 1
                return best_match
        
        return None
    
    def put(self, question: str, result: dict[str, Any], question_embedding: list[float] | None = None):
        """Store result in cache."""
        if len(self._cache) >= self._max_size:
            # Evict oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]
        
        entry = CacheEntry(
            question=question,
            question_hash=self._hash_question(question),
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            tokens=result.get("tokens", {}),
            model=result.get("model"),
            timestamp=time.time(),
            embedding=question_embedding,
        )
        self._cache[entry.question_hash] = entry
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        active = sum(1 for e in self._cache.values() if now - e.timestamp < self._ttl_seconds)
        total_hits = sum(e.hits for e in self._cache.values())
        return {
            "size": len(self._cache),
            "active_entries": active,
            "total_hits": total_hits,
            "max_size": self._max_size,
        }
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()


# Global cache instance
semantic_cache = SemanticCache()
