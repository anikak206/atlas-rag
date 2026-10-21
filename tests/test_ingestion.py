"""Ingestion helpers: digests, token estimates and chunking.

Actual embedding/persistence paths need a provider and PostgreSQL, so they are
covered by the integration checklist in the README, not by this suite.
"""

from __future__ import annotations

import hashlib

import pytest

from services.ingestion import (
    EmptyDocumentError,
    content_digest,
    estimate_tokens,
    split_text,
)


def test_content_digest_is_sha256_of_utf8() -> None:
    assert content_digest("hello") == hashlib.sha256(b"hello").hexdigest()
    assert len(content_digest("hello")) == 64
    assert content_digest("hello") != content_digest("hello!")
    assert content_digest("héllo") == hashlib.sha256("héllo".encode()).hexdigest()


def test_content_digest_is_stable_across_calls() -> None:
    body = "The quick brown fox jumps over the lazy dog. " * 20
    assert content_digest(body) == content_digest(body)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", 1), ("abc", 1), ("a" * 4, 1), ("a" * 400, 100), ("a" * 401, 100)],
)
def test_estimate_tokens_floors_at_one(text: str, expected: int) -> None:
    assert estimate_tokens(text) == expected


def test_split_text_produces_multiple_trimmed_chunks(settings_env: object) -> None:
    pytest.importorskip("tiktoken")
    settings_env(CHUNK_SIZE="200", CHUNK_OVERLAP="20")  # type: ignore[operator]
    sentence = "Refunds are processed within five business days of approval. "
    chunks = split_text(sentence * 80)

    assert len(chunks) > 1
    assert all(chunk == chunk.strip() for chunk in chunks)
    assert all(chunk for chunk in chunks)
    # Chunking must not lose content (the splitter packs every sentence).
    assert "Refunds are processed" in chunks[0]
    assert sum(len(chunk) for chunk in chunks) >= len(sentence * 80) * 0.8


def test_split_text_single_short_document_stays_one_chunk(settings_env: object) -> None:
    pytest.importorskip("tiktoken")
    settings_env(CHUNK_SIZE="256", CHUNK_OVERLAP="32")  # type: ignore[operator]
    assert split_text("One short sentence.") == ["One short sentence."]


def test_split_text_rejects_blank_content(settings_env: object) -> None:
    pytest.importorskip("tiktoken")
    settings_env(CHUNK_SIZE="256", CHUNK_OVERLAP="32")  # type: ignore[operator]
    with pytest.raises(EmptyDocumentError):
        split_text("   \n\n\t  ")
