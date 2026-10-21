"""Prompt assembly, token accounting and generation guards.

``retrieve_chunks`` executes SQL against pgvector, so it is exercised by the
integration checklist rather than here; the prompt/usage/answer pieces around it
are pure and fully covered.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from config import get_settings
from models import UserRole
from services.providers import EmbeddingDimensionMismatchError
from services.retrieval import (
    NO_CONTEXT_ANSWER,
    GenerationError,
    RetrievalError,
    RetrievedChunk,
    build_prompt,
    extract_token_usage,
    generate_answer,
    get_query_embedding,
)


def _chunk(**overrides: Any) -> RetrievedChunk:
    values: dict[str, Any] = {
        "chunk_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "document_title": "Employee Handbook",
        "chunk_index": 0,
        "content": "Employees receive 24 days of annual leave.",
        "distance": 0.12,
        "score": 0.88,
        "page_number": None,
        "heading_path": None,
        "min_role_required": UserRole.INTERN,
    }
    values.update(overrides)
    return RetrievedChunk(**values)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def test_build_prompt_numbers_passages_with_provenance() -> None:
    prompt = build_prompt(
        "How much leave do I get?",
        [
            _chunk(),
            _chunk(
                document_title="FY2026 Budget",
                chunk_index=3,
                heading_path="Cloud > Caps",
                page_number=7,
                min_role_required=UserRole.FINANCE,
            ),
        ],
    )
    assert "[1] Employee Handbook (chunk 0" in prompt
    assert "[2] FY2026 Budget (chunk 3; Cloud > Caps; p. 7; access >= Finance)" in prompt
    assert "access >= Intern" in prompt
    assert "QUESTION: How much leave do I get?" in prompt
    assert "ONLY the numbered CONTEXT" in prompt
    assert prompt.rstrip().endswith("ANSWER:")


def test_build_prompt_without_context_still_documents_question() -> None:
    prompt = build_prompt("What is our policy?", [])
    assert "CONTEXT:" in prompt
    assert "QUESTION: What is our policy?" in prompt


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------
def test_reported_usage_is_read_from_openai_raw_payload() -> None:
    response = SimpleNamespace(
        text="answer",
        raw=SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7)
        ),
    )
    assert extract_token_usage(
        response, prompt_text="p", completion_text="a"
    ) == (11, 7, False)


def test_reported_usage_is_read_from_dict_payload() -> None:
    response = SimpleNamespace(
        text="answer", raw={"usage": {"prompt_tokens": 40, "completion_tokens": 9}}
    )
    assert extract_token_usage(response, prompt_text="p", completion_text="a") == (
        40,
        9,
        False,
    )


def test_missing_usage_falls_back_to_character_estimate() -> None:
    response = SimpleNamespace(text="answer", raw=None)
    prompt_tokens, completion_tokens, estimated = extract_token_usage(
        response, prompt_text="p" * 400, completion_text="a" * 40
    )
    assert estimated is True
    assert prompt_tokens == 100
    assert completion_tokens == 10


def test_partial_or_invalid_usage_triggers_estimation() -> None:
    response = SimpleNamespace(
        text="answer", raw={"usage": {"prompt_tokens": 10, "completion_tokens": -1}}
    )
    prompt_tokens, completion_tokens, estimated = extract_token_usage(
        response, prompt_text="p" * 40, completion_text="a" * 8
    )
    assert estimated is True
    assert (prompt_tokens, completion_tokens) == (10, 2)


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
class _ExplodingLLM:
    """Fails loudly if the pipeline calls the model without context."""

    def complete(self, prompt: str) -> Any:
        raise AssertionError(f"LLM must not be called, got prompt: {prompt!r}")


class _FakeLLM:
    def __init__(self, text: str = "  You get 24 days [1].  ") -> None:
        self._text = text
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        return SimpleNamespace(
            text=self._text,
            raw={"usage": {"prompt_tokens": 30, "completion_tokens": 12}},
        )


async def test_empty_retrieval_returns_deterministic_refusal_without_llm() -> None:
    result = await generate_answer(
        question="Anything?", contexts=[], llm=_ExplodingLLM()
    )
    assert result.text == NO_CONTEXT_ANSWER
    assert result.model is None
    assert (result.prompt_tokens, result.completion_tokens) == (0, 0)
    assert result.estimated is False


async def test_generation_uses_context_and_reports_usage() -> None:
    llm = _FakeLLM()
    result = await generate_answer(
        question="How much leave?", contexts=[_chunk()], llm=llm
    )
    assert result.text == "You get 24 days [1]."
    assert result.model == get_settings().llm_model
    assert (result.prompt_tokens, result.completion_tokens) == (30, 12)
    assert result.estimated is False
    assert "Employees receive 24 days" in llm.prompts[0]


async def test_empty_completion_is_an_error() -> None:
    with pytest.raises(GenerationError, match="empty completion"):
        await generate_answer(
            question="Q", contexts=[_chunk()], llm=_FakeLLM(text="   ")
        )


async def test_provider_failure_is_wrapped() -> None:
    class _BrokenLLM:
        def complete(self, prompt: str) -> Any:
            raise TimeoutError("upstream timed out")

    with pytest.raises(GenerationError, match="timed out"):
        await generate_answer(question="Q", contexts=[_chunk()], llm=_BrokenLLM())


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------
async def test_query_embedding_is_dimension_checked() -> None:
    expected = get_settings().embedding_dimension
    good = SimpleNamespace(get_query_embedding=lambda question: [0.0] * expected)
    assert await get_query_embedding("hello", embed_model=good) == [0.0] * expected

    bad = SimpleNamespace(get_query_embedding=lambda question: [0.0] * (expected - 3))
    with pytest.raises(EmbeddingDimensionMismatchError):
        await get_query_embedding("hello", embed_model=bad)


async def test_query_embedding_provider_failure_is_wrapped() -> None:
    class _BrokenEmbedding:
        def get_query_embedding(self, question: str) -> list[float]:
            raise ConnectionError("provider unreachable")

    with pytest.raises(RetrievalError, match="provider unreachable"):
        await get_query_embedding("hello", embed_model=_BrokenEmbedding())
