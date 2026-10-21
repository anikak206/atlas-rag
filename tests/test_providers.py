"""Provider factory behaviour without network access or credentials.

The OpenAI integrations ship with the llama-index meta package, but the
Ollama integrations are optional. Those branches are asserted to fail with an
actionable :class:`ProviderConfigurationError`; if an operator has installed
the integration, the corresponding test is skipped rather than silently wrong.
"""

from __future__ import annotations

import importlib.util

import pytest

from config import get_settings
from services.providers import (
    EmbeddingDimensionMismatchError,
    ProviderConfigurationError,
    create_embed_model,
    create_llm,
    resolve_provider,
    validate_embedding_dimension,
)


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # pragma: no cover - parent package missing
        return False


def test_configured_provider_is_read_from_settings(settings_env: object) -> None:
    settings_env(LLM_PROVIDER="ollama")  # type: ignore[operator]
    assert resolve_provider() == "ollama"
    settings_env(LLM_PROVIDER="openai")  # type: ignore[operator]
    assert resolve_provider() == "openai"


def test_unsupported_provider_lists_valid_options() -> None:
    with pytest.raises(ProviderConfigurationError, match="Unsupported LLM provider"):
        create_llm("anthropic")


@pytest.mark.skipif(
    _installed("llama_index.llms.ollama"),
    reason="Ollama LLM integration is installed in this environment",
)
def test_ollama_llm_missing_integration_hints_at_pip_install() -> None:
    with pytest.raises(ProviderConfigurationError, match="llama-index-llms-ollama"):
        create_llm("ollama")


@pytest.mark.skipif(
    _installed("llama_index.embeddings.ollama"),
    reason="Ollama embedding integration is installed in this environment",
)
def test_ollama_embedding_missing_integration_hints_at_pip_install() -> None:
    with pytest.raises(
        ProviderConfigurationError, match="llama-index-embeddings-ollama"
    ):
        create_embed_model("ollama")


def test_openai_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
            create_llm()
        with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
            create_embed_model()
    finally:
        get_settings.cache_clear()


def test_embedding_dimension_guard() -> None:
    expected = get_settings().embedding_dimension
    validate_embedding_dimension([0.0] * expected)
    with pytest.raises(EmbeddingDimensionMismatchError, match=str(expected)):
        validate_embedding_dimension([0.0] * (expected - 1))
    with pytest.raises(EmbeddingDimensionMismatchError):
        validate_embedding_dimension([])
