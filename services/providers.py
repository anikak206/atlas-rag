"""Config-driven provider factories for embeddings and chat models.

The provider is selected once via ``LLM_PROVIDER`` (see :mod:`config`); llama
integrations are imported lazily inside each factory so that:

* importing this module never performs network I/O or requires credentials;
* a missing optional integration (e.g. Ollama) produces an actionable
  :class:`ProviderConfigurationError` instead of an ImportError at request time.

Adding a provider = adding two factory functions and two registry entries.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final, TypeVar

from config import get_settings

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms import LLM

ProviderT = TypeVar("ProviderT")

__all__ = [
    "EmbeddingDimensionMismatchError",
    "ProviderConfigurationError",
    "create_embed_model",
    "create_llm",
    "resolve_provider",
    "validate_embedding_dimension",
]


class ProviderConfigurationError(RuntimeError):
    """A provider cannot be constructed from the current configuration."""


class EmbeddingDimensionMismatchError(RuntimeError):
    """An embedding vector's width does not match ``EMBEDDING_DIMENSION``."""


def resolve_provider() -> str:
    """Return the configured provider name (``openai`` or ``ollama``)."""
    return get_settings().llm_provider


def validate_embedding_dimension(vector: list[float]) -> None:
    """Raise :class:`EmbeddingDimensionMismatchError` on width mismatch.

    Guards the first embedding call of every ingestion/query path so a model
    change (e.g. ``EMBEDDING_MODEL`` swapped without a schema migration) fails
    loudly instead of producing unsearchable rows.
    """
    expected = get_settings().embedding_dimension
    if len(vector) != expected:
        raise EmbeddingDimensionMismatchError(
            f"Embedding has {len(vector)} dimensions but the schema was created "
            f"for {expected} (EMBEDDING_DIMENSION). Either restore the model or "
            "run a schema migration plus full re-indexing."
        )


# ---------------------------------------------------------------------------
# Embedding factories
# ---------------------------------------------------------------------------
def _openai_embedding() -> BaseEmbedding:
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
    except ImportError as exc:  # pragma: no cover - meta package ships this
        raise ProviderConfigurationError(
            "OpenAI embedding integration is not installed: "
            "pip install llama-index-embeddings-openai"
        ) from exc

    settings = get_settings()
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    if api_key is None:
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderConfigurationError(
                "LLM_PROVIDER=openai but neither OPENAI_API_KEY nor the .env "
                "OPENAI_API_KEY setting is present."
            )
    try:
        return OpenAIEmbedding(model=settings.embedding_model, api_key=api_key)
    except (TypeError, ValueError) as exc:
        raise ProviderConfigurationError(
            f"Could not construct OpenAIEmbedding({settings.embedding_model!r}): {exc}"
        ) from exc


def _ollama_embedding() -> BaseEmbedding:
    try:
        from llama_index.embeddings.ollama import OllamaEmbedding
    except ImportError as exc:
        raise ProviderConfigurationError(
            "Ollama embedding integration is not installed: "
            "pip install llama-index-embeddings-ollama"
        ) from exc

    settings = get_settings()
    try:
        return OllamaEmbedding(
            model_name=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
    except (TypeError, ValueError) as exc:
        raise ProviderConfigurationError(
            f"Could not construct OllamaEmbedding({settings.embedding_model!r}) "
            f"against {settings.ollama_base_url}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Chat-LLM factories
# ---------------------------------------------------------------------------
def _openai_llm() -> LLM:
    try:
        from llama_index.llms.openai import OpenAI
    except ImportError as exc:  # pragma: no cover - meta package ships this
        raise ProviderConfigurationError(
            "OpenAI LLM integration is not installed: "
            "pip install llama-index-llms-openai"
        ) from exc

    settings = get_settings()
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    if api_key is None:
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderConfigurationError(
                "LLM_PROVIDER=openai but neither OPENAI_API_KEY nor the .env "
                "OPENAI_API_KEY setting is present."
            )
    try:
        return OpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=api_key,
        )
    except (TypeError, ValueError) as exc:
        raise ProviderConfigurationError(
            f"Could not construct OpenAI LLM({settings.llm_model!r}): {exc}"
        ) from exc


def _ollama_llm() -> LLM:
    try:
        from llama_index.llms.ollama import Ollama
    except ImportError as exc:
        raise ProviderConfigurationError(
            "Ollama LLM integration is not installed: "
            "pip install llama-index-llms-ollama"
        ) from exc

    settings = get_settings()
    try:
        return Ollama(model=settings.llm_model, base_url=settings.ollama_base_url)
    except (TypeError, ValueError) as exc:
        raise ProviderConfigurationError(
            f"Could not construct Ollama LLM({settings.llm_model!r}) against "
            f"{settings.ollama_base_url}: {exc}"
        ) from exc


_EMBEDDING_FACTORIES: Final[dict[str, Callable[[], BaseEmbedding]]] = {
    "openai": _openai_embedding,
    "ollama": _ollama_embedding,
}

_LLM_FACTORIES: Final[dict[str, Callable[[], LLM]]] = {
    "openai": _openai_llm,
    "ollama": _ollama_llm,
}


def _build(
    registry: dict[str, Callable[[], ProviderT]],
    kind: str,
    provider: str | None,
) -> ProviderT:
    selected = provider or resolve_provider()
    factory = registry.get(selected)
    if factory is None:
        raise ProviderConfigurationError(
            f"Unsupported {kind} provider {selected!r}; configured providers are "
            f"{sorted(registry)}. Set LLM_PROVIDER accordingly."
        )
    return factory()


def create_embed_model(provider: str | None = None) -> BaseEmbedding:
    """Instantiate the embedding model for ``provider`` (default: config)."""
    return _build(_EMBEDDING_FACTORIES, "embedding", provider)


def create_llm(provider: str | None = None) -> LLM:
    """Instantiate the chat LLM for ``provider`` (default: config)."""
    return _build(_LLM_FACTORIES, "LLM", provider)
