"""Environment-driven application settings (Pydantic Settings v2).

All runtime configuration enters the process through environment variables or
an optional ``.env`` file located next to this module. No other module reads
``os.environ`` directly.

Design notes
------------
* Secrets are modelled as :class:`pydantic.SecretStr` so they are redacted in
  logs, ``repr()`` and ``model_dump()``.
* Connection strings are derived (``postgresql+asyncpg://`` for the runtime
  async engine, ``postgresql+psycopg2://`` for Alembic) instead of being
  duplicated across environments.
* Cross-field and environment-specific invariants are enforced centrally in
  :meth:`Settings._cross_field_checks`, so misconfiguration fails at boot
  rather than at first request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final, Literal
from urllib.parse import quote_plus, urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EnvironmentName = Literal["development", "staging", "production", "test"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DistanceStrategy = Literal["cosine", "l2", "inner_product"]
# Providers with live factories in services/providers.py. Adding a provider
# means: (1) extend this Literal, (2) register embedding + LLM factories, (3) add
# its credentials/settings here. The factory raises a clear configuration error
# for integrations that are not installed rather than failing mid-request.
LLMProvider = Literal["openai", "ollama"]

#: Embedding dimensions supported by the schema (see models.DocumentChunk).
SUPPORTED_EMBEDDING_DIMENSIONS: Final[frozenset[int]] = frozenset(
    {384, 768, 1024, 1536, 3072}
)

#: Values that must never reach a production deployment.
INSECURE_SECRET_VALUES: Final[frozenset[str]] = frozenset(
    {
        "change-me",
        "changeme",
        "secret",
        "dev-secret",
        "dev-insecure-secret-change-me",
        "password",
        "rag_dev_password",
    }
)

_SUPPORTED_DSN_SCHEMES: Final[frozenset[str]] = frozenset({"postgres", "postgresql"})


def _with_driver(database_url: str, driver: str) -> str:
    """Return ``database_url`` with its SQLAlchemy driver rewritten.

    Accepts ``postgres://``, ``postgresql://`` or ``postgresql+<driver>://`` and
    returns ``postgresql+<driver>://``.

    Raises:
        ValueError: If the URL scheme is not a PostgreSQL scheme.
    """
    parsed = urlparse(database_url)
    scheme = (parsed.scheme or "").split("+", 1)[0]
    if scheme not in _SUPPORTED_DSN_SCHEMES:
        raise ValueError(
            f"Unsupported database URL scheme {parsed.scheme!r}; "
            "expected 'postgres://' or 'postgresql://'."
        )
    return parsed._replace(scheme=f"postgresql+{driver}").geturl()


class Settings(BaseSettings):
    """Typed, validated, environment-backed application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = Field(default="Multi-Tenant Enterprise Knowledge RAG Engine")
    api_prefix: str = Field(default="/api/v1", pattern=r"^/api/v[0-9]+$")
    environment: EnvironmentName = "development"
    debug: bool = False
    log_level: LogLevel = "INFO"
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------
    postgres_user: str = Field(default="rag", min_length=1)
    postgres_password: SecretStr = Field(default=SecretStr("rag_dev_password"))
    postgres_host: str = Field(default="localhost", min_length=1)
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = Field(default="rag_engine", min_length=1, max_length=63)
    database_url: str | None = Field(
        default=None,
        description="Optional full DSN override; wins over POSTGRES_* components. "
        "Percent-encode any special characters in the password.",
    )

    db_pool_size: int = Field(default=10, ge=1, le=100)
    db_max_overflow: int = Field(default=20, ge=0, le=200)
    db_pool_timeout: float = Field(default=30.0, gt=0.0, le=300.0)
    db_pool_recycle: int = Field(default=1800, ge=300, le=86400)
    db_echo: bool = False

    # ------------------------------------------------------------------
    # pgvector / chunking / retrieval
    # ------------------------------------------------------------------
    pgvector_extension: str = Field(default="vector", pattern=r"^[a-z_]+$")
    embedding_dimension: int = Field(default=1536)
    embedding_model: str = Field(default="text-embedding-3-small", min_length=1)
    distance_strategy: DistanceStrategy = "cosine"
    hnsw_m: int = Field(default=16, ge=2, le=128)
    hnsw_ef_construction: int = Field(default=64, ge=4, le=1000)
    chunk_size: int = Field(default=1024, ge=128, le=8192)
    chunk_overlap: int = Field(default=128, ge=0)
    retrieval_top_k: int = Field(default=8, ge=1, le=100)
    retrieval_min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # Auth / JWT (RBAC)
    # ------------------------------------------------------------------
    secret_key: SecretStr = Field(default=SecretStr("dev-insecure-secret-change-me"))
    jwt_algorithm: str = Field(default="HS256", pattern=r"^HS(256|384|512)$")
    access_token_expire_minutes: int = Field(default=60, ge=5, le=1440)
    refresh_token_expire_days: int = Field(default=14, ge=1, le=90)

    # ------------------------------------------------------------------
    # LLM (llama-index)
    # ------------------------------------------------------------------
    llm_provider: LLMProvider = "openai"
    llm_model: str = Field(default="gpt-4o-mini", min_length=1)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=1, le=8192)
    openai_api_key: SecretStr | None = None
    request_timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for a local Ollama server (LLM_PROVIDER=ollama).",
    )

    # ------------------------------------------------------------------
    # External web search (agentic fallback)
    # ------------------------------------------------------------------
    tavily_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Tavily API key enabling the agentic web-search fallback. When unset, "
            "a low-confidence query reports that external search is not configured."
        ),
    )

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------
    audit_enabled: bool = True
    audit_slow_query_ms: float = Field(default=1000.0, gt=0.0)
    audit_store_response_snippet: bool = False
    audit_response_snippet_chars: int = Field(default=512, ge=0, le=4096)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("embedding_dimension")
    @classmethod
    def _validate_embedding_dimension(cls, value: int) -> int:
        """Reject dimensions the vector index/schema cannot support."""
        if value not in SUPPORTED_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"embedding_dimension={value} is not supported; "
                f"choose one of {sorted(SUPPORTED_EMBEDDING_DIMENSIONS)}. "
                "Changing the dimension later requires a schema migration and re-indexing."
            )
        return value

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, value: str | None) -> str | None:
        """Normalise an optional DSN override and validate its scheme."""
        if value is None or not value.strip():
            return None
        candidate = value.strip()
        scheme = (urlparse(candidate).scheme or "").split("+", 1)[0]
        if scheme not in _SUPPORTED_DSN_SCHEMES:
            raise ValueError(
                f"database_url has unsupported scheme {scheme!r}; expected PostgreSQL."
            )
        return candidate

    @field_validator("allowed_origins")
    @classmethod
    def _validate_allowed_origins(cls, value: list[str]) -> list[str]:
        """Normalise CORS origins (strip trailing slashes, drop blanks)."""
        cleaned = [origin.strip().rstrip("/") for origin in value if origin and origin.strip()]
        if not cleaned:
            raise ValueError("allowed_origins must contain at least one origin.")
        return cleaned

    @model_validator(mode="after")
    def _cross_field_checks(self) -> Settings:
        """Enforce invariants that span multiple fields."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})."
            )
        if self.api_prefix.rstrip("/") == "/api":
            raise ValueError("api_prefix must include a version segment, e.g. '/api/v1'.")

        if self.environment == "production":
            problems: list[str] = []
            if self.debug:
                problems.append("DEBUG must be false in production")
            secret = self.secret_key.get_secret_value()
            if secret in INSECURE_SECRET_VALUES or len(secret) < 32:
                problems.append("SECRET_KEY must be a unique value of at least 32 characters")
            password = self.postgres_password.get_secret_value()
            if password in INSECURE_SECRET_VALUES or len(password) < 12:
                problems.append("POSTGRES_PASSWORD must be a unique value of at least 12 characters")
            if not self.audit_enabled:
                problems.append("AUDIT_ENABLED must remain true in production")
            if problems:
                raise ValueError(
                    "Invalid production configuration: " + "; ".join(problems) + "."
                )
        return self

    # ------------------------------------------------------------------
    # Derived connection strings
    # ------------------------------------------------------------------
    @property
    def async_database_url(self) -> str:
        """DSN for the runtime async engine (``postgresql+asyncpg://``)."""
        if self.database_url:
            return _with_driver(self.database_url, "asyncpg")
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password.get_secret_value())
        return (
            f"postgresql+asyncpg://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """DSN for sync tooling such as Alembic (``postgresql+psycopg2://``)."""
        if self.database_url:
            return _with_driver(self.database_url, "psycopg2")
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password.get_secret_value())
        return (
            f"postgresql+psycopg2://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def is_production(self) -> bool:
        """True when running in the ``production`` environment."""
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton (cached)."""
    return Settings()
