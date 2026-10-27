"""Offline tests for Phase-6 async ingestion and agentic RAG.

The suite stays database-free (see ``tests/conftest.py``): the ingestion job body
is exercised with fake SQLAlchemy objects, and the routes run against the real
FastAPI app with the auth/session dependencies overridden.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Self
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from api.deps import get_current_user
from api.routes import documents_async
from api.routes import query as query_routes
from database import get_session
from models import User, UserRole
from services import agentic_rag, async_ingestion, task_registry
from services.async_ingestion import (
    TaskPayloadError,
    build_ingest_request,
    run_ingestion_task,
)
from services.ingestion import DuplicateDocumentError, IngestResult
from services.retrieval import GenerationResult, RetrievedChunk
from tasks.ingestion_tasks import ingest_document_async

# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    task_registry.clear()
    yield
    task_registry.clear()


@pytest.fixture
def user() -> User:
    return User(
        id=uuid4(),
        tenant_id=uuid4(),
        email="intern@example.test",
        hashed_password="not-a-real-hash",
        role=UserRole.INTERN,
        is_active=True,
    )


@pytest.fixture
def app(user: User):
    from main import create_application

    application = create_application()
    application.dependency_overrides[get_current_user] = lambda: user
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


def _job(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": str(uuid4()),
        "tenant_id": str(uuid4()),
        "user_id": str(uuid4()),
        "title": "Leave policy",
        "content": "Full-time employees receive 24 days of paid annual leave.",
        "mime_type": "text/plain",
        "min_role_required": "Intern",
        "metadata": {"format": "text"},
    }
    payload.update(overrides)
    return payload


class _FakeSession:
    def __init__(self, actor: User | None) -> None:
        self.actor = actor
        self.committed = False

    async def get(self, _model: Any, _key: Any) -> User | None:
        return self.actor

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        return None


class _FakeSessionContext:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> _FakeSession:
        return self.session

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _FakeSessionFactory:
    """Callable factory mirroring ``async_sessionmaker``'s usage."""

    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    def __call__(self) -> _FakeSessionContext:
        return _FakeSessionContext(self._session)


class _FakeEngine:
    def __init__(self) -> None:
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


@dataclass
class _FakeLLMResponse:
    text: str


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self._text = text

    def complete(self, _prompt: str) -> _FakeLLMResponse:
        return _FakeLLMResponse(self._text)


class _BrokerDown:
    def apply_async(self, *_args: Any, **_kwargs: Any) -> None:
        from kombu.exceptions import OperationalError

        raise OperationalError("redis://localhost:6379/0 is unreachable")


class _BrokerUp:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def apply_async(self, args: tuple[dict[str, Any]], **kwargs: Any) -> None:
        self.calls.append((args[0], kwargs))


def _chunk(score: float, chunk_id: UUID | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or uuid4(),
        document_id=uuid4(),
        document_title="Handbook",
        chunk_index=0,
        content="Full-time employees receive 24 days of paid annual leave.",
        distance=1.0 - score,
        score=score,
        page_number=4,
        heading_path="Leave",
        min_role_required=UserRole.INTERN,
    )


# ---------------------------------------------------------------------------
# Job payload validation
# ---------------------------------------------------------------------------


def test_build_ingest_request_maps_payload() -> None:
    job = _job(min_role_required="Finance")

    request = build_ingest_request(job)

    assert request.title == "Leave policy"
    assert request.min_role_required is UserRole.FINANCE
    assert request.mime_type == "text/plain"
    assert request.metadata == {"format": "text"}


def test_build_ingest_request_rejects_incomplete_payload() -> None:
    job = _job()
    del job["content"]

    with pytest.raises(TaskPayloadError, match="content"):
        build_ingest_request(job)


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_ingestion(monkeypatch: pytest.MonkeyPatch, user: User):
    engine = _FakeEngine()
    session = _FakeSession(actor=user)
    captured: dict[str, Any] = {}

    async def fake_ingest_text(
        _session: Any, *, tenant_id: UUID, actor: User, payload: Any
    ) -> IngestResult:
        captured.update(tenant_id=tenant_id, actor=actor, payload=payload)
        return IngestResult(
            document_id=uuid4(),
            chunks_created=3,
            elapsed_ms=12.5,
            content_hash="deadbeef",
        )

    monkeypatch.setattr(async_ingestion, "get_engine", lambda: engine)
    monkeypatch.setattr(
        async_ingestion,
        "async_sessionmaker",
        lambda *_args, **_kwargs: _FakeSessionFactory(session),
    )
    monkeypatch.setattr(async_ingestion, "ingest_text", fake_ingest_text)
    return SimpleNamespace(engine=engine, session=session, captured=captured)


async def test_run_ingestion_task_indexes_document(fake_ingestion, user: User) -> None:
    job = _job(tenant_id=str(user.tenant_id), user_id=str(user.id), min_role_required="Intern")

    result = await run_ingestion_task(job)

    assert result["status"] == "completed"
    assert result["chunks_created"] == 3
    assert result["content_hash"] == "deadbeef"
    assert fake_ingestion.session.committed is True
    assert fake_ingestion.engine.disposed is True
    captured = fake_ingestion.captured
    assert captured["tenant_id"] == user.tenant_id
    assert captured["actor"] is user
    assert captured["payload"].content.startswith("Full-time employees")


async def test_run_ingestion_task_reuses_supplied_engine(fake_ingestion, user: User) -> None:
    job = _job(tenant_id=str(user.tenant_id), user_id=str(user.id))

    await run_ingestion_task(
        job,
        engine=fake_ingestion.engine,
        session_factory=_FakeSessionFactory(fake_ingestion.session),
    )

    # The inline worker shares the application's pool; it must not dispose it.
    assert fake_ingestion.engine.disposed is False


async def test_run_ingestion_task_rejects_unknown_user(
    monkeypatch: pytest.MonkeyPatch, user: User
) -> None:
    engine = _FakeEngine()
    monkeypatch.setattr(async_ingestion, "get_engine", lambda: engine)
    monkeypatch.setattr(
        async_ingestion,
        "async_sessionmaker",
        lambda *_args, **_kwargs: _FakeSessionFactory(_FakeSession(actor=None)),
    )

    job = _job(tenant_id=str(user.tenant_id), user_id=str(uuid4()))

    with pytest.raises(TaskPayloadError, match="does not exist"):
        await run_ingestion_task(job)


# ---------------------------------------------------------------------------
# Celery task semantics
# ---------------------------------------------------------------------------


def test_celery_task_returns_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_runner(_job: dict[str, Any]) -> dict[str, Any]:
        return {"status": "completed", "document_id": str(uuid4())}

    monkeypatch.setattr("tasks.ingestion_tasks.run_ingestion_task", fake_runner)

    result = ingest_document_async.run(_job())

    assert result["status"] == "completed"


def test_celery_task_does_not_retry_permanent_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def duplicate_runner(_job: dict[str, Any]) -> dict[str, Any]:
        raise DuplicateDocumentError("Identical content is already indexed")

    monkeypatch.setattr("tasks.ingestion_tasks.run_ingestion_task", duplicate_runner)

    result = ingest_document_async.run(_job())

    assert result["status"] == "failed"
    assert result["retryable"] is False
    assert "already indexed" in result["error"]


# ---------------------------------------------------------------------------
# Async upload routes
# ---------------------------------------------------------------------------


async def test_async_upload_falls_back_to_inline_queue(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, user: User
) -> None:
    queued: list[dict[str, Any]] = []

    async def fake_enqueue(task: dict[str, Any]) -> None:
        queued.append(task)

    async def noop_start(*_args: Any, **_kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(documents_async, "ingest_document_async", _BrokerDown())
    monkeypatch.setattr(documents_async.ingestion_queue, "enqueue", fake_enqueue)
    monkeypatch.setattr(documents_async, "start_inline_worker", noop_start)

    response = await client.post(
        "/api/v1/documents/async/upload",
        files={"file": ("handbook.txt", b"Full-time employees receive 24 days of leave.", "text/plain")},
        data={"min_role_required": "Intern"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["backend"] == "inline"
    assert body["filename"] == "handbook.txt"
    assert len(queued) == 1
    assert queued[0]["task_id"] == body["task_id"]
    assert queued[0]["tenant_id"] == str(user.tenant_id)
    assert task_registry.get(body["task_id"]) is not None


async def test_async_upload_prefers_celery_when_available(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    celery = _BrokerUp()
    monkeypatch.setattr(documents_async, "ingest_document_async", celery)

    response = await client.post(
        "/api/v1/documents/async/upload",
        files={"file": ("handbook.txt", b"Employees accrue leave monthly.", "text/plain")},
    )

    assert response.status_code == 202
    assert response.json()["backend"] == "celery"
    assert celery.calls[0][1]["retry"] is False


async def test_async_upload_rejects_unsupported_type(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/documents/async/upload",
        files={"file": ("payload.bin", b"\x00\x01\x02", "application/octet-stream")},
    )

    assert response.status_code == 415


async def test_async_upload_rejects_empty_file(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/documents/async/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 422


async def test_async_upload_rejects_pdf_without_text(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/documents/async/upload",
        files={
            "file": (
                "scan.pdf",
                b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422


async def test_status_reports_completed_inline_task(
    client: AsyncClient, user: User
) -> None:
    task_id = str(uuid4())
    task_registry.register(task_id, user.tenant_id)
    task_registry.mark_completed(task_id, {"document_id": str(uuid4()), "chunks_created": 2})

    response = await client.get(f"/api/v1/documents/async/status/{task_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["ready"] is True
    assert body["result"]["chunks_created"] == 2
    assert body["backend"] == "inline"


async def test_status_hides_other_tenants_tasks(client: AsyncClient) -> None:
    task_id = str(uuid4())
    task_registry.register(task_id, uuid4())

    response = await client.get(f"/api/v1/documents/async/status/{task_id}")

    assert response.status_code == 404


async def test_status_reports_unreachable_backend(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_state(_task_id: str) -> None:
        return None

    monkeypatch.setattr(documents_async, "_celery_state", no_state)

    response = await client.get(f"/api/v1/documents/async/status/{uuid4()}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unknown"
    assert body["ready"] is False


# ---------------------------------------------------------------------------
# Agentic RAG
# ---------------------------------------------------------------------------


async def test_decompose_query_parses_json_array() -> None:
    llm = _FakeLLM('Sure: ["What is the leave policy?", "Who approves leave?"]')

    assert await agentic_rag.decompose_query("complex question", llm) == [
        "What is the leave policy?",
        "Who approves leave?",
    ]


@pytest.mark.parametrize("model_output", ["not json at all", "[]", '{"not": "a list"}'])
async def test_decompose_query_falls_back_to_original(model_output: str) -> None:
    assert await agentic_rag.decompose_query("simple question", _FakeLLM(model_output)) == [
        "simple question"
    ]


async def test_decompose_query_without_provider_is_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.providers import ProviderConfigurationError

    def no_provider() -> None:
        raise ProviderConfigurationError("OPENAI_API_KEY is not set")

    monkeypatch.setattr(agentic_rag, "create_llm", no_provider)

    assert await agentic_rag.decompose_query("What is X and how does Y work?") == [
        "What is X and how does Y work?"
    ]


async def test_web_fallback_without_key_does_not_invent_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agentic_rag, "get_settings", lambda: SimpleNamespace(tavily_api_key=None)
    )

    result = await agentic_rag.web_search_fallback("unknown topic")

    assert result["source_type"] == "web_fallback"
    assert result["sources"] == []
    assert "TAVILY_API_KEY" in result["answer"]


async def test_web_fallback_uses_tavily_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agentic_rag,
        "get_settings",
        lambda: SimpleNamespace(tavily_api_key=SecretStr("tvly-test")),
    )

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "answer": "External answer",
                "results": [
                    {"title": "Docs", "url": "https://example.test/docs", "content": "Snippet"}
                ],
            }

    class _Client:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_exc: object) -> bool:
            return False

        async def post(self, *_args: Any, **_kwargs: Any) -> _Response:
            return _Response()

    monkeypatch.setattr(agentic_rag.httpx, "AsyncClient", _Client)

    result = await agentic_rag.web_search_fallback("anything")

    assert result["answer"] == "External answer"
    assert result["sources"][0]["url"] == "https://example.test/docs"


async def test_agentic_query_falls_back_when_confidence_is_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sub_questions(_question: str, _llm: Any = None) -> list[str]:
        return ["q"]

    async def no_chunks(*_args: Any, **_kwargs: Any) -> list[RetrievedChunk]:
        return []

    async def fallback(_query: str) -> dict[str, Any]:
        return {"answer": "from the web", "sources": [], "source_type": "web_fallback"}

    monkeypatch.setattr(agentic_rag, "decompose_query", no_sub_questions)
    monkeypatch.setattr(agentic_rag, "_retrieve", no_chunks)
    monkeypatch.setattr(agentic_rag, "web_search_fallback", fallback)

    result = await agentic_rag.agentic_query(
        object(), tenant_id=uuid4(), user=object(), question="q", allowed_roles=[]
    )

    assert result["source_type"] == "web_fallback"
    assert result["confidence"] == 0.0
    assert result["chunks"] == []


async def test_agentic_query_deduplicates_sub_question_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _chunk(score=0.92)
    calls: list[str] = []

    async def two_sub_questions(_question: str, _llm: Any = None) -> list[str]:
        return ["first part", "second part"]

    async def overlapping(*_args: Any, **_kwargs: Any) -> list[RetrievedChunk]:
        calls.append(_kwargs["question"])
        return [shared]

    async def generated(**_kwargs: Any) -> GenerationResult:
        return GenerationResult(
            text="24 days",
            model="fake",
            prompt_tokens=10,
            completion_tokens=2,
            estimated=True,
        )

    monkeypatch.setattr(agentic_rag, "decompose_query", two_sub_questions)
    monkeypatch.setattr(agentic_rag, "_retrieve", overlapping)
    monkeypatch.setattr(agentic_rag, "generate_answer", generated)

    result = await agentic_rag.agentic_query(
        object(), tenant_id=uuid4(), user=object(), question="q", allowed_roles=[]
    )

    assert calls == ["first part", "second part"]
    assert len(result["chunks"]) == 1
    assert result["source_type"] == "internal"
    assert result["confidence"] == pytest.approx(0.92)
    assert result["tokens"]["total_tokens"] == 12
    assert result["sub_questions"] == ["first part", "second part"]


# ---------------------------------------------------------------------------
# /query wiring for agentic requests
# ---------------------------------------------------------------------------


async def test_query_endpoint_returns_agentic_web_fallback(
    app, user: User, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Session:
        async def commit(self) -> None:
            return None

        async def rollback(self) -> None:
            return None

    async def fake_agentic(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "answer": "No confident internal match",
            "sources": [{"type": "web", "title": "T", "url": "https://example.test"}],
            "chunks": [],
            "source_type": "web_fallback",
            "confidence": 0.05,
            "sub_questions": ["a", "b"],
        }

    async def fake_audit(_session: Any, _log: Any) -> int:
        return 17

    app.dependency_overrides[get_session] = lambda: _Session()
    monkeypatch.setattr(query_routes, "agentic_query", fake_agentic)
    monkeypatch.setattr(query_routes, "record_audit", fake_audit)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post(
            "/api/v1/query", json={"question": "unknown topic", "agentic": True}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["source_type"] == "web_fallback"
    assert body["confidence"] == pytest.approx(0.05)
    assert body["sub_questions"] == ["a", "b"]
    assert body["web_sources"][0]["url"] == "https://example.test"
    assert body["sources"] == []
    assert body["audit_id"] == 17
    assert body["applied_role"] == UserRole.INTERN.value
