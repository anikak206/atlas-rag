"""Offline schema validation harness.

Exercises settings validation, ORM metadata, PostgreSQL DDL compilation, RBAC
logic and the FastAPI app factory **without a database**, so it can run in CI
before infrastructure exists:

    python scripts/validate_schema.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FAILURES: list[str] = []


def check(label: str, fn) -> None:
    try:
        result = fn()
        print(f"PASS | {label}" + (f" -> {result}" if result else ""))
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(label)
        print(f"FAIL | {label}: {type(exc).__name__}: {exc}")
        traceback.print_exc()


# --- 1. module imports -----------------------------------------------------
import config as config_mod
import database as db_mod
import models as m
import rbac as rbac_mod


# --- 2. settings -----------------------------------------------------------
def settings_summary() -> str:
    s = config_mod.get_settings()
    assert s.async_database_url.startswith("postgresql+asyncpg://"), s.async_database_url
    assert s.sync_database_url.startswith("postgresql+psycopg2://"), s.sync_database_url
    assert "rag_dev_password" not in s.model_dump_json()
    # production guard must reject the shipped defaults
    try:
        config_mod.Settings(environment="production", debug=False)
        raise AssertionError("production defaults should be rejected")
    except ValueError:
        pass
    # cross-field guard
    try:
        config_mod.Settings(chunk_size=512, chunk_overlap=512)
        raise AssertionError("chunk_overlap >= chunk_size should be rejected")
    except ValueError:
        pass
    return f"dim={s.embedding_dimension} strategy={s.distance_strategy}"


check("config.Settings validation + derived DSNs", settings_summary)


# --- 3. metadata / schema --------------------------------------------------
EXPECTED_TABLES = {"tenants", "users", "documents", "document_chunks", "audit_logs"}


def tables_registered() -> str:
    tables = set(db_mod.Base.metadata.tables)
    assert tables == EXPECTED_TABLES, f"missing={EXPECTED_TABLES - tables} extra={tables - EXPECTED_TABLES}"
    return ", ".join(sorted(tables))


check("all 5 tables registered on Base.metadata", tables_registered)


def chunk_embedding_column() -> str:
    col = m.document_chunks.c["embedding"] if hasattr(m, "document_chunks") else db_mod.Base.metadata.tables["document_chunks"].c["embedding"]
    dtype = col.type
    assert "vector" in str(dtype).lower(), dtype
    assert getattr(dtype, "dim", None) == 1536, getattr(dtype, "dim", None)
    return str(dtype)


check("document_chunks.embedding is Vector(1536)", chunk_embedding_column)


def indexes_present() -> str:
    chunks = db_mod.Base.metadata.tables["document_chunks"]
    hnsw = [i for i in chunks.indexes if i.name == "ix_document_chunks_embedding_hnsw"]
    assert hnsw, [i.name for i in chunks.indexes]
    opts = hnsw[0].dialect_options["postgresql"]
    assert opts["using"] == "hnsw", opts
    assert opts["ops"]["embedding"] == "vector_cosine_ops", opts["ops"]
    assert opts["with"]["m"] == 16 and opts["with"]["ef_construction"] == 64, opts["with"]
    docs = db_mod.Base.metadata.tables["documents"]
    gin = [i for i in docs.indexes if i.dialect_options["postgresql"].get("using") == "gin"]
    assert gin, "missing GIN index on documents.metadata"
    audits = db_mod.Base.metadata.tables["audit_logs"]
    brin = [i for i in audits.indexes if i.dialect_options["postgresql"].get("using") == "brin"]
    assert brin, "missing BRIN index on audit_logs.created_at"
    return "hnsw+gin+brin"


check("ANN/GIN/BRIN indexes configured", indexes_present)


def rbac_columns() -> str:
    docs = db_mod.Base.metadata.tables["documents"]
    col = docs.c["min_role_required"]
    # Native PG enum: inspect the type object (str() would render via the
    # generic dialect and show VARCHAR).
    assert col.type.native_enum is True, col.type
    assert col.type.name == "user_role", col.type.name
    assert col.type.enums == ["Intern", "Finance", "Admin"], col.type.enums
    assert col.nullable is False
    users = db_mod.Base.metadata.tables["users"]
    assert users.c["role"].type.native_enum is True
    # enum stores VALUES not names
    values = {v.value for v in m.UserRole}
    assert values == {"Admin", "Finance", "Intern"}, values
    return f"min_role_required native enum user_role default={col.server_default.arg!r}"


check("RBAC columns + user_role enum values", rbac_columns)


def audit_columns() -> str:
    t = db_mod.Base.metadata.tables["audit_logs"]
    required = [
        "user_id", "tenant_id", "query_text", "response_time",
        "latency", "tokens_used",
    ]
    missing = [c for c in required if c not in t.c]
    assert not missing, f"missing audit columns: {missing}"
    assert t.c["tenant_id"].nullable is False
    assert t.c["query_text"].nullable is False
    assert "ix_audit_logs_tenant_id_created_at" in {i.name for i in t.indexes}
    return f"{len(t.columns)} columns"


check("audit_logs contract (user_id, tenant_id, query, response_time, latency, tokens_used)", audit_columns)


def ddl_compiles() -> str:
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateIndex, CreateTable

    dialect = postgresql.dialect()
    total = 0
    for table in db_mod.Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=dialect))
        assert "CREATE TABLE" in ddl, table.name
        total += len(ddl)
        for index in table.indexes:
            idx_ddl = str(CreateIndex(index).compile(dialect=dialect))
            assert "CREATE INDEX" in idx_ddl or "CREATE UNIQUE INDEX" in idx_ddl, idx_ddl
    # AuditLog PK must be an identity, not a serial trigger
    audit = db_mod.Base.metadata.tables["audit_logs"]
    assert audit.c["id"].identity is not None, "audit_logs.id must use IDENTITY"
    return f"{total} chars of DDL across {len(db_mod.Base.metadata.tables)} tables"


check("PostgreSQL DDL compiles for every table/index", ddl_compiles)


# --- 4. RBAC logic ----------------------------------------------------------
def rbac_logic() -> str:
    assert rbac_mod.role_satisfies(m.UserRole.ADMIN, m.UserRole.INTERN)
    assert rbac_mod.role_satisfies(m.UserRole.FINANCE, m.UserRole.FINANCE)
    assert not rbac_mod.role_satisfies(m.UserRole.INTERN, m.UserRole.FINANCE)
    assert rbac_mod.accessible_min_roles(m.UserRole.FINANCE) == frozenset(
        {m.UserRole.INTERN, m.UserRole.FINANCE}
    )
    assert rbac_mod.highest_role([m.UserRole.INTERN, m.UserRole.ADMIN]) is m.UserRole.ADMIN
    try:
        rbac_mod.require_role(m.UserRole.INTERN, m.UserRole.FINANCE, resource="budget.pdf")
        raise AssertionError("expected AuthorizationError")
    except rbac_mod.AuthorizationError as exc:
        assert "budget.pdf" in str(exc)
    try:
        rbac_mod.assert_same_tenant("t-1", "t-2")
        raise AssertionError("expected TenantIsolationError")
    except rbac_mod.TenantIsolationError:
        pass
    return "hierarchy, matrix, guards"


check("rbac.role_satisfies / require_role / tenant guard", rbac_logic)


# --- 5. session context + app ----------------------------------------------
def session_context_roundtrip() -> str:
    import uuid as uuid_mod

    tid = str(uuid_mod.uuid4())
    with db_mod.session_context(tenant_id=tid, actor_id="u-1", request_id="r-1"):
        assert db_mod.TENANT_ID_CTX.get() == tid
        assert db_mod.ACTOR_ID_CTX.get() == "u-1"
    assert db_mod.TENANT_ID_CTX.get() is None, "context must reset"
    return "bound and reset"


check("database.session_context contextvar roundtrip", session_context_roundtrip)


def app_factory() -> str:
    import main as main_mod

    # FastAPI 0.156 keeps included routers as wrapper objects in ``app.routes``,
    # so the OpenAPI document is the stable, complete view of the API surface.
    routes = {
        f"{method.upper()} {path}"
        for path, operations in main_mod.app.openapi()["paths"].items()
        for method in operations
    }
    prefix = config_mod.get_settings().api_prefix
    expected = {
        "GET /health",
        "GET /ready",
        f"POST {prefix}/auth/signup",
        f"POST {prefix}/auth/login",
        f"POST {prefix}/auth/refresh",
        f"GET {prefix}/auth/me",
        f"POST {prefix}/documents",
        f"POST {prefix}/documents/upload",
        f"GET {prefix}/documents",
        f"GET {prefix}/documents/{{document_id}}",
        f"DELETE {prefix}/documents/{{document_id}}",
        f"POST {prefix}/query",
        f"GET {prefix}/audit",
    }
    missing = expected.difference(routes)
    assert not missing, f"missing routes: {sorted(missing)}"
    return f"{len(routes)} routes, title={main_mod.app.title!r}"


check("main.create_application (FastAPI app boots)", app_factory)


def email_validation() -> str:
    user = m.User(email="  User@Example.COM ", hashed_password="x")
    assert user.email == "user@example.com", user.email
    try:
        m.User(email="not-an-email", hashed_password="x")
        raise AssertionError("expected invalid email to be rejected")
    except ValueError:
        pass
    return "normalised and validated"


check("models.User email normalisation", email_validation)

print()
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILED -> {FAILURES}")
    raise SystemExit(1)
print("RESULT: ALL CHECKS PASSED")
