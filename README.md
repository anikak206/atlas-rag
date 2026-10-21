# Multi-Tenant Enterprise Knowledge RAG Engine

Production-grade RAG platform with **tenant isolation**, **RBAC-aware retrieval** and
**append-only audit logging**.

Stack: Python 3.11+ · FastAPI · SQLAlchemy 2.0 (async) · PostgreSQL 14+ with **pgvector** ·
Alembic · llama-index.

The schema layer (`0001_baseline`) and the application layer (auth, ingestion, retrieval,
audit, API) are both implemented. End-to-end verification against a live PostgreSQL instance
is the remaining operational step — see [§10](#10-verification).

---

## 1. Directory structure

```
knowledge-rag-engine/
├── requirements.txt                  # runtime + dev dependencies
├── pyproject.toml                    # ruff / pytest / mypy configuration
├── .env.example                      # every supported environment variable, documented
├── config.py                         # Pydantic Settings: typed, validated, fail-fast configuration
├── database.py                       # async engine/session, tenant GUC propagation, startup RLS assertion
├── models.py                         # Tenants, Users, Documents, DocumentChunks (Vector), AuditLogs
├── rbac.py                           # role hierarchy + authorization checks (Intern<Finance<Admin)
├── main.py                           # app factory, CORS, request-id middleware, router mounting, probes
│
├── alembic.ini                       # migration tooling config (DSN resolved from config.py)
├── alembic/
│   ├── env.py                        # metadata target + sync psycopg2 engine for migrations
│   ├── script.py.mako                # revision template
│   └── versions/
│       ├── 0001_baseline_pgvector_rls.py    # extension + tables + HNSW + triggers + RLS policies
│       └── 0002_auth_rls.py                 # SECURITY DEFINER login + RLS FORCE adjustment
│
├── api/
│   ├── deps.py                       # session, JWT identity, RBAC guard factory, pagination, audit context
│   └── routes/
│       ├── auth.py                   # signup · login · refresh · me
│       ├── documents.py              # ingest (JSON/multipart) · list · detail · soft delete
│       ├── query.py                  # RAG query with guaranteed audit write
│       └── audit.py                  # Admin audit readback
│
├── schemas/                          # Pydantic contracts
│   ├── common.py                     # pagination envelope, error body
│   ├── auth.py                       # signup/login/refresh/me + password policy
│   ├── documents.py                  # ingest/read/delete contracts
│   └── query.py                      # query/answer/source/audit contracts
│
├── services/
│   ├── security.py                   # bcrypt hashing + JWT issuance/validation
│   ├── auth.py                       # slugify, signup, authenticate, load_user, token issuance
│   ├── providers.py                  # config-driven embedding/LLM factories (openai · ollama)
│   ├── ingestion.py                  # digest → chunk → embed → persist (single transaction)
│   ├── retrieval.py                  # pgvector search + grounded prompt + token accounting
│   └── audit.py                      # audit row builder (pure) + flush-only writer
│
├── tests/                            # 118 offline unit tests (no database required)
│   ├── conftest.py                   # settings cache isolation
│   ├── test_rbac.py                  # hierarchy, min-role visibility, tenant guard
│   ├── test_security.py              # bcrypt + JWT failure modes
│   ├── test_schemas.py               # validation policies, aliases, pagination
│   ├── test_audit.py                 # clamping, truncation, retention, request-id coercion
│   ├── test_providers.py             # provider factories, dimension guard
│   ├── test_ingestion.py             # digests, token estimate, chunker
│   ├── test_retrieval.py             # prompt assembly, usage extraction, refusal path
│   └── test_api_layer.py             # RBAC guard, error mapping, paging, audit context
│
├── scripts/
│   ├── validate_schema.py            # offline smoke test (settings, DDL, RBAC, routes)
│   └── seed.py                       # idempotent demo tenant/users/documents
│
├── Dockerfile                        # non-root runtime image with healthcheck
└── docker-compose.yml                # pgvector/pgvector:pg16 + API service
```

## 2. Requirement → implementation mapping

| Requirement | Implementation |
|---|---|
| Multi-tenant isolation | `tenants` is the root; `users`, `documents`, `document_chunks`, `audit_logs` carry non-null `tenant_id` FKs (`ON DELETE CASCADE`). RLS policies on every table compare `tenant_id` to the `app.tenant_id` GUC, which `database.py` sets at **transaction start** from task-local contextvars. Missing GUC ⇒ **no rows** (fail closed). Every query additionally filters `tenant_id` explicitly (defence in depth). |
| RBAC metadata | `documents.min_role_required` (`Intern` < `Finance` < `Admin`) with a composite index. `rbac.py` centralises the hierarchy; `api/deps.require_role()` guards routes (audit = Admin); retrieval and document listing filter `min_role_required IN accessible_min_roles(role)`, so interns never see Finance/Admin titles. |
| Audit log | `audit_logs`: `user_id`, `tenant_id`, `query_text`, `response_time`, `latency` (ms), `tokens_used`/`prompt_tokens`/`completion_tokens`, plus `request_id`, role/email snapshots, `model_name`, `retrieved_chunk_count`, `success`/`error_code`/`error_message`, `filters` (JSONB), `ip_address`, `user_agent`. The query route writes **exactly one row per request** — success or failure; if the audit write fails, a successful answer is withheld. |
| pgvector schema | `document_chunks.embedding vector(EMBEDDING_DIMENSION)` with **HNSW** index using the operator class for `DISTANCE_STRATEGY`; GIN `jsonb_path_ops` index on `documents.metadata`. Retrieval uses `pgvector`'s `cosine_distance()` in native SQLAlchemy so tenant/RBAC/JSONB filters live in the same SQL statement. |
| Async SQLAlchemy | `get_engine()`, `get_session_factory()`, `get_session()` (dependency, rollback on error), `bind_identity()`/`session_context()`, `init_database()`, `check_connection()`, `dispose_engine()`; `create_sync_engine()` serves Alembic/ops. |
| Auth & identity | JWT HS256 (`iss=rag-engine`, typed access/refresh tokens, `jti`, `iat`/`nbf`/`exp`), bcrypt cost 12 with a 72-byte guard, tenant-scoped login via a **`SECURITY DEFINER`** function, role re-read from the database on every request. |
| RAG pipeline | llama-index `SentenceSplitter` chunking, provider-selected embeddings, cosine search, numbered context prompt with citation instructions, deterministic refusal when nothing is retrievable. |

## 3. Quickstart

```bash
# 1. Environment
cp .env.example .env

# 2. Infrastructure (PostgreSQL 16 + pgvector)
docker compose up -d db

# 3. Dependencies
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 4. Schema (0001 baseline → 0002 auth/RLS)
alembic upgrade head

# 5. Demo data (owner/migration credentials; see §4)
export SEED_DATABASE_URL="postgresql+asyncpg://rag_owner:<password>@localhost:5432/rag_engine"
python scripts/seed.py --with-documents

# 6. Run
uvicorn main:app --reload
# liveness: GET /health   readiness: GET /ready   OpenAPI: GET /docs
```

Full stack: `docker compose up --build`.

### Walkthrough

```bash
BASE=http://localhost:8000/api/v1

# Signup (creates the tenant and its first Admin, returns tokens)
curl -s -X POST $BASE/auth/signup -H 'content-type: application/json' -d '{
  "organization": "Acme Corporation", "tenant_slug": "acme",
  "email": "admin@acme.test", "password": "Sunrise-Cobalt-2026"
}'

# Login
TOKEN=$(curl -s -X POST $BASE/auth/login -H 'content-type: application/json' -d '{
  "tenant_slug": "acme", "email": "admin@acme.test", "password": "Sunrise-Cobalt-2026"
}' | python -c "import json,sys; print(json.load(sys.stdin)['tokens']['access_token'])")

# Ingest (any authenticated member; min_role_required controls retrieval)
curl -s -X POST $BASE/documents -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{
    "title": "Leave Policy",
    "content": "Full-time employees receive 24 days of paid annual leave per year.",
    "min_role_required": "Intern",
    "metadata": {"department": "HR"}
  }'

# Ask a question (RBAC-filtered, audited, token-accounted)
curl -s -X POST $BASE/query -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"question": "How much leave do I get?"}'

# Audit readback (Admin only)
curl -s "$BASE/audit?limit=20" -H "authorization: Bearer $TOKEN"
```

## 4. Database roles (required for RLS to be meaningful)

PostgreSQL lets a **table owner** bypass RLS unless `FORCE ROW LEVEL SECURITY` is set. The
baseline used `FORCE`; revision `0002` relaxes it to plain `ENABLE` because login must read
`tenants`/`users` **before** identity exists. That bypass is narrowed to one audited
`SECURITY DEFINER` function, and startup refuses to run in production when the runtime role
could bypass RLS (`database._verify_rls_enforcement`).

| Role | Purpose | Privileges |
|---|---|---|
| `rag_owner` | applies Alembic migrations, runs `scripts/seed.py` | owns schema objects, `CREATE` on database |
| `rag_app` | runtime API connections | `CONNECT`, `SELECT/INSERT/UPDATE/DELETE` on tables, **not** owner, no `BYPASSRLS` |

```sql
CREATE ROLE rag_app LOGIN PASSWORD '<strong-password>' NOCREATEDB NOSUPERUSER;
GRANT CONNECT ON DATABASE rag_engine TO rag_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rag_app;

-- after `alembic upgrade head`:
ALTER TABLE tenants, users, documents, document_chunks, audit_logs OWNER TO rag_owner;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rag_app;
GRANT EXECUTE ON FUNCTION authenticate_user(TEXT, TEXT) TO rag_app;
```

`0002` already grants `EXECUTE` to `rag_app` when the role exists; the final `GRANT` above is
the idempotent form for environments where the role is created later. The function is
`SECURITY DEFINER` with `SET search_path = public, pg_catalog, pg_temp` and `REVOKE ALL …
FROM PUBLIC`, so search-path hijacking is not possible.

## 5. API surface (mounted under `settings.api_prefix`, default `/api/v1`)

| Method | Path | Access | Purpose |
|---|---|---|---|
| `POST` | `/auth/signup` | public | Create tenant + first Admin, return tokens |
| `POST` | `/auth/login` | public | Tenant-scoped authentication |
| `POST` | `/auth/refresh` | refresh token | New token pair; role re-read from DB |
| `GET` | `/auth/me` | any member | Current user + tenant |
| `POST` | `/documents` | any member | Ingest JSON text document |
| `POST` | `/documents/upload` | any member | Ingest UTF-8 text/JSON/XML file (≤2M chars) |
| `GET` | `/documents` | any member | List documents visible to the caller's role |
| `GET` | `/documents/{id}` | any member | Detail (404 when invisible to the role) |
| `DELETE` | `/documents/{id}` | **Admin** | Soft delete (`deleted_at`), excluded from retrieval |
| `POST` | `/query` | any member | RAG answer with sources + audit row |
| `GET` | `/audit` | **Admin** | Filtered, paginated audit readback |

## 6. Authentication & authorization flow

- **Tokens** — HS256, issuer `rag-engine`; claims `sub`, `tenant_id`, `role`, `type`
  (`access`/`refresh`), `jti`, `iat`, `nbf`, `exp`. `decode_token()` enforces signature,
  algorithm, issuer, expiry, token type **and explicit claim presence**.
- **Role freshness** — `services.auth.load_user()` re-reads the user (and therefore the role)
  from the database on every request; revoked roles apply immediately, even with a valid JWT.
  Refreshes mint tokens with the current role.
- **Login** — `authenticate_user(slug, email)` returns only the columns login needs; the flow
  verifies the password *before* revealing tenant suspension / disabled-account states, and
  all credential failures share one 401 body.
- **Passwords** — ≥12 chars, letter + digit, not in a weak list, ≤128 chars **and** ≤72 UTF-8
  bytes (bcrypt limit); cost 12; hashed/verified in worker threads so requests never block.
- **Signup** — tenant id is generated client-side and bound as the GUC *before* `BEGIN`, so the
  first insert passes its own `WITH CHECK` policy; a slug conflict returns 409 (auto-generated
  slugs retry once with a random suffix).
- **RBAC** — `require_role(min_role, resource=…)` dependency factory; 403 with a message naming
  the resource. Document visibility uses `accessible_min_roles()` in SQL.

## 7. Retrieval & generation

1. Query text is embedded with the configured provider; the vector width is checked against
   `EMBEDDING_DIMENSION` (fails loudly if the model changed without a migration).
2. A single SQL statement joins chunks to documents and filters by `tenant_id`,
   `status='indexed'`, `deleted_at IS NULL`, `min_role_required IN (allowed roles)` and optional
   JSONB `@>` metadata containment, ordering by cosine distance (`ORDER BY embedding <=> :vec`).
3. Results below `RETRIEVAL_MIN_SCORE` are dropped (default 0 keeps this a no-op).
4. Context is numbered `[1]…[n]` with title, chunk index, heading path, page and the access
   level that admitted it; the model is instructed to answer only from context and cite.
5. With **no** retrievable context the pipeline returns a deterministic refusal **without
   calling the LLM** (no tokens, no hallucination).
6. Token usage is read from the provider response when present, otherwise estimated at
   ~4 chars/token and flagged `tokens.estimated=true`.

## 8. Audit guarantees

- Exactly one `audit_logs` row per `/query` request, committed with the response decision.
- If the audit insert/commit fails, the request fails (500) rather than returning an unaudited
  answer; if the pipeline already failed, the original 502/503/500 is preserved and the audit
  failure is logged.
- `latency`/`tokens_used` are clamped non-negative (DB check constraints); error messages are
  truncated to 2,000 chars; user agents to 512.
- Answer text is **not** stored by default; enable `AUDIT_STORE_RESPONSE_SNIPPET=true` for a
  truncated snippet (`AUDIT_RESPONSE_SNIPPET_CHARS`).
- `request_id` correlates API response, response header and audit row; non-UUID client
  correlation ids are replaced inside the UUID column while the original stays in logs/headers.

## 9. Configuration highlights

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai` | Implemented: `openai`, `ollama`. Others require a factory + integration. |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | `text-embedding-3-small` / `1536` | Dimension changes require a migration + full re-index. |
| `LLM_MODEL` | `gpt-4o-mini` | Any chat model the provider exposes. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Used when `LLM_PROVIDER=ollama`. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1024` / `128` | Tokens (llama-index splitter). |
| `RETRIEVAL_TOP_K` / `RETRIEVAL_MIN_SCORE` | `8` / `0.0` | Per-request `top_k` may override (1–50). |
| `AUDIT_STORE_RESPONSE_SNIPPET` | `false` | Privacy default: answers are not persisted. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` / `REFRESH_TOKEN_EXPIRE_DAYS` | `30` / `7` | JWT lifetimes. |

Ollama integrations are optional: `pip install llama-index-llms-ollama
llama-index-embeddings-ollama`. Missing integrations raise `ProviderConfigurationError` with
the exact install hint instead of an ImportError mid-request.

## 10. Verification

Offline (no database needed):

```bash
pytest                      # 118 unit tests
ruff check .                # lint
python scripts/validate_schema.py   # settings, DDL compile, RBAC, route inventory
alembic heads               # -> 0002_auth_rls (head)
```

Live integration checklist (requires PostgreSQL + pgvector):

1. `alembic upgrade head` (applies `0001` → `0002`).
2. `\df authenticate_user` — function exists, owner is the migration role; `rag_app` has
   `EXECUTE` and no table privileges beyond DML.
3. Start the app with `ENVIRONMENT=production` as the **owner** role — startup must abort
   (RLS bypass assertion); start as `rag_app` — it must boot.
4. `python scripts/seed.py --with-documents` (owner credentials).
5. Log in as Intern → `/documents` hides Finance/Admin docs and queries about the budget/incident
   playbook return the no-context refusal with zero tokens.
6. Log in as Finance/Admin → visibility widens accordingly; `/audit` is 403 for Intern/Finance.
7. Confirm one audit row per `/query` (success **and** induced provider failure) and that a second
   tenant's JWT sees zero rows from the first tenant's ids.
8. `DELETE /documents/{id}` as Admin removes the document from retrieval.

## 11. Security & operational invariants

- **Fail-closed RLS** — no `app.tenant_id` ⇒ empty result sets, never cross-tenant leakage.
- **GUC timing** — identity is bound before the transaction begins; `SET LOCAL` semantics keep
  the tenant stable for the whole transaction even after the contextvars are reset.
- **Parameterised GUCs** — `set_config` receives bind parameters; identity values are never
  interpolated into SQL.
- **Startup RLS assertion** — warns in development and hard-fails in production when the
  runtime role owns the tables or has `BYPASSRLS`.
- **Secrets** — `SecretStr` everywhere; production boot fails if `SECRET_KEY`/`POSTGRES_PASSWORD`
  are default, short or reused.
- **Append-only audit** — the application never updates or deletes audit rows; `AuditLog.__repr__`
  excludes prompt/answer text.
- **Immutable migrations** — `0001_baseline` is the only revision importing application models;
  later revisions are explicit DDL (`0002_auth_rls`).
- **Embedding dimension is a schema decision** — changing it requires a migration plus full
  re-indexing; `document_chunks.embedding_model` enables selective re-embeds.

## 12. Schema overview

```
tenants            id(PK) · name · slug(uq) · status · settings(jsonb) · created_at · updated_at
users              id(PK) · tenant_id(FK) · email(uq per tenant) · hashed_password · role ·
                   is_active · is_superuser · last_login_at · timestamps
documents          id(PK) · tenant_id(FK) · title · source_uri · content_hash(uq per tenant) ·
                   status · min_role_required(RBAC) · metadata(jsonb, GIN) · chunk_count ·
                   created_by(FK) · indexed_at · deleted_at · timestamps
document_chunks    id(PK) · tenant_id(FK) · document_id(FK, CASCADE) · chunk_index(uq) ·
                   content · embedding vector(1536) [HNSW] · embedding_model · token_count ·
                   page_number · heading_path · metadata(jsonb) · timestamps
audit_logs         id(BIGINT identity) · request_id · tenant_id · user_id · actor_email ·
                   actor_role · query_text · response_time · latency · tokens_used ·
                   prompt_tokens · completion_tokens · model_name · retrieved_chunk_count ·
                   success · error_code · error_message · response_text · filters(jsonb) ·
                   ip_address · user_agent · created_at [BRIN]
```
