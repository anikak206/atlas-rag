"""baseline: pgvector extension, core tables, triggers and RLS policies

Revision ID: 0001_baseline
Revises:
Create Date: 2026-10-21

Design notes
------------
* This is the **baseline** revision, so it materialises the schema directly
  from ``Base.metadata`` (models.py). That guarantees zero drift between the
  ORM models and the deployed schema at project birth. Every revision after
  this one MUST be authored as explicit DDL (``alembic revision --autogenerate``
  plus review) and must never import application code.
* Requires a **live** database (offline ``--sql`` mode cannot render
  ``metadata.create_all``).
* Creating the ``vector`` extension needs a privileged role; the baseline is
  meant to be applied by the migration/owner role, not the runtime app role.
* Row Level Security: policies match ``current_setting('app.tenant_id', true)``,
  which database.py sets at the start of every transaction. A missing GUC
  yields ``NULL`` comparisons -> **no rows visible** (fail closed).
* The runtime role must NOT own these tables, otherwise PostgreSQL lets it
  bypass RLS. Use an owner role for migrations and a separate ``rag_app`` role
  for the API.
"""
from __future__ import annotations

from typing import Final

import models  # noqa: F401  (registers metadata for create_all)
from alembic import op
from database import Base

# revision identifiers, used by Alembic.
revision: str = "0001_baseline"
down_revision: str | None = None
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

#: Tables guarded by RLS and the column each policy matches against.
_RLS_TABLES: Final[tuple[tuple[str, str], ...]] = (
    ("tenants", "id"),
    ("users", "tenant_id"),
    ("documents", "tenant_id"),
    ("document_chunks", "tenant_id"),
    ("audit_logs", "tenant_id"),
)

#: Tables whose updated_at is maintained by a trigger (covers raw SQL writes).
_TRIGGERED_TABLES: Final[tuple[str, ...]] = ("tenants", "users", "documents", "document_chunks")

_SET_UPDATED_AT_SQL: Final[str] = """
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$
"""


def _policy_name(table: str) -> str:
    return f"tenant_isolation_{table}"


def upgrade() -> None:
    bind = op.get_bind()

    # 1. pgvector must exist before the Vector column types are created.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Tables, enums and indexes (baseline only - see module docstring).
    Base.metadata.create_all(bind=bind, checkfirst=True)

    # 3. updated_at triggers so bulk/raw SQL updates stay accurate.
    op.execute(_SET_UPDATED_AT_SQL)
    for table in _TRIGGERED_TABLES:
        op.execute(
            f"CREATE TRIGGER set_{table}_updated_at "
            f"BEFORE UPDATE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION set_updated_at()"
        )

    # 4. Multi-tenant row level security (fail closed when GUC is absent).
    predicate = "current_setting('app.tenant_id', true)::uuid"
    for table, column in _RLS_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {_policy_name(table)} ON {table} "
            f"USING ({column} = {predicate}) "
            f"WITH CHECK ({column} = {predicate})"
        )


def downgrade() -> None:
    # Policies are dropped together with their tables; disable first for clarity.
    for table, _column in _RLS_TABLES:
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    for table in _TRIGGERED_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS set_{table}_updated_at ON {table}")

    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind, checkfirst=True)
    op.execute("DROP FUNCTION IF EXISTS set_updated_at()")

    # The shared 'vector' extension is intentionally left installed.
