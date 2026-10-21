"""0002: SECURITY DEFINER login function + RLS FORCE adjustment.

Revision ID: 0002_auth_rls
Revises: 0001_baseline

Why this revision exists
------------------------
Authentication must read ``tenants`` (slug -> id) and ``users`` (credential
row) *before* any tenant identity exists, yet every table is RLS-protected and
fails closed without ``app.tenant_id``. Two options were considered:

1. Relax the policies when the GUC is absent - this weakens isolation exactly
   in the case RLS exists to prevent (application code forgetting to bind
   identity), so it was rejected.
2. Expose one narrowly-scoped ``SECURITY DEFINER`` function returning only the
   fields login needs, executable by the application role and auditable - the
   pattern PostgreSQL ships for pre-authentication lookups.

This revision implements (2). Because ``FORCE ROW LEVEL SECURITY`` subjects even
the table owner (and therefore the function owner) to the policies, FORCE is
relaxed to plain ENABLE here: the owner - migrations, ops scripts and
``authenticate_user`` - bypasses RLS, while the **non-owner** application role
(``rag_app``, README section 4) remains fully subject to it. The application
asserts this posture at startup (``database._verify_rls_enforcement``) and
refuses to boot in production when the runtime role would bypass RLS, so the
relaxation cannot silently become an isolation hole.
"""

from __future__ import annotations

from typing import Final

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_auth_rls"
down_revision: str | None = "0001_baseline"
branch_labels: str | list[str] | None = None
depends_on: str | list[str] | None = None

#: Same set as 0001_baseline._RLS_TABLES (repeated to keep revisions immutable
#: and self-contained - never import across revision files).
_RLS_TABLES: Final[tuple[str, ...]] = (
    "tenants",
    "users",
    "documents",
    "document_chunks",
    "audit_logs",
)

#: Pre-authentication credential lookup used by services.auth.authenticate.
#: Only the columns login needs cross the boundary; ``search_path`` is pinned so
#: the SECURITY DEFINER body cannot be hijacked via search-path manipulation.
_AUTHENTICATE_FUNCTION_SQL: Final[str] = """
    CREATE OR REPLACE FUNCTION authenticate_user(
        p_tenant_slug TEXT,
        p_email TEXT
    )
    RETURNS TABLE (
        p_user_id UUID,
        p_tenant_id UUID,
        p_hashed_password TEXT,
        p_role user_role,
        p_is_active BOOLEAN,
        p_tenant_status tenant_status
    )
    LANGUAGE plpgsql
    STABLE
    SECURITY DEFINER
    SET search_path = public, pg_catalog, pg_temp
    AS $fn$
    BEGIN
        RETURN QUERY
        SELECT u.id,
               u.tenant_id,
               u.hashed_password,
               u.role,
               u.is_active,
               t.status
          FROM users u
          JOIN tenants t ON t.id = u.tenant_id
         WHERE t.slug = lower(p_tenant_slug)
           AND u.email = lower(p_email);
    END;
    $fn$
"""

#: The application role is created by ops SQL (README section 4), which may run
#: after migrations; grant conditionally so this revision never fails on a
#: database where it does not exist yet.
_GRANT_IF_ROLE_EXISTS_SQL: Final[str] = """
    DO $grant$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_app') THEN
            GRANT EXECUTE ON FUNCTION authenticate_user(TEXT, TEXT) TO rag_app;
        END IF;
    END
    $grant$
"""


def upgrade() -> None:
    # Relax FORCE so the table owner can serve the SECURITY DEFINER login
    # function and ops/seed scripts; ENABLE keeps the app role fully guarded.
    for table in _RLS_TABLES:
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")

    op.execute(_AUTHENTICATE_FUNCTION_SQL)
    op.execute("REVOKE ALL ON FUNCTION authenticate_user(TEXT, TEXT) FROM PUBLIC")
    op.execute(_GRANT_IF_ROLE_EXISTS_SQL)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS authenticate_user(TEXT, TEXT)")
    for table in _RLS_TABLES:
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
