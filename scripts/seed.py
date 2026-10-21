"""Idempotent demo/seed data loader.

Creates one tenant with three users (one per role) and, optionally, three
RBAC-classified sample documents so the API can be exercised end to end:

    python scripts/seed.py                       # tenant + users only
    python scripts/seed.py --with-documents      # also chunk/embed sample docs

Credentials
-----------
This script must connect with the **migration/owner role** (or any role with
``BYPASSRLS``): it resolves a tenant by slug *before* any tenant identity exists,
which the row-level security policies correctly refuse to expose to the
application role. The script verifies the connection's RLS posture up front and
refuses to run otherwise. Override the target with ``SEED_DATABASE_URL``.

Idempotency
-----------
Tenant by slug, users by (tenant_id, email), documents by content hash - reruns
never duplicate rows. The password is only applied when a user is first created.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import get_settings
from database import session_context
from models import Document, Tenant, TenantStatus, User, UserRole
from schemas.auth import enforce_password_policy
from schemas.documents import DocumentIngestRequest
from services.ingestion import DuplicateDocumentError, content_digest, ingest_text
from services.providers import ProviderConfigurationError
from services.security import hash_password

DEFAULT_TENANT_SLUG = "acme"
DEFAULT_TENANT_NAME = "Acme Corporation"
DEFAULT_PASSWORD = "ChangeMe-Acme-2026"  # documented demo credential

#: Role ladder demoed by the seed: each user may retrieve its own tier and below.
_USER_SPECS: tuple[tuple[str, str, UserRole], ...] = (
    ("admin@acme.test", "Avery Admin", UserRole.ADMIN),
    ("finance@acme.test", "Finley Finance", UserRole.FINANCE),
    ("intern@acme.test", "Indy Intern", UserRole.INTERN),
)

#: (title, min_role_required, body) - deliberately overlapping facts so RBAC
#: differences are observable in query answers.
_SAMPLE_DOCUMENTS: tuple[tuple[str, UserRole, str], ...] = (
    (
        "Employee Handbook - Leave Policy",
        UserRole.INTERN,
        (
            "Full-time employees receive 24 days of paid annual leave per calendar "
            "year, accrued monthly. Interns receive 2 days of leave per month. "
            "Requests must be submitted in the HR portal at least five working days "
            "before the first day of absence. Unused leave may be carried over up to "
            "a maximum of 5 days into the next year. Public holidays are additional "
            "and are listed in the HR portal. Parental leave follows statutory "
            "minimums and must be requested eight weeks in advance."
        ),
    ),
    (
        "FY2026 Operating Budget Summary",
        UserRole.FINANCE,
        (
            "The FY2026 operating budget totals 48.2 million USD, a 6 percent "
            "increase over FY2025. Engineering receives 41 percent, Sales 27 "
            "percent, Operations 18 percent, and G&A 14 percent. The cloud "
            "infrastructure line item is capped at 4.5 million USD and requires "
            "Finance approval for any increase above 10 percent. Hiring assumes 85 "
            "net new positions. Capital expenditure is limited to 9 million USD. "
            "Quarterly reforecasts are due on the tenth business day after quarter "
            "close."
        ),
    ),
    (
        "Security Incident Response Playbook",
        UserRole.ADMIN,
        (
            "On suspected data exfiltration, the on-call engineer must page the "
            "Security Incident Commander within 15 minutes, isolate affected "
            "workloads, and preserve forensic snapshots before remediation. "
            "Customer notification is required within 72 hours of confirmed "
            "personal-data exposure. Root-cause analysis is due within ten business "
            "days. Access keys implicated in an incident must be rotated immediately "
            "and all active sessions revoked. The incident bridge uses a restricted "
            "channel limited to Admin-level responders."
        ),
    ),
)

_RLS_POSTURE_SQL = text(
    """
    SELECT current_user AS role_name,
           COALESCE(
               (SELECT rolbypassrls FROM pg_roles WHERE rolname = current_user),
               false
           ) AS bypasses_rls,
           (
               SELECT count(*)
                 FROM pg_class c
                 JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'tenants'
                  AND c.relkind = 'r'
                  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
                  AND pg_get_userbyid(c.relowner) = current_user
           ) AS owned_tables
    """
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tenant-slug", default=DEFAULT_TENANT_SLUG)
    parser.add_argument("--tenant-name", default=DEFAULT_TENANT_NAME)
    parser.add_argument(
        "--password",
        default=os.environ.get("SEED_PASSWORD", DEFAULT_PASSWORD),
        help="Password for all seeded users (env: SEED_PASSWORD).",
    )
    parser.add_argument(
        "--with-documents",
        action="store_true",
        help="Also ingest the sample documents (requires a configured embedding provider).",
    )
    return parser.parse_args(argv)


async def _assert_owner_connection(session: AsyncSession) -> None:
    """Refuse to seed with a role whose reads the RLS policies rightfully hide."""
    row = (await session.execute(_RLS_POSTURE_SQL)).one()
    if not (row.bypasses_rls or row.owned_tables > 0):
        raise SystemExit(
            f"Seed aborted: role {row.role_name!r} is neither the table owner nor "
            "BYPASSRLS, so the tenant slug lookup is RLS-filtered to zero rows. "
            "Run with migration/owner credentials (or set SEED_DATABASE_URL) and "
            "ensure 'alembic upgrade head' has been applied."
        )


async def _get_or_create_tenant(
    session: AsyncSession, *, slug: str, name: str
) -> tuple[Tenant, bool]:
    """Return ``(tenant, created)``, creating it with a pre-bound GUC if absent."""
    existing = await session.scalar(select(Tenant).where(Tenant.slug == slug))
    if existing is not None:
        return existing, False

    tenant_id = uuid.uuid4()
    with session_context(tenant_id=tenant_id):
        tenant = Tenant(id=tenant_id, name=name, slug=slug, status=TenantStatus.ACTIVE)
        session.add(tenant)
        try:
            await session.flush()
        except IntegrityError:
            # A concurrent seed won the race; reuse its tenant.
            await session.rollback()
            winner = await session.scalar(select(Tenant).where(Tenant.slug == slug))
            if winner is None:  # pragma: no cover - violated constraint was not the slug
                raise
            return winner, False
    return tenant, True


async def _ensure_users(
    session: AsyncSession, *, tenant: Tenant, password: str
) -> tuple[list[User], int]:
    """Create the missing demo users; existing users keep their password and role."""
    hashed_password = await asyncio.to_thread(hash_password, password)
    users: list[User] = []
    created = 0
    with session_context(tenant_id=tenant.id):
        for email, full_name, role in _USER_SPECS:
            existing = await session.scalar(
                select(User).where(User.tenant_id == tenant.id, User.email == email)
            )
            if existing is not None:
                users.append(existing)
                print(f"  = user {email:<22} role={existing.role.value:<8} (exists)")
                continue
            user = User(
                tenant_id=tenant.id,
                email=email,
                hashed_password=hashed_password,
                full_name=full_name,
                role=role,
            )
            session.add(user)
            users.append(user)
            created += 1
            print(f"  + user {email:<22} role={role.value}")
        await session.flush()
    return users, created


async def _seed_documents(session: AsyncSession, *, tenant: Tenant, actor: User) -> int:
    """Ingest the sample documents that are not yet present; returns count created."""
    created = 0
    for title, min_role, body in _SAMPLE_DOCUMENTS:
        digest = content_digest(body)
        with session_context(tenant_id=tenant.id):
            exists = await session.scalar(
                select(Document.id).where(
                    Document.tenant_id == tenant.id,
                    Document.content_hash == digest,
                )
            )
            if exists is not None:
                print(f"  = doc  {title:<42} (exists)")
                continue
            try:
                result = await ingest_text(
                    session,
                    tenant_id=tenant.id,
                    actor=actor,
                    payload=DocumentIngestRequest(
                        title=title,
                        content=body,
                        min_role_required=min_role,
                        source_system="seed",
                    ),
                )
            except DuplicateDocumentError:
                await session.rollback()
                print(f"  = doc  {title:<42} (exists)")
                continue
            await session.commit()
            created += 1
            print(
                f"  + doc  {title:<42} role={min_role.value:<8} "
                f"chunks={result.chunks_created}"
            )
    return created


async def seed(args: argparse.Namespace) -> int:
    """Create the demo tenant, users and optional documents. Returns an exit code."""
    try:
        password = enforce_password_policy(args.password)
    except ValueError as exc:
        print(f"Seed aborted: {exc}", file=sys.stderr)
        return 2

    settings = get_settings()
    database_url = os.environ.get("SEED_DATABASE_URL", settings.async_database_url)
    engine = create_async_engine(database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    users: list[User] = []
    created_users = 0
    created_documents = 0
    try:
        async with session_factory() as session:
            await _assert_owner_connection(session)
            tenant, created_tenant = await _get_or_create_tenant(
                session, slug=args.tenant_slug, name=args.tenant_name
            )
            await session.commit()
            print(
                f"{'Created' if created_tenant else 'Using existing'} tenant "
                f"{tenant.slug!r} ({tenant.id})"
            )

            users, created_users = await _ensure_users(
                session, tenant=tenant, password=password
            )
            await session.commit()

            if args.with_documents:
                actor = next(
                    (user for user in users if user.role is UserRole.ADMIN), None
                )
                if actor is None:  # pragma: no cover - _USER_SPECS guarantees this
                    raise SystemExit("Seed aborted: no Admin user available as actor")
                print("Ingesting sample documents...")
                created_documents = await _seed_documents(
                    session, tenant=tenant, actor=actor
                )
    except ProviderConfigurationError as exc:
        print(f"Seed aborted while embedding documents: {exc}", file=sys.stderr)
        return 3
    finally:
        await engine.dispose()

    print()
    print("Seed complete")
    print(f"  tenant   : {args.tenant_slug}")
    print(f"  users    : {created_users} created, {len(users)} total")
    print(f"  documents: {created_documents} created")
    print("  login    : POST /api/v1/auth/login")
    print("             tenant_slug=<slug>, email=<seeded email>, password=<seed password>")
    print("  accounts : " + ", ".join(f"{user.email} ({user.role.value})" for user in users))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    return asyncio.run(seed(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
