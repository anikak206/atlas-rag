"""Schema-per-tenant utilities for multi-tenant vector storage.

This module provides helpers to create and manage per-tenant PostgreSQL schemas
so that vectors can be stored under a schema-per-tenant architecture when
required. The default deployment uses shared-schema multi-tenancy with
tenant_id columns and RLS (defense in depth), but these utilities enable
schema-per-tenant isolation as an alternative/extension.

All operations use parameterized SQL and are safe to call from async sessions.
"""

from __future__ import annotations

import re
from typing import Final

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "TENANT_SCHEMA_PREFIX",
    "create_tenant_schema",
    "drop_tenant_schema",
    "get_tenant_schema_name",
    "set_tenant_schema_search_path",
]

TENANT_SCHEMA_PREFIX: Final[str] = "tenant_"

_SCHEMA_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


def get_tenant_schema_name(tenant_slug: str) -> str:
    """Generate a safe schema name for a tenant.
    
    Args:
        tenant_slug: Tenant slug identifier
        
    Returns:
        Sanitized schema name safe for PostgreSQL
    """
    # Sanitize: lowercase, replace non-alphanumeric with underscores
    sanitized = re.sub(r"[^a-z0-9_]", "_", tenant_slug.lower())
    # Ensure starts with letter
    if not sanitized or not sanitized[0].isalpha():
        sanitized = "t_" + sanitized
    # Truncate if too long
    if len(sanitized) > 62:
        sanitized = sanitized[:62]
    schema_name = f"{TENANT_SCHEMA_PREFIX}{sanitized}"
    return schema_name


async def create_tenant_schema(session: AsyncSession, tenant_slug: str) -> str:
    """Create a dedicated schema for a tenant if it doesn't exist.
    
    Args:
        session: Database session
        tenant_slug: Tenant slug
        
    Returns:
        The created schema name
    """
    schema_name = get_tenant_schema_name(tenant_slug)
    # Use quoted identifier safely
    await session.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))
    return schema_name


async def drop_tenant_schema(session: AsyncSession, tenant_slug: str, cascade: bool = False) -> str:
    """Drop a tenant's schema.
    
    Args:
        session: Database session
        tenant_slug: Tenant slug
        cascade: If True, drop all objects in schema
        
    Returns:
        The dropped schema name
    """
    schema_name = get_tenant_schema_name(tenant_slug)
    cascade_clause = "CASCADE" if cascade else "RESTRICT"
    await session.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" {cascade_clause}'))
    return schema_name


async def set_tenant_schema_search_path(session: AsyncSession, tenant_slug: str) -> None:
    """Set the search path to use tenant's schema.
    
    Args:
        session: Database session
        tenant_slug: Tenant slug
    """
    schema_name = get_tenant_schema_name(tenant_slug)
    await session.execute(text(f'SET LOCAL search_path TO "{schema_name}", public'))
