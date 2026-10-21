"""RBAC primitives: role hierarchy, per-resource authorization and tenant checks.

The database stores *metadata* (``documents.min_role_required``); this module
is the single place that decides whether an authenticated actor's role satisfies
that metadata. Keep it dependency-free so API handlers, services and tests can
import it without side effects.
"""

from __future__ import annotations

from typing import Final
from uuid import UUID

from models import UserRole

#: Explicit privilege ranking; lower number = less privileged.
ROLE_RANK: Final[dict[UserRole, int]] = {
    UserRole.INTERN: 0,
    UserRole.FINANCE: 1,
    UserRole.ADMIN: 2,
}

#: Hierarchical ordering (least -> most privileged).
ROLE_HIERARCHY: Final[tuple[UserRole, ...]] = (
    UserRole.INTERN,
    UserRole.FINANCE,
    UserRole.ADMIN,
)


class AuthorizationError(PermissionError):
    """Raised when a role does not satisfy a resource's minimum requirement."""

    def __init__(self, *, actual: UserRole, required: UserRole, resource: str) -> None:
        self.actual = actual
        self.required = required
        self.resource = resource
        super().__init__(
            f"Role '{actual.value}' is insufficient for {resource}: "
            f"requires '{required.value}' or higher."
        )


class TenantIsolationError(PermissionError):
    """Raised when an actor acts on a resource belonging to another tenant."""

    def __init__(self, *, actor_tenant: str | UUID, resource_tenant: str | UUID) -> None:
        self.actor_tenant = str(actor_tenant)
        self.resource_tenant = str(resource_tenant)
        super().__init__(
            "Cross-tenant access denied: actor tenant "
            f"{self.actor_tenant} != resource tenant {self.resource_tenant}."
        )


def role_rank(role: UserRole) -> int:
    """Return the numeric privilege rank of *role*.

    Raises:
        ValueError: If *role* is not a known role (defensive against stale DB values).
    """
    try:
        return ROLE_RANK[role]
    except KeyError as exc:
        raise ValueError(f"Unknown role {role!r}; expected one of {[r.value for r in UserRole]}") from exc


def role_satisfies(actual: UserRole, required: UserRole) -> bool:
    """Return ``True`` when *actual* is at least as privileged as *required*."""
    return role_rank(actual) >= role_rank(required)


def require_role(actual: UserRole, required: UserRole, *, resource: str) -> None:
    """Raise :class:`AuthorizationError` unless *actual* satisfies *required*.

    Args:
        actual: Role held by the authenticated actor.
        required: Minimum role demanded by the resource (``min_role_required``).
        resource: Human-readable resource label used in the error message.

    Raises:
        AuthorizationError: When the actor's role is too low.
    """
    if not role_satisfies(actual, required):
        raise AuthorizationError(actual=actual, required=required, resource=resource)


def assert_same_tenant(
    actor_tenant_id: str | UUID,
    resource_tenant_id: str | UUID,
    *,
    resource: str = "resource",
) -> None:
    """Raise :class:`TenantIsolationError` on cross-tenant access attempts."""
    if str(actor_tenant_id) != str(resource_tenant_id):
        raise TenantIsolationError(
            actor_tenant=actor_tenant_id,
            resource_tenant=resource_tenant_id,
        )


def accessible_min_roles(actual: UserRole) -> frozenset[UserRole]:
    """Return every ``min_role_required`` value the actor is allowed to see.

    Useful for building explicit ``WHERE min_role_required IN (...)`` clauses as
    a defence-in-depth layer on top of PostgreSQL RLS.
    """
    return frozenset(role for role in UserRole if role_satisfies(actual, role))


def highest_role(roles: list[UserRole]) -> UserRole:
    """Return the most privileged role in *roles*.

    Raises:
        ValueError: When *roles* is empty.
    """
    if not roles:
        raise ValueError("highest_role() requires at least one role")
    return max(roles, key=role_rank)
