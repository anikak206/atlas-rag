"""RBAC hierarchy, min-role accessibility and tenant isolation."""

from __future__ import annotations

from uuid import uuid4

import pytest

from models import UserRole
from rbac import (
    ROLE_HIERARCHY,
    AuthorizationError,
    TenantIsolationError,
    accessible_min_roles,
    assert_same_tenant,
    highest_role,
    require_role,
    role_rank,
    role_satisfies,
)


@pytest.mark.parametrize("actual", ROLE_HIERARCHY)
@pytest.mark.parametrize("required", ROLE_HIERARCHY)
def test_role_satisfies_matrix(actual: UserRole, required: UserRole) -> None:
    assert role_satisfies(actual, required) is (
        role_rank(actual) >= role_rank(required)
    )


def test_role_ranks_are_strictly_ordered() -> None:
    ranks = [role_rank(role) for role in ROLE_HIERARCHY]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


def test_intern_cannot_read_finance_or_admin_documents() -> None:
    intern_visible = accessible_min_roles(UserRole.INTERN)
    assert intern_visible == frozenset({UserRole.INTERN})
    assert not role_satisfies(UserRole.INTERN, UserRole.FINANCE)
    assert not role_satisfies(UserRole.INTERN, UserRole.ADMIN)


def test_finance_reads_intern_and_finance_but_not_admin() -> None:
    assert accessible_min_roles(UserRole.FINANCE) == frozenset(
        {UserRole.INTERN, UserRole.FINANCE}
    )
    with pytest.raises(AuthorizationError):
        require_role(UserRole.FINANCE, UserRole.ADMIN, resource="audit logs")


def test_admin_reads_everything() -> None:
    assert accessible_min_roles(UserRole.ADMIN) == frozenset(UserRole)


def test_require_role_allows_equal_rank_and_names_resource() -> None:
    require_role(UserRole.FINANCE, UserRole.FINANCE, resource="budget")
    with pytest.raises(AuthorizationError, match="budget") as exc_info:
        require_role(UserRole.INTERN, UserRole.FINANCE, resource="budget")
    assert exc_info.value.actual is UserRole.INTERN
    assert exc_info.value.required is UserRole.FINANCE


def test_assert_same_tenant_accepts_equivalent_identifiers() -> None:
    tenant_id = uuid4()
    assert_same_tenant(tenant_id, str(tenant_id), resource="document")
    assert_same_tenant(str(tenant_id), tenant_id, resource="document")


def test_assert_same_tenant_rejects_cross_tenant_access() -> None:
    actor_tenant, resource_tenant = uuid4(), uuid4()
    with pytest.raises(TenantIsolationError) as exc_info:
        assert_same_tenant(actor_tenant, resource_tenant, resource="document")
    assert exc_info.value.actor_tenant == str(actor_tenant)
    assert exc_info.value.resource_tenant == str(resource_tenant)


def test_highest_role_returns_the_most_privileged() -> None:
    assert highest_role([UserRole.INTERN, UserRole.ADMIN, UserRole.FINANCE]) is UserRole.ADMIN
    assert highest_role([UserRole.INTERN]) is UserRole.INTERN
    with pytest.raises(ValueError):
        highest_role([])
