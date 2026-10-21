"""Shared fixtures for the offline unit suite.

The suite is deliberately database-free: everything that requires PostgreSQL
(RLS policies, vector search, audit persistence) is covered by the integration
checklist in the README instead. Environment overrides go through
``settings_env`` so cached :class:`config.Settings` instances never leak
between tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from config import get_settings


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., None]]:
    """Return a callable applying env overrides to a freshly built Settings.

    Usage::

        settings_env(AUDIT_STORE_RESPONSE_SNIPPET="true")
        assert get_settings().audit_store_response_snippet is True
    """

    def _apply(**overrides: str) -> None:
        for key, value in overrides.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()

    yield _apply
    get_settings.cache_clear()
