"""Alembic environment: metadata target and sync/async DSN wiring.

The runtime application uses an async engine (database.py); migrations run on
the sync psycopg2 engine built from the same validated settings.
"""

from __future__ import annotations

import logging
from typing import Any

import models  # noqa: F401  (importing registers every table on Base.metadata)
from alembic import context
from config import get_settings
from database import Base, create_sync_engine

logger = logging.getLogger("alembic.env")

config = context.config

if config.config_file_name is not None:
    # alembic.ini configures the root logger; keep a compact format.
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s | %(message)s")

target_metadata = Base.metadata


def _connection_url() -> str:
    """Resolve the migration DSN from validated application settings."""
    return get_settings().sync_database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to the database."""
    context.configure(
        url=_connection_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=False,
    )
    with context.begin_transaction():
        context.run_migrations()
    logger.info("Offline migration SQL generated")


def run_migrations_online() -> None:
    """Run migrations against a live database using a sync engine."""
    engine = create_sync_engine(_connection_url())
    connectable: Any = engine

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=False,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()
    logger.info("Online migration completed")


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
