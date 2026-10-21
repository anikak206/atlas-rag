"""Phase-3: Add GIN index on chunk metadata for RBAC pre-filtering.

Revision ID: 0003_phase3_vector_metadata
Revises: 0002_auth_rls
Create Date: 2026-10-21 12:00:00.000000

Adds metadata pre-filtering support and ensures RBAC metadata is stored
in chunk metadata for vector queries.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0003_phase3_vector_metadata"
down_revision: Union[str, None] = "0002_auth_rls"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add GIN index on document_chunks metadata for RBAC pre-filtering."""
    op.create_index(
        "ix_document_chunks_metadata_gin",
        "document_chunks",
        ["metadata"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"metadata": "jsonb_path_ops"},
        if_not_exists=True,
    )


def downgrade() -> None:
    """Remove GIN index on document_chunks metadata."""
    op.drop_index(
        "ix_document_chunks_metadata_gin",
        table_name="document_chunks",
        if_exists=True,
    )
