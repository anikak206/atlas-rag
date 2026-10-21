"""Phase-4: Conversations, messages, hybrid search support.

Revision ID: 0004_phase4_chat_hybrid
Revises: 0003_phase3_vector_metadata
Create Date: 2026-10-21 14:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "0004_phase4_chat_hybrid"
down_revision: Union[str, None] = "0003_phase3_vector_metadata"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_conversations_tenant_id_user_id', 'conversations', ['tenant_id', 'user_id'])
    op.create_index('ix_conversations_tenant_id_updated_at', 'conversations', ['tenant_id', 'updated_at'])
    
    op.create_table(
        'messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.String(length=20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('citations', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_messages_conversation_id_created_at', 'messages', ['conversation_id', 'created_at'])
    op.create_index('ix_messages_tenant_id_conversation_id', 'messages', ['tenant_id', 'conversation_id'])


def downgrade() -> None:
    op.drop_index('ix_messages_tenant_id_conversation_id', table_name='messages')
    op.drop_index('ix_messages_conversation_id_created_at', table_name='messages')
    op.drop_table('messages')
    op.drop_index('ix_conversations_tenant_id_updated_at', table_name='conversations')
    op.drop_index('ix_conversations_tenant_id_user_id', table_name='conversations')
    op.drop_table('conversations')
