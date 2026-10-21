"""Multi-turn chat service with contextual conversation and question rephrasing."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models import Conversation, Message
from services.providers import create_llm


async def create_conversation(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
    title: str = "New Chat",
) -> Conversation:
    """Create a new conversation."""
    conversation = Conversation(
        tenant_id=tenant_id,
        user_id=user_id,
        title=title,
    )
    session.add(conversation)
    await session.flush()
    return conversation


async def get_conversation(
    session: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Conversation | None:
    """Get conversation with messages."""
    result = await session.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(
            Conversation.id == conversation_id,
            Conversation.tenant_id == tenant_id,
            Conversation.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def get_user_conversations(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
    limit: int = 50,
) -> list[Conversation]:
    """Get all conversations for a user."""
    result = await session.execute(
        select(Conversation)
        .where(
            Conversation.tenant_id == tenant_id,
            Conversation.user_id == user_id,
            Conversation.is_active == True,
        )
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def add_message(
    session: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    tenant_id: uuid.UUID,
    role: str,
    content: str,
    citations: dict[str, Any] | None = None,
    tokens_used: int | None = None,
) -> Message:
    """Add message to conversation."""
    message = Message(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        role=role,
        content=content,
        citations=citations,
        tokens_used=tokens_used,
    )
    session.add(message)
    await session.flush()
    return message


async def rephrase_question_with_context(
    question: str,
    messages: list[Message],
    llm=None,
) -> str:
    """Rephrase question to be standalone using conversation history."""
    if not messages:
        return question
    
    # Get last 6 messages for context
    recent = messages[-6:] if len(messages) > 6 else messages
    history = "\n".join([f"{m.role}: {m.content[:200]}" for m in recent])
    
    model = llm if llm is not None else create_llm()
    prompt = f"""Given the conversation history below, rephrase the follow-up question to be a standalone question that can be understood without the conversation context.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""
    
    try:
        import asyncio
        response = await asyncio.to_thread(model.complete, prompt)
        rephrased = (response.text or "").strip()
        return rephrased if rephrased else question
    except Exception:  # noqa: BLE001
        return question


def get_conversation_context(messages: list[Message], max_messages: int = 10) -> str:
    """Build context string from conversation history."""
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    return "\n\n".join([f"{m.role.upper()}: {m.content}" for m in recent])
