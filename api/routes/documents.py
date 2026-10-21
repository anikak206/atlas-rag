"""Document routes: ingestion (JSON + multipart), listing, detail, soft delete.

Visibility has two layers everywhere: RLS scopes rows to the tenant, and every
query additionally filters ``min_role_required IN accessible_min_roles(role)``
so a user never even sees titles of documents their role may not read. Failed
ingestions roll back through the session dependency's teardown, so no
half-indexed document survives an error.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Final
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from api.deps import CurrentUser, PageParamsDep, SessionDep, require_role
from models import Document, DocumentStatus, User, UserRole
from rbac import accessible_min_roles
from schemas.common import Page
from schemas.documents import (
    MAX_CONTENT_CHARS,
    DeleteResponse,
    DocumentIngestRequest,
    DocumentRead,
    IngestResponse,
)
from services.ingestion import (
    DuplicateDocumentError,
    EmptyDocumentError,
    IngestionError,
    IngestResult,
    ingest_text,
)
from services.providers import ProviderConfigurationError

logger: Final[logging.Logger] = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

#: Non-``text/*`` uploads still accepted by the text-oriented ingest pipeline.
_TEXTISH_MIME_TYPES: Final[frozenset[str]] = frozenset(
    {"application/json", "application/xml", "application/xhtml+xml"}
)


def _visible_to(user: User) -> ColumnElement[bool]:
    """SQL predicate matching :func:`rbac.accessible_min_roles` for *user*."""
    return Document.min_role_required.in_(list(accessible_min_roles(user.role)))


async def _run_ingestion(
    session: AsyncSession, user: User, payload: DocumentIngestRequest
) -> IngestResponse:
    """Ingest, commit, and project the stored document (shared by both POSTs)."""
    try:
        result: IngestResult = await ingest_text(
            session, tenant_id=user.tenant_id, actor=user, payload=payload
        )
    except DuplicateDocumentError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)
        ) from exc
    except ProviderConfigurationError as exc:
        logger.error("Ingestion aborted, provider not configured: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Document indexing provider is not configured",
        ) from exc
    except IngestionError as exc:
        logger.error("Ingestion failed: %s", exc)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Document ingestion failed"
        ) from exc

    await session.commit()
    document = await session.get(Document, result.document_id)
    if document is None:  # pragma: no cover - defensive
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Document disappeared after commit",
        )
    return IngestResponse(
        document=DocumentRead.model_validate(document),
        chunks_created=result.chunks_created,
        elapsed_ms=result.elapsed_ms,
    )


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a text document (chunk + embed + index)",
)
async def ingest_document(
    payload: DocumentIngestRequest,
    user: CurrentUser,
    session: SessionDep,
) -> IngestResponse:
    return await _run_ingestion(session, user, payload)


@router.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest an uploaded UTF-8 text file",
)
async def upload_document(
    user: CurrentUser,
    session: SessionDep,
    file: Annotated[UploadFile, File(description="UTF-8 text, JSON or XML file")],
    min_role_required: Annotated[UserRole, Form()] = UserRole.INTERN,
    source_system: Annotated[str | None, Form(max_length=64)] = None,
    title: Annotated[str | None, Form(max_length=1000)] = None,
) -> IngestResponse:
    mime = (file.content_type or "text/plain").lower()
    if not (mime.startswith("text/") or mime in _TEXTISH_MIME_TYPES):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported media type {mime!r}; upload UTF-8 text, JSON or XML",
        )

    # Read one byte past the cap so oversize files are rejected without ever
    # buffering them fully (each character occupies at least one byte).
    raw = await file.read(MAX_CONTENT_CHARS + 1)
    if len(raw) > MAX_CONTENT_CHARS:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds the {MAX_CONTENT_CHARS} character limit",
        )
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "File is not valid UTF-8 text",
        ) from exc

    payload = DocumentIngestRequest(
        title=((title or file.filename or "").strip()[:1000] or "Untitled upload"),
        content=content,
        min_role_required=min_role_required,
        source_system=source_system,
        mime_type=mime,
    )
    if not payload.content.strip():
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "Uploaded file is empty"
        )
    return await _run_ingestion(session, user, payload)


@router.get(
    "",
    response_model=Page[DocumentRead],
    summary="List tenant documents visible to your role",
)
async def list_documents(
    user: CurrentUser,
    session: SessionDep,
    page: PageParamsDep,
    doc_status: Annotated[
        DocumentStatus | None, Query(description="Filter by lifecycle status")
    ] = None,
    min_role: Annotated[
        UserRole | None, Query(description="Filter by required role")
    ] = None,
) -> Page[DocumentRead]:
    filters: list[ColumnElement[bool]] = [
        Document.deleted_at.is_(None),
        _visible_to(user),
    ]
    if doc_status is not None:
        filters.append(Document.status == doc_status)
    if min_role is not None:
        filters.append(Document.min_role_required == min_role)

    total = (
        await session.scalar(
            select(func.count()).select_from(Document).where(*filters)
        )
        or 0
    )
    rows = (
        (
            await session.execute(
                select(Document)
                .where(*filters)
                .order_by(Document.created_at.desc(), Document.id.desc())
                .limit(page.limit)
                .offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    return Page.build(
        [DocumentRead.model_validate(row) for row in rows],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/{document_id}", response_model=DocumentRead, summary="Document detail")
async def get_document(
    document_id: UUID, user: CurrentUser, session: SessionDep
) -> DocumentRead:
    # Inaccessible documents read as 404 (never 403) so titles of
    # higher-privilege documents are not confirmed to exist.
    row = await session.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.deleted_at.is_(None),
            _visible_to(user),
        )
    )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return DocumentRead.model_validate(row)


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    summary="Soft-delete a document (Admin)",
)
async def delete_document(
    document_id: UUID,
    admin: Annotated[User, Depends(require_role(UserRole.ADMIN, resource="documents"))],
    session: SessionDep,
) -> DeleteResponse:
    document = await session.scalar(
        select(Document).where(
            Document.id == document_id, Document.deleted_at.is_(None)
        )
    )
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    document.deleted_at = datetime.now(UTC)
    await session.commit()
    return DeleteResponse(id=document.id)
