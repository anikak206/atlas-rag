"""Shared API contracts: pagination envelope and error bodies."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field, computed_field

ItemT = TypeVar("ItemT")

#: Hard ceilings enforced by the pagination dependency in ``api/deps.py``.
MAX_PAGE_LIMIT: int = 100
DEFAULT_PAGE_LIMIT: int = 20


class Page(BaseModel, Generic[ItemT]):
    """Uniform pagination envelope for every list endpoint."""

    items: list[ItemT]
    total: int = Field(ge=0, description="Total rows matching the filter")
    limit: int = Field(ge=1, le=MAX_PAGE_LIMIT)
    offset: int = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_more(self) -> bool:
        """True when another page exists after this one."""
        return self.offset + len(self.items) < self.total

    @classmethod
    def build(cls, items: list[ItemT], *, total: int, limit: int, offset: int) -> Page[ItemT]:
        """Construct a page from a fetched slice plus the matching total count."""
        return cls(items=items, total=total, limit=limit, offset=offset)


class ErrorResponse(BaseModel):
    """OpenAPI documentation model for non-2xx responses."""

    detail: str
    request_id: str | None = None
