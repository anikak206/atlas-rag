"""Multipart upload validation and text extraction.

The synchronous and asynchronous ingestion routes accept the same formats,
enforce the same size ceiling and must report the same failures, so those rules
live here instead of being duplicated (and inevitably drifting) in each endpoint.

Parsing is deliberately *not* deferred to the worker: an unsupported or empty
file should be rejected while the client is still connected, and only the slow
part of ingestion (chunking + embedding) belongs on a queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from schemas.documents import MAX_CONTENT_CHARS
from services.advanced_parsing import extract_text_auto

__all__ = [
    "MAX_UPLOAD_BYTES",
    "SUPPORTED_EXTENSIONS",
    "TEXTISH_MIME_TYPES",
    "EmptyUploadError",
    "ExtractedUpload",
    "MissingParserError",
    "UnsupportedUploadError",
    "UploadRejectedError",
    "UploadTooLargeError",
    "extract_upload",
]

#: ``text/*`` is always accepted; these non-text MIME types are still readable
#: by the text-oriented extraction pipeline.
TEXTISH_MIME_TYPES: Final[frozenset[str]] = frozenset(
    {"application/json", "application/xml", "application/xhtml+xml"}
)

SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".json", ".xml", ".md"}
)

#: Binary formats occupy more bytes on disk than the text they contain, so an
#: upload may exceed the extracted-text ceiling before parsing. The ceiling is
#: re-checked against the extracted text below.
MAX_UPLOAD_BYTES: Final[int] = MAX_CONTENT_CHARS * 2 + 1000


class UploadRejectedError(ValueError):
    """An upload the client can fix (wrong type, too large, no text)."""


class UnsupportedUploadError(UploadRejectedError):
    """The media type or extension is not one this pipeline can read."""


class UploadTooLargeError(UploadRejectedError):
    """The upload (or the text extracted from it) exceeds the size limit."""


class EmptyUploadError(UploadRejectedError):
    """The upload was empty or produced no extractable text."""


class MissingParserError(UploadRejectedError):
    """A parser dependency is not installed on this server (mapped to 503)."""


@dataclass(frozen=True, slots=True)
class ExtractedUpload:
    """Validated upload ready to be turned into an ingest payload."""

    title: str
    content: str
    mime_type: str
    metadata: dict[str, Any]


def _is_supported_mime(mime: str) -> bool:
    return (
        mime.startswith("text/")
        or mime in TEXTISH_MIME_TYPES
        or "pdf" in mime
        or "word" in mime
        or "officedocument" in mime
        or "powerpoint" in mime
    )


def extract_upload(
    raw: bytes,
    mime_type: str | None,
    filename: str | None,
    *,
    title: str | None = None,
) -> ExtractedUpload:
    """Validate *raw* and extract its text.

    Raises:
        UnsupportedUploadError: Media type/extension is not supported.
        EmptyUploadError: No bytes or no extractable text.
        UploadTooLargeError: Extracted text exceeds ``MAX_CONTENT_CHARS``.
        MissingParserError: The parser for this format is not installed.
        UploadRejectedError: The file is corrupt or cannot be parsed.
    """
    mime = (mime_type or "text/plain").lower()
    lower_name = (filename or "").lower()
    extension_supported = any(
        lower_name.endswith(extension) for extension in SUPPORTED_EXTENSIONS
    )
    if not (_is_supported_mime(mime) or extension_supported):
        raise UnsupportedUploadError(
            f"Unsupported media type {mime!r}; upload PDF, DOCX, PPTX, or text files"
        )
    if not raw:
        raise EmptyUploadError("Empty file")

    try:
        content, metadata = extract_text_auto(raw, mime, filename)
    except ImportError as exc:
        raise MissingParserError(
            f"A parser required for {mime!r} is not installed on this server: {exc}"
        ) from exc
    except UploadRejectedError:
        raise
    except (ValueError, RuntimeError, OSError) as exc:
        # Parser libraries raise their own error types (docx/fitz/pptx); none of
        # them mean the server is broken, they mean the file could not be read.
        raise UploadRejectedError(f"Parsing failed: {exc}") from exc

    content = content.strip()
    if not content:
        raise EmptyUploadError("Uploaded file contains no extractable text")
    if len(content) > MAX_CONTENT_CHARS:
        raise UploadTooLargeError(
            f"Extracted text exceeds the {MAX_CONTENT_CHARS} character limit"
        )

    resolved_title = (title or filename or "").strip()[:1000] or "Untitled upload"
    return ExtractedUpload(
        title=resolved_title,
        content=content,
        mime_type=mime or str(metadata.get("format") or "text/plain"),
        metadata=dict(metadata),
    )
