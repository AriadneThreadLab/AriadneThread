"""Turn a fetched wiki page into a hashed document + chunks (no I/O)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.core.errors import WikiParseError
from app.rag.chunking import PreparedChunk, chunk_sections
from app.rag.corpus import CorpusPage
from app.rag.hashing import sha256_text
from app.rag.mediawiki import FetchedWikiPage
from app.rag.wikitext import extract_sections, join_sections_for_hash


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    """Fully prepared OSM documentation document ready for persistence."""

    domain: str
    title: str
    source_url: str
    source_type: str
    license: str
    retrieved_at: datetime
    content_hash: str
    chunks: tuple[PreparedChunk, ...]


def prepare_document(
    page: CorpusPage,
    fetched: FetchedWikiPage,
    *,
    max_chars: int,
    min_chars: int,
) -> PreparedDocument:
    """Build document and chunk payloads from a whitelist page and API result."""
    sections = extract_sections(fetched.wikitext)
    chunks = chunk_sections(sections, max_chars=max_chars, min_chars=min_chars)
    if not chunks:
        raise WikiParseError(f"no chunks produced for {page.url}")

    document_text = join_sections_for_hash(sections)
    # Prefer the curated corpus title for stable attribution; fall back to API.
    title = page.title or fetched.title

    return PreparedDocument(
        domain=page.domain,
        title=title,
        source_url=page.url,
        source_type=page.source_type,
        license=page.license,
        retrieved_at=fetched.retrieved_at,
        content_hash=sha256_text(document_text),
        chunks=tuple(chunks),
    )
