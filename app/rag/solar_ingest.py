"""Whitelist-only ingest of packaged solar-PV knowledge markdown.

Reads local files listed in ``SOLAR_PV_KNOWLEDGE_CORPUS``. No HTTP, no crawler,
no link following. Scientific numeric calculations are rejected before persist.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from app.core.errors import IngestionError, KnowledgeNumericContentError, MarkdownParseError
from app.rag.chunking import chunk_sections
from app.rag.corpus import (
    SOLAR_PV_KNOWLEDGE_CORPUS,
    SolarCorpusEntry,
    require_allowed_solar_source,
)
from app.rag.hashing import sha256_text
from app.rag.markdown import sections_from_markdown
from app.rag.numeric_guard import reject_numeric_calculations
from app.rag.persist import KnowledgeRepository, PersistResult, persist_document
from app.rag.prepare import PreparedDocument
from app.rag.wikitext import join_sections_for_hash

logger = logging.getLogger(__name__)

SOLAR_DOCUMENTS_DIR = Path(__file__).resolve().parent / "solar_documents"


def packaged_solar_document_path(entry: SolarCorpusEntry) -> Path:
    """Absolute path of one packaged markdown file."""
    return SOLAR_DOCUMENTS_DIR / entry.filename


def prepare_solar_document(
    entry: SolarCorpusEntry,
    markdown: str,
    *,
    retrieved_at: datetime,
    max_chars: int,
    min_chars: int,
) -> PreparedDocument:
    """Chunk a whitelist solar-PV note. Never writes embeddings."""
    require_allowed_solar_source(entry.citation_url)
    reject_numeric_calculations(markdown, source=entry.filename)
    sections = sections_from_markdown(markdown)
    chunks = chunk_sections(sections, max_chars=max_chars, min_chars=min_chars)
    if not chunks:
        raise IngestionError(f"no chunks produced for solar document {entry.slug}")
    return PreparedDocument(
        domain=entry.domain,
        title=entry.title,
        source_url=entry.citation_url,
        source_type=entry.source_type,
        license=entry.license,
        retrieved_at=retrieved_at,
        content_hash=sha256_text(join_sections_for_hash(sections)),
        chunks=tuple(chunks),
    )


async def ingest_solar_knowledge(
    repository: KnowledgeRepository,
    *,
    entries: Sequence[SolarCorpusEntry] | None = None,
    documents_dir: Path | None = None,
    retrieved_at: datetime | None = None,
    max_chars: int = 2000,
    min_chars: int = 200,
) -> tuple[PersistResult, ...]:
    """Persist every packaged solar-PV document. Offline; no network."""
    selected = tuple(entries) if entries is not None else SOLAR_PV_KNOWLEDGE_CORPUS
    root = documents_dir if documents_dir is not None else SOLAR_DOCUMENTS_DIR
    stamp = retrieved_at if retrieved_at is not None else datetime.now(tz=timezone.utc)
    outcomes: list[PersistResult] = []
    for entry in selected:
        path = root / entry.filename
        try:
            markdown = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IngestionError(f"cannot read packaged solar document {path}: {exc}") from exc
        try:
            prepared = prepare_solar_document(
                entry,
                markdown,
                retrieved_at=stamp,
                max_chars=max_chars,
                min_chars=min_chars,
            )
        except (MarkdownParseError, KnowledgeNumericContentError, IngestionError):
            raise
        result = await persist_document(repository, prepared)
        logger.info(
            "solar ingest %s slug=%s chunks=%s",
            result.status,
            entry.slug,
            result.chunk_count,
        )
        outcomes.append(result)
    return tuple(outcomes)
