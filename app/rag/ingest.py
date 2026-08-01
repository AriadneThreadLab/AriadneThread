"""Whitelist-only OSM documentation ingestion pipeline.

Fetch → parse → chunk → hash → persist. Embeddings are intentionally deferred
to a later phase.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field

from app.core.errors import CorpusWhitelistError, IngestionError
from app.db.session import Database
from app.rag.corpus import OSM_KNOWLEDGE_CORPUS, CorpusPage, require_allowed_source
from app.rag.mediawiki import MediaWikiClient
from app.rag.persist import (
    KnowledgeRepository,
    PersistResult,
    SqlAlchemyKnowledgeRepository,
    persist_document,
)
from app.rag.prepare import prepare_document

logger = logging.getLogger(__name__)

RepositoryFactory = Callable[[], AbstractAsyncContextManager[KnowledgeRepository]]


@dataclass(frozen=True, slots=True)
class PageIngestOutcome:
    """Per-page ingestion summary (no document body)."""

    title: str
    source_url: str
    status: str
    chunk_count: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class IngestSummary:
    """Corpus-level ingestion summary."""

    outcomes: tuple[PageIngestOutcome, ...] = field(default_factory=tuple)

    @property
    def inserted(self) -> int:
        return sum(1 for item in self.outcomes if item.status == "inserted")

    @property
    def updated(self) -> int:
        return sum(1 for item in self.outcomes if item.status == "updated")

    @property
    def unchanged(self) -> int:
        return sum(1 for item in self.outcomes if item.status == "unchanged")

    @property
    def failed(self) -> int:
        return sum(1 for item in self.outcomes if item.status == "failed")


@dataclass(frozen=True, slots=True)
class IngestConfig:
    """Chunk-size knobs for one ingestion run."""

    max_chars: int = 2000
    min_chars: int = 200
    stop_on_error: bool = False


async def ingest_prepared(
    repository: KnowledgeRepository,
    page: CorpusPage,
    client: MediaWikiClient,
    *,
    config: IngestConfig,
) -> PersistResult:
    """Fetch, prepare and persist one whitelist page via ``repository``."""
    require_allowed_source(page.url)
    fetched = await client.fetch_page(page.api_title)
    prepared = prepare_document(
        page,
        fetched,
        max_chars=config.max_chars,
        min_chars=config.min_chars,
    )
    if prepared.source_url != page.url:
        raise CorpusWhitelistError("prepared source URL drifted from whitelist entry")
    return await persist_document(repository, prepared)


async def ingest_page(
    database: Database,
    client: MediaWikiClient,
    page: CorpusPage,
    *,
    config: IngestConfig,
) -> PersistResult:
    """Ingest one whitelist page inside a single database transaction."""
    async with database.session() as session:
        repository = SqlAlchemyKnowledgeRepository(session)
        return await ingest_prepared(repository, page, client, config=config)


def sqlalchemy_repository_factory(database: Database) -> RepositoryFactory:
    """Build a per-page SQLAlchemy repository factory bound to ``database``."""

    @asynccontextmanager
    async def factory() -> AsyncIterator[KnowledgeRepository]:
        async with database.session() as session:
            yield SqlAlchemyKnowledgeRepository(session)

    return factory


async def ingest_osm_knowledge(
    database: Database,
    client: MediaWikiClient,
    *,
    pages: Sequence[CorpusPage] | None = None,
    config: IngestConfig | None = None,
) -> IngestSummary:
    """Ingest every page on the OSM knowledge whitelist into PostgreSQL."""
    return await ingest_osm_knowledge_with_factory(
        repository_factory=sqlalchemy_repository_factory(database),
        client=client,
        pages=pages,
        config=config,
    )


async def ingest_osm_knowledge_with_factory(
    *,
    repository_factory: RepositoryFactory,
    client: MediaWikiClient,
    pages: Sequence[CorpusPage] | None = None,
    config: IngestConfig | None = None,
) -> IngestSummary:
    """Ingest using a repository factory (SQLAlchemy or in-memory)."""
    corpus = tuple(pages) if pages is not None else OSM_KNOWLEDGE_CORPUS
    resolved = config or IngestConfig()
    outcomes: list[PageIngestOutcome] = []

    for page in corpus:
        require_allowed_source(page.url)
        try:
            async with repository_factory() as repository:
                result = await ingest_prepared(repository, page, client, config=resolved)
            outcomes.append(
                PageIngestOutcome(
                    title=page.title,
                    source_url=page.url,
                    status=result.status,
                    chunk_count=result.chunk_count,
                )
            )
            logger.info(
                "ingest %s %s (%s chunks)",
                result.status,
                page.title,
                result.chunk_count,
            )
        except IngestionError as exc:
            logger.error("ingest failed for %s: %s", page.title, exc.message)
            outcomes.append(
                PageIngestOutcome(
                    title=page.title,
                    source_url=page.url,
                    status="failed",
                    error=exc.message,
                )
            )
            if resolved.stop_on_error:
                break
        except Exception as exc:
            logger.exception("unexpected ingest failure for %s", page.title)
            outcomes.append(
                PageIngestOutcome(
                    title=page.title,
                    source_url=page.url,
                    status="failed",
                    error=str(exc),
                )
            )
            if resolved.stop_on_error:
                break

    return IngestSummary(outcomes=tuple(outcomes))
