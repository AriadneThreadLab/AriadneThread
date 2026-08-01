"""Idempotent OSM wiki ingestion with mocked HTTP and in-memory persistence."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import UUID

import httpx
import pytest
from app.core.errors import CorpusWhitelistError
from app.rag.chunking import PreparedChunk
from app.rag.corpus import (
    OSM_KNOWLEDGE_CORPUS,
    OSM_WIKI_LICENSE,
    OSM_WIKI_SOURCE_TYPE,
    CorpusPage,
    is_allowed_api_title,
    require_allowed_source,
)
from app.rag.hashing import sha256_text
from app.rag.ingest import IngestConfig, ingest_osm_knowledge_with_factory, ingest_prepared
from app.rag.mediawiki import FetchedWikiPage, MediaWikiClient
from app.rag.persist import InMemoryKnowledgeRepository, persist_document
from app.rag.prepare import PreparedDocument, prepare_document
from app.rag.wikitext import extract_sections, join_sections_for_hash


def _mw_client(pages: dict[str, str]) -> MediaWikiClient:
    def handler(request: httpx.Request) -> httpx.Response:
        title = request.url.params["titles"]
        if title not in pages:
            return httpx.Response(
                200,
                json={"query": {"pages": [{"title": title, "missing": True}]}},
            )
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": [
                        {
                            "pageid": 1,
                            "title": title,
                            "revisions": [{"slots": {"main": {"content": pages[title]}}}],
                        }
                    ]
                }
            },
        )

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        timeout=httpx.Timeout(5.0),
    )
    return MediaWikiClient(
        api_url="https://wiki.openstreetmap.org/w/api.php",
        user_agent="OSM-GeoAgent-test",
        timeout_seconds=5.0,
        max_retries=0,
        client=http,
    )


def _memory_factory(repo: InMemoryKnowledgeRepository):
    @asynccontextmanager
    async def factory():
        yield repo

    return factory


def _prepared(
    page: CorpusPage,
    *,
    body: str = "Public parks use leisure=park.",
    section: str | None = "Description",
) -> PreparedDocument:
    chunk = PreparedChunk(
        section=section,
        content=body,
        chunk_index=0,
        content_hash=sha256_text(body),
    )
    return PreparedDocument(
        domain=page.domain,
        title=page.title,
        source_url=page.url,
        source_type=page.source_type,
        license=page.license,
        retrieved_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        content_hash=sha256_text(body),
        chunks=(chunk,),
    )


def test_whitelist_rejects_unknown_sources():
    with pytest.raises(CorpusWhitelistError):
        require_allowed_source("https://wiki.openstreetmap.org/wiki/Tag:amenity%3Dfuel")
    assert is_allowed_api_title("Tag:leisure=park")
    assert not is_allowed_api_title("Tag:amenity=fuel")


def test_canonical_source_construction_and_attribution_fields():
    page = OSM_KNOWLEDGE_CORPUS[2]
    assert page.api_title == "Tag:leisure=park"
    assert page.url == "https://wiki.openstreetmap.org/wiki/Tag:leisure%3Dpark"
    assert page.source_type == OSM_WIKI_SOURCE_TYPE
    assert page.license == OSM_WIKI_LICENSE


def test_prepare_document_builds_chunks_and_hashes():
    page = OSM_KNOWLEDGE_CORPUS[2]
    wikitext = (
        "Lead about parks.\n\n"
        "== Description ==\n"
        "The tag {{Tag|leisure|park}} marks a public park.\n"
    )
    fetched = FetchedWikiPage(
        title="Tag:leisure=park",
        wikitext=wikitext,
        page_id=1,
        retrieved_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    prepared = prepare_document(page, fetched, max_chars=2000, min_chars=50)
    assert prepared.source_url == page.url
    assert prepared.domain == "osm_knowledge"
    assert prepared.source_type == "osm_wiki"
    assert prepared.license.startswith("CC-BY-SA")
    assert prepared.chunks
    expected_hash = sha256_text(join_sections_for_hash(extract_sections(wikitext)))
    assert prepared.content_hash == expected_hash


async def test_persist_insert_then_unchanged_on_repeat():
    page = OSM_KNOWLEDGE_CORPUS[0]
    repo = InMemoryKnowledgeRepository()
    prepared = _prepared(page)

    first = await persist_document(repo, prepared)
    second = await persist_document(repo, prepared)

    assert first.status == "inserted"
    assert second.status == "unchanged"
    assert second.document_id == first.document_id
    assert len(repo.documents) == 1


async def test_persist_updates_changed_document_and_replaces_chunks():
    page = OSM_KNOWLEDGE_CORPUS[0]
    repo = InMemoryKnowledgeRepository()
    original = _prepared(page, body="Original body about parks.")
    updated = _prepared(page, body="Updated body with landuse=grass notes.")

    first = await persist_document(repo, original)
    second = await persist_document(repo, updated)

    assert first.status == "inserted"
    assert second.status == "updated"
    assert second.document_id == first.document_id
    stored = repo.bodies[UUID(second.document_id)]
    assert stored.chunks[0].content == "Updated body with landuse=grass notes."
    assert len(stored.chunks) == 1


async def test_ingest_pipeline_inserts_whitelist_page():
    page = OSM_KNOWLEDGE_CORPUS[2]
    client = _mw_client(
        {page.api_title: ("A public park.\n\n== Description ==\nUse leisure=park.\n")}
    )
    repo = InMemoryKnowledgeRepository()
    result = await ingest_prepared(
        repo, page, client, config=IngestConfig(max_chars=2000, min_chars=20)
    )
    await client.aclose()

    assert result.status == "inserted"
    assert result.chunk_count >= 1
    assert (page.domain, page.url) in repo.documents


async def test_ingest_corpus_is_idempotent_across_runs():
    page = OSM_KNOWLEDGE_CORPUS[2]
    wikitext = "Lead\n\n== Description ==\nleisure=park marks public parks.\n"
    client = _mw_client({page.api_title: wikitext})
    repo = InMemoryKnowledgeRepository()
    factory = _memory_factory(repo)

    first = await ingest_osm_knowledge_with_factory(
        repository_factory=factory,
        client=client,
        pages=[page],
        config=IngestConfig(min_chars=20),
    )
    second = await ingest_osm_knowledge_with_factory(
        repository_factory=factory,
        client=client,
        pages=[page],
        config=IngestConfig(min_chars=20),
    )
    await client.aclose()

    assert first.inserted == 1
    assert second.unchanged == 1
    assert second.inserted == 0
    assert len(repo.documents) == 1


async def test_changed_wikitext_updates_existing_document():
    page = OSM_KNOWLEDGE_CORPUS[2]
    repo = InMemoryKnowledgeRepository()
    factory = _memory_factory(repo)

    client_v1 = _mw_client({page.api_title: "Version one\n\n== A ==\nOld text.\n"})
    first = await ingest_osm_knowledge_with_factory(
        repository_factory=factory,
        client=client_v1,
        pages=[page],
        config=IngestConfig(min_chars=10),
    )
    await client_v1.aclose()

    client_v2 = _mw_client({page.api_title: "Version two\n\n== A ==\nNew text.\n"})
    second = await ingest_osm_knowledge_with_factory(
        repository_factory=factory,
        client=client_v2,
        pages=[page],
        config=IngestConfig(min_chars=10),
    )
    await client_v2.aclose()

    assert first.inserted == 1
    assert second.updated == 1
    body = next(iter(repo.bodies.values()))
    assert "New text" in body.chunks[0].content


async def test_non_whitelist_page_cannot_be_ingested():
    rogue = CorpusPage(page="Tag:amenity%3Dfuel", title="fuel")
    client = _mw_client({rogue.api_title: "Fuel stations"})
    repo = InMemoryKnowledgeRepository()
    with pytest.raises(CorpusWhitelistError):
        await ingest_prepared(repo, rogue, client, config=IngestConfig(min_chars=10))
    await client.aclose()
