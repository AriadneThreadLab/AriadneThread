"""The knowledge corpus is an explicit whitelist, not a crawl."""

from __future__ import annotations

from app.rag.contracts import OSM_KNOWLEDGE_DOMAIN, RetrievedPassage
from app.rag.corpus import (
    OSM_KNOWLEDGE_CORPUS,
    OSM_WIKI_SOURCE_TYPE,
    is_allowed_api_title,
    is_allowed_source,
)


def test_corpus_contains_exactly_the_approved_pages():
    assert {page.page for page in OSM_KNOWLEDGE_CORPUS} == {
        "Map_Features",
        "Tag",
        "Tag:leisure%3Dpark",
        "Tag:landuse%3Dgrass",
        "Tag:natural%3Dwood",
        "Tag:landuse%3Dforest",
        "Overpass_API/Overpass_QL",
        "Overpass_API/Overpass_API_by_Example",
    }
    assert len(OSM_KNOWLEDGE_CORPUS) == 8


def test_every_page_is_an_osm_wiki_url_in_the_osm_knowledge_domain():
    for page in OSM_KNOWLEDGE_CORPUS:
        assert page.url.startswith("https://wiki.openstreetmap.org/wiki/")
        assert page.domain == OSM_KNOWLEDGE_DOMAIN
        assert page.source_type == OSM_WIKI_SOURCE_TYPE
        assert is_allowed_api_title(page.api_title)


def test_only_whitelisted_urls_may_be_ingested():
    assert is_allowed_source("https://wiki.openstreetmap.org/wiki/Map_Features")
    assert not is_allowed_source("https://wiki.openstreetmap.org/wiki/Tag:amenity%3Dfuel")
    assert not is_allowed_source("https://example.com/Map_Features")


def test_passage_keeps_its_source_url_for_citation():
    passage = RetrievedPassage(
        content="A park is an area of open space.",
        document_title="Tag: leisure=park",
        source_url=OSM_KNOWLEDGE_CORPUS[2].url,
        score=0.8,
    )
    assert passage.source_url.endswith("Tag:leisure%3Dpark")
    assert passage.section is None
