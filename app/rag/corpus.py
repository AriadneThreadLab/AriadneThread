"""The whitelisted OSM documentation corpus.

Ingestion is restricted to this explicit list of OSM wiki pages. There is no
crawler and no link following: adding knowledge is a deliberate code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote

from app.rag.contracts import OSM_KNOWLEDGE_DOMAIN

_WIKI_BASE = "https://wiki.openstreetmap.org/wiki/"

#: OSM wiki content is licensed CC-BY-SA for wiki text.
OSM_WIKI_LICENSE = "CC-BY-SA-2.0 (OpenStreetMap wiki)"

#: Persisted on every ingested document; distinguishes wiki text from later sources.
OSM_WIKI_SOURCE_TYPE = "osm_wiki"


@dataclass(frozen=True, slots=True)
class CorpusPage:
    """One approved source document."""

    page: str
    title: str

    @property
    def url(self) -> str:
        return f"{_WIKI_BASE}{self.page}"

    @property
    def domain(self) -> str:
        return OSM_KNOWLEDGE_DOMAIN

    @property
    def api_title(self) -> str:
        """Title form accepted by the MediaWiki API (decoded path segment)."""
        return unquote(self.page)

    @property
    def source_type(self) -> str:
        return OSM_WIKI_SOURCE_TYPE

    @property
    def license(self) -> str:
        return OSM_WIKI_LICENSE


OSM_KNOWLEDGE_CORPUS: tuple[CorpusPage, ...] = (
    CorpusPage(page="Map_Features", title="Map Features"),
    CorpusPage(page="Tag", title="Tag"),
    CorpusPage(page="Tag:leisure%3Dpark", title="Tag: leisure=park"),
    CorpusPage(page="Tag:landuse%3Dgrass", title="Tag: landuse=grass"),
    CorpusPage(page="Tag:natural%3Dwood", title="Tag: natural=wood"),
    CorpusPage(page="Tag:landuse%3Dforest", title="Tag: landuse=forest"),
    CorpusPage(page="Overpass_API/Overpass_QL", title="Overpass API: Overpass QL"),
    CorpusPage(
        page="Overpass_API/Overpass_API_by_Example",
        title="Overpass API: Overpass API by Example",
    ),
)

_ALLOWED_URLS = frozenset(page.url for page in OSM_KNOWLEDGE_CORPUS)
_ALLOWED_API_TITLES = frozenset(page.api_title for page in OSM_KNOWLEDGE_CORPUS)


def is_allowed_source(url: str) -> bool:
    """Whether a URL may be ingested into the knowledge corpus."""
    return url in _ALLOWED_URLS


def is_allowed_api_title(title: str) -> bool:
    """Whether a MediaWiki title belongs to the whitelist."""
    return title in _ALLOWED_API_TITLES


def require_allowed_source(url: str) -> None:
    """Raise if ``url`` is not an approved OSM documentation source."""
    if not is_allowed_source(url):
        from app.core.errors import CorpusWhitelistError

        raise CorpusWhitelistError(f"source URL is not on the OSM knowledge whitelist: {url}")
