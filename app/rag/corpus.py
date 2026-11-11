"""The whitelisted knowledge corpora.

Ingestion is restricted to these explicit lists. There is no crawler and no
link following: adding knowledge is a deliberate code change.

Solar-PV notes explain methodology, standards, and regulation. They must not
contain scientific numeric calculations or site indicator values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import unquote

from app.rag.contracts import OSM_KNOWLEDGE_DOMAIN, SOLAR_PV_KNOWLEDGE_DOMAIN

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


SolarDocumentKind = Literal[
    "pv_methodology",
    "panel_documentation",
    "installation_guidelines",
    "grid_rules",
    "permitting_rules",
    "regulatory",
    "technical_standard",
    "scientific_methodology",
]

SOLAR_DOCUMENT_KINDS: frozenset[str] = frozenset(
    {
        "pv_methodology",
        "panel_documentation",
        "installation_guidelines",
        "grid_rules",
        "permitting_rules",
        "regulatory",
        "technical_standard",
        "scientific_methodology",
    }
)

#: Grounding notes are original text; citation URLs point at public sources.
SOLAR_PV_LICENSE = "CC-BY-4.0 (Ariadne Thread solar-PV grounding notes)"


@dataclass(frozen=True, slots=True)
class SolarCorpusEntry:
    """One approved solar-PV knowledge document (packaged markdown + citation)."""

    slug: str
    title: str
    kind: SolarDocumentKind
    citation_url: str
    filename: str
    license: str = SOLAR_PV_LICENSE

    @property
    def domain(self) -> str:
        return SOLAR_PV_KNOWLEDGE_DOMAIN

    @property
    def source_url(self) -> str:
        return self.citation_url

    @property
    def source_type(self) -> str:
        return self.kind


SOLAR_PV_KNOWLEDGE_CORPUS: tuple[SolarCorpusEntry, ...] = (
    SolarCorpusEntry(
        slug="ghi-definition",
        title="Global Horizontal Irradiance in solar assessment",
        kind="scientific_methodology",
        citation_url=(
            "https://joint-research-centre.ec.europa.eu/"
            "photovoltaic-geographical-information-system-pvgis_en"
        ),
        filename="ghi-definition.md",
    ),
    SolarCorpusEntry(
        slug="slope-in-pv-siting",
        title="Why slope matters for photovoltaic siting",
        kind="pv_methodology",
        citation_url="https://iea-pvps.org/",
        filename="slope-in-pv-siting.md",
    ),
    SolarCorpusEntry(
        slug="site-exclusion-policy",
        title="How photovoltaic site exclusion is explained",
        kind="scientific_methodology",
        citation_url="https://www.irena.org/Energy-Transition/Technology/Solar-energy",
        filename="site-exclusion-policy.md",
    ),
    SolarCorpusEntry(
        slug="grid-connection-rules",
        title="Grid-connection rules for photovoltaic plants",
        kind="grid_rules",
        citation_url="https://www.entsoe.eu/network_codes/",
        filename="grid-connection-rules.md",
    ),
    SolarCorpusEntry(
        slug="permitting-and-regulation",
        title="Permitting and regulation that can affect a solar site",
        kind="regulatory",
        citation_url="https://energy.ec.europa.eu/topics/renewable-energy/solar-energy_en",
        filename="permitting-and-regulation.md",
    ),
    SolarCorpusEntry(
        slug="panel-documentation",
        title="Photovoltaic module documentation and nameplate evidence",
        kind="panel_documentation",
        citation_url="https://www.iec.ch/",
        filename="panel-documentation.md",
    ),
    SolarCorpusEntry(
        slug="installation-guidelines",
        title="Photovoltaic installation guidelines",
        kind="installation_guidelines",
        citation_url="https://www.iso.org/ics/27.160.html",
        filename="installation-guidelines.md",
    ),
    SolarCorpusEntry(
        slug="technical-standards",
        title="Technical standards used in photovoltaic projects",
        kind="technical_standard",
        citation_url="https://www.iec.ch/homepage",
        filename="technical-standards.md",
    ),
    SolarCorpusEntry(
        slug="permitting-rules",
        title="Permitting process notes for photovoltaic development",
        kind="permitting_rules",
        citation_url="https://iea-pvps.org/key-topics/",
        filename="permitting-rules.md",
    ),
)

_ALLOWED_SOLAR_URLS = frozenset(entry.citation_url for entry in SOLAR_PV_KNOWLEDGE_CORPUS)


def is_allowed_solar_source(url: str) -> bool:
    """Whether a citation URL belongs to the solar-PV knowledge whitelist."""
    return url in _ALLOWED_SOLAR_URLS


def require_allowed_solar_source(url: str) -> None:
    """Raise if ``url`` is not an approved solar-PV documentation citation."""
    if not is_allowed_solar_source(url):
        from app.core.errors import CorpusWhitelistError

        raise CorpusWhitelistError(f"source URL is not on the solar-PV knowledge whitelist: {url}")
