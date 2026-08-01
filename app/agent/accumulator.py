"""Accumulate structured tool payloads into one agent result.

GeoJSON and documentation passages stay here for the API/UI. Only compact
observations are ever appended to the LLM conversation.
"""

from __future__ import annotations

from typing import Any

from app.agent.contracts import SourceReference
from app.osm.contracts import GeoJsonFeatureCollection
from app.rag.contracts import RetrievedPassage
from app.tools.query_osm import TOOL_NAME as QUERY_OSM
from app.tools.query_osm import QueryOsmResult
from app.tools.search_osm_knowledge import (
    TOOL_NAME as SEARCH_OSM_KNOWLEDGE,
)
from app.tools.search_osm_knowledge import (
    SearchOsmKnowledgeResult,
)


class ResultAccumulator:
    """Mutable per-run store for verified tool outcomes."""

    def __init__(self) -> None:
        self.geojson: GeoJsonFeatureCollection | None = None
        self.feature_count: int | None = None
        self.overpass_query: str | None = None
        self.passages: list[RetrievedPassage] = []
        self.sources: list[SourceReference] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self._source_keys: set[tuple[str, str, str | None]] = set()

    def absorb(self, tool_name: str, payload: Any) -> None:
        if tool_name == QUERY_OSM and isinstance(payload, QueryOsmResult):
            self._absorb_query_osm(payload)
            return
        if tool_name == SEARCH_OSM_KNOWLEDGE and isinstance(payload, SearchOsmKnowledgeResult):
            self._absorb_knowledge(payload)

    def _absorb_query_osm(self, result: QueryOsmResult) -> None:
        # Replace rather than merge: one live FeatureCollection per successful query.
        self.geojson = result.geojson
        self.feature_count = result.feature_count
        self.overpass_query = result.overpass_query
        self.warnings.extend(result.warnings)
        self._add_source(
            SourceReference(
                kind="osm_features",
                title="OpenStreetMap features (Overpass)",
                url=None,
            )
        )

    def _absorb_knowledge(self, result: SearchOsmKnowledgeResult) -> None:
        self.passages.extend(result.passages)
        for passage in result.passages:
            title = passage.document_title
            if passage.section:
                title = f"{title} > {passage.section}"
            self._add_source(
                SourceReference(
                    kind="osm_documentation",
                    title=title,
                    url=passage.source_url,
                )
            )

    def _add_source(self, source: SourceReference) -> None:
        key = (source.kind, source.title, source.url)
        if key in self._source_keys:
            return
        self._source_keys.add(key)
        self.sources.append(source)

    def note_error(self, code: str, message: str) -> None:
        self.errors.append(f"{code}: {message}")

    def note_warning(self, message: str) -> None:
        self.warnings.append(message)
