"""Accumulate structured tool payloads into one agent result.

GeoJSON and documentation passages stay here for the API/UI. Only compact
observations are ever appended to the LLM conversation.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.contracts import SourceReference
from app.agent.geojson_combine import combine_target_feature_collections
from app.analytics.charts import AnalysisChart
from app.analytics.contracts import AnalysisBlock, KnowledgeSourceRef
from app.analytics.datasets import DatasetRegistry
from app.osm.contracts import GeoJsonFeatureCollection
from app.rag.contracts import RetrievedPassage
from app.tools.analyze_features import TOOL_NAME as ANALYZE_FEATURES
from app.tools.analyze_features import AnalyzeFeaturesResult
from app.tools.energy_report import LAYER_SIMBENCH, EnergyAnalysisReport, layer_name_for
from app.tools.energy_tools import TOOL_NAME as ANALYZE_ENERGY_GRID
from app.tools.energy_tools import AnalyzeEnergyGridResult
from app.tools.query_osm import TOOL_NAME as QUERY_OSM
from app.tools.query_osm import QueryOsmResult
from app.tools.resolve_place import TOOL_NAME as RESOLVE_PLACE
from app.tools.resolve_place import PlaceResolutionResult
from app.tools.search_osm_knowledge import (
    TOOL_NAME as SEARCH_OSM_KNOWLEDGE,
)
from app.tools.search_osm_knowledge import (
    SearchOsmKnowledgeResult,
)
from app.tools.simbench_tools import TOOL_NAME as SIMBENCH_QUERY
from app.tools.simbench_tools import SimBenchQueryResult

_TAG_TITLE_RE = re.compile(
    r"^Tag:\s*([A-Za-z0-9_:-]+)=([A-Za-z0-9_:-]+)$",
    re.IGNORECASE,
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
        self.effective_limit: int | None = None
        self.scope_summary: str | None = None
        self.validated_tags: list[str] = []
        self.live_query_failed: bool = False
        self.live_error_code: str | None = None
        self.analysis: AnalysisBlock | None = None
        self.datasets = DatasetRegistry()
        self.target_feature_counts: dict[str, int] = {}
        self._target_collections: list[tuple[str, GeoJsonFeatureCollection]] = []
        self._source_keys: set[tuple[str, str, str | None]] = set()
        self._overpass_queries: list[str] = []
        self.last_simbench_network_id: str | None = None
        self.energy_analysis: EnergyAnalysisReport | None = None
        self.charts: list[AnalysisChart] = []
        self.energy_workflow_stopped: bool = False
        self.energy_terminal_error_code: str | None = None

    def absorb(self, tool_name: str, payload: Any) -> None:
        if tool_name == QUERY_OSM and isinstance(payload, QueryOsmResult):
            self._absorb_query_osm(payload)
            return
        if tool_name == SEARCH_OSM_KNOWLEDGE and isinstance(payload, SearchOsmKnowledgeResult):
            self._absorb_knowledge(payload)
            return
        if tool_name == RESOLVE_PLACE and isinstance(payload, PlaceResolutionResult):
            self._absorb_place(payload)
            return
        if tool_name == ANALYZE_FEATURES and isinstance(payload, AnalyzeFeaturesResult):
            self._absorb_analysis(payload)
            return
        if tool_name == SIMBENCH_QUERY and isinstance(payload, SimBenchQueryResult):
            self._absorb_simbench(payload)
            return
        if tool_name == ANALYZE_ENERGY_GRID and isinstance(payload, AnalyzeEnergyGridResult):
            self._absorb_energy_analysis(payload)

    def _absorb_query_osm(self, result: QueryOsmResult) -> None:
        self.absorb_osm_target_collection(
            analysis_target=result.analysis_target or "query",
            geojson=result.geojson,
            feature_count=result.feature_count,
            warnings=result.warnings,
            effective_limit=result.effective_limit,
            scope_summary=result.scope_summary,
            overpass_query=result.overpass_query,
        )

    def absorb_osm_target_collection(
        self,
        *,
        analysis_target: str,
        geojson: GeoJsonFeatureCollection,
        feature_count: int,
        warnings: list[str] | None = None,
        effective_limit: int | None = None,
        scope_summary: str | None = None,
        overpass_query: str | None = None,
    ) -> None:
        """Merge one target's OSM FeatureCollection into the response map.

        Used for live ``query_osm`` results and for restored execution-memory
        collections that originally came from ``query_osm``.
        """
        self.target_feature_counts[analysis_target] = feature_count
        self._target_collections.append((analysis_target, geojson))
        if len(self._target_collections) == 1:
            self.geojson = geojson
            self.feature_count = feature_count
            self.scope_summary = scope_summary
        else:
            combined = combine_target_feature_collections(self._target_collections)
            self.geojson = combined
            self.feature_count = len(combined.get("features", []))
            summaries = [f"{label}: {count}" for label, count in self.target_feature_counts.items()]
            self.scope_summary = "Comparison targets — " + "; ".join(summaries)
        if overpass_query:
            self.overpass_query = overpass_query
            self._overpass_queries.append(overpass_query)
        if effective_limit is not None:
            self.effective_limit = effective_limit
        self.live_query_failed = False
        self.live_error_code = None
        if warnings:
            self.warnings.extend(warnings)
        self._add_source(
            SourceReference(
                kind="osm_features",
                title="OpenStreetMap features (Overpass)",
                url=None,
            )
        )

    def _absorb_simbench(self, result: SimBenchQueryResult) -> None:
        """Attach SimBench metadata and geometries. Never labeled as OSM."""
        if result.warnings:
            self.warnings.extend(result.warnings)
        self._add_source(
            SourceReference(
                kind="simbench_network",
                title=result.source_title,
                url=None,
            )
        )
        if result.metadata is not None:
            self.last_simbench_network_id = result.metadata.network_id
        if result.action != "load" or result.geojson is None:
            if result.scope_summary and not self.scope_summary:
                self.scope_summary = result.scope_summary
            return
        target = LAYER_SIMBENCH
        self.target_feature_counts[target] = result.feature_count
        self._target_collections.append((target, result.geojson))
        if len(self._target_collections) == 1:
            self.geojson = result.geojson
            self.feature_count = result.feature_count
            self.scope_summary = result.scope_summary
        else:
            combined = combine_target_feature_collections(self._target_collections)
            self.geojson = combined
            self.feature_count = len(combined.get("features", []))
            summaries = [f"{label}: {count}" for label, count in self.target_feature_counts.items()]
            self.scope_summary = "Comparison targets — " + "; ".join(summaries)
        self.live_query_failed = False
        self.live_error_code = None

    def _absorb_energy_analysis(self, result: AnalyzeEnergyGridResult) -> None:
        """Attach GeoLoadST map output. Never labeled as OSM."""
        if result.warnings:
            self.warnings.extend(result.warnings)
        self._add_source(
            SourceReference(
                kind="energy_analysis",
                title=result.source_title,
                url=None,
            )
        )
        self.last_simbench_network_id = result.network_id
        if result.analysis is not None:
            self.energy_analysis = result.analysis
        if result.charts:
            self.charts.extend(result.charts)
        layer = result.geojson if result.geojson is not None else result.spatial_layer
        features = layer.get("features") if isinstance(layer, dict) else None
        if not isinstance(features, list) or not features:
            if result.scope_summary and not self.scope_summary:
                self.scope_summary = result.scope_summary
            return
        target = layer_name_for(result.capability, result.analysis)
        collection = {"type": "FeatureCollection", "features": features}
        count = result.feature_count or len(features)
        self.target_feature_counts[target] = count
        self._target_collections.append((target, collection))
        if len(self._target_collections) == 1:
            self.geojson = collection
            self.feature_count = count
            self.scope_summary = result.scope_summary
        else:
            combined = combine_target_feature_collections(self._target_collections)
            self.geojson = combined
            self.feature_count = len(combined.get("features", []))
            summaries = [f"{label}: {count}" for label, count in self.target_feature_counts.items()]
            self.scope_summary = "Comparison targets — " + "; ".join(summaries)
        self.live_query_failed = False
        self.live_error_code = None

    def note_live_query_failure(self, meta: dict[str, Any]) -> None:
        """Record validated query provenance after Overpass fails (no fake GeoJSON)."""
        query = meta.get("overpass_query")
        if isinstance(query, str) and query.strip():
            self.overpass_query = query
        limit = meta.get("effective_limit")
        if isinstance(limit, int) and limit > 0:
            self.effective_limit = limit
        scope = meta.get("scope_summary")
        if isinstance(scope, str) and scope.strip():
            self.scope_summary = scope
        tags = meta.get("validated_tags")
        if isinstance(tags, list):
            self.validated_tags = [str(tag) for tag in tags if isinstance(tag, str)]
        code = meta.get("error_code")
        if isinstance(code, str) and code:
            self.live_error_code = code
        self.live_query_failed = True
        # Keep prior successful target collections if any; do not invent empty FC.
        if not self._target_collections:
            self.geojson = None
            self.feature_count = None

    def _absorb_knowledge(self, result: SearchOsmKnowledgeResult) -> None:
        self.absorb_documentation_passages(result.passages)

    def absorb_documentation_passages(self, passages: list[RetrievedPassage]) -> None:
        """Attach OSM documentation citations from a live search or restored memory."""
        self.passages.extend(passages)
        for passage in passages:
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

    def _absorb_place(self, result: PlaceResolutionResult) -> None:
        # Provenance lives in the execution trace; do not mix with OSM feature sources.
        del result

    def _absorb_analysis(self, result: AnalyzeFeaturesResult) -> None:
        grounding = [
            KnowledgeSourceRef(title=source.title, url=source.url)
            for source in self.sources
            if source.kind == "osm_documentation"
        ][:3]
        # Attach grounding into result data provenance when completed.
        analysis_result = result.result
        if analysis_result is not None and grounding:
            updated_targets = []
            for target in analysis_result.targets:
                updated_targets.append(
                    target.model_copy(
                        update={
                            "data_provenance": target.data_provenance.model_copy(
                                update={"grounding_sources": grounding}
                            )
                        }
                    )
                )
            analysis_result = analysis_result.model_copy(update={"targets": updated_targets})

        self.analysis = AnalysisBlock(
            plan=result.plan,
            decision_trace=result.decision_trace,
            result=analysis_result,
            comparison=result.comparison,
            report=result.report,
            status=result.status,
        )
        self.warnings.extend(result.warnings)

    def absorb_grounding_tags(self, tags: list[str]) -> None:
        if tags and not self.validated_tags:
            self.validated_tags = list(tags)

    def documented_tag_hints(self) -> list[str]:
        hints: list[str] = []
        seen: set[str] = set()
        for passage in self.passages:
            match = _TAG_TITLE_RE.match(passage.document_title.strip())
            if match is None:
                continue
            tag = f"{match.group(1)}={match.group(2)}"
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            hints.append(tag)
        return hints

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
