"""Partition deterministic analytical evidence from knowledge-document evidence.

RAG passages never populate the analytical list. SolarDSS slots and engine
outputs never populate the knowledge list.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.analytics.contracts import AnalysisBlock
from app.rag.contracts import (
    OSM_KNOWLEDGE_DOMAIN,
    SOLAR_PV_KNOWLEDGE_DOMAIN,
    RetrievedPassage,
)
from app.solar.contracts import SolarDecisionReport
from app.solar.display import SolarDecisionSupportReport
from app.tools.energy_report import EnergyAnalysisReport

DETERMINISTIC_EVIDENCE_CLASS: Literal["deterministic_analytical"] = "deterministic_analytical"
KNOWLEDGE_EVIDENCE_CLASS: Literal["knowledge_document"] = "knowledge_document"


class DeterministicAnalyticalEvidence(BaseModel):
    """One engine-backed analytical item. Never sourced from RAG."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_class: Literal["deterministic_analytical"] = DETERMINISTIC_EVIDENCE_CLASS
    engine: str = Field(min_length=1, max_length=80)
    provenance: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=500)
    citations: tuple[str, ...] = ()
    science_not_from_rag: Literal[True] = True
    payload: dict[str, Any] = Field(default_factory=dict)


class KnowledgeDocumentEvidence(BaseModel):
    """One retrieved documentation citation. Never a SolarDSS indicator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_class: Literal["knowledge_document"] = KNOWLEDGE_EVIDENCE_CLASS
    domain: str
    document_kind: str | None = None
    title: str
    section: str | None = None
    url: str
    score: float
    document_id: str | None = None
    chunk_id: str | None = None


class EvidencePartition(BaseModel):
    """Public split between engine evidence and document evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deterministic_analytical: tuple[DeterministicAnalyticalEvidence, ...] = ()
    knowledge_documents: tuple[KnowledgeDocumentEvidence, ...] = ()


def knowledge_from_passages(
    passages: Sequence[RetrievedPassage],
    *,
    limit: int = 8,
) -> tuple[KnowledgeDocumentEvidence, ...]:
    """Map retrieved passages to knowledge-document citations."""
    ranked = sorted(passages, key=lambda item: item.score, reverse=True)
    items: list[KnowledgeDocumentEvidence] = []
    seen: set[tuple[str, str | None, str]] = set()
    for passage in ranked:
        key = (passage.document_title, passage.section, passage.source_url)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            KnowledgeDocumentEvidence(
                domain=passage.domain or OSM_KNOWLEDGE_DOMAIN,
                document_kind=passage.source_type,
                title=passage.document_title,
                section=passage.section,
                url=passage.source_url,
                score=passage.score,
                document_id=str(passage.document_id) if passage.document_id else None,
                chunk_id=str(passage.chunk_id) if passage.chunk_id else None,
            )
        )
        if len(items) >= limit:
            break
    return tuple(items)


def partition_evidence(
    *,
    passages: Sequence[RetrievedPassage] = (),
    solar_decision: SolarDecisionReport | None = None,
    analysis: AnalysisBlock | None = None,
    energy_analysis: EnergyAnalysisReport | None = None,
    solar_results: SolarDecisionSupportReport | None = None,
) -> EvidencePartition:
    """Build the response evidence split. RAG never fills analytical items."""
    analytical: list[DeterministicAnalyticalEvidence] = []
    if solar_decision is not None:
        citations = []
        if solar_decision.capability_id:
            citations.append(f"capability:{solar_decision.capability_id}")
        citations.append(f"requirements_source:{solar_decision.requirements_source}")
        analytical.append(
            DeterministicAnalyticalEvidence(
                engine="SolarDSS",
                provenance="solardss_decision_routing",
                title="Solar decision routing (indicators not computed from RAG)",
                summary=(
                    "Structured SolarDecisionProblem slots from analyze_solar_site. "
                    "GHI, slope, and other scientific indicators are not taken from "
                    "retrieved documents."
                ),
                citations=tuple(citations),
                payload={
                    "status": solar_decision.status,
                    "mode": solar_decision.mode,
                    "site_selection_scope": solar_decision.site_selection_scope,
                    "capability_id": solar_decision.capability_id,
                    "problem": solar_decision.problem,
                    "science": "not_computed",
                    "source": (
                        "solardss"
                        if solar_decision.requirements_source == "solardss"
                        else "host_mirror"
                    ),
                },
            )
        )
    if analysis is not None and analysis.result is not None:
        indicator_id = analysis.decision_trace.selected_indicator_id
        citations = []
        if isinstance(indicator_id, str) and indicator_id:
            citations.append(f"indicator:{indicator_id}")
        analytical.append(
            DeterministicAnalyticalEvidence(
                engine="SpatialAnalyticsEngine",
                provenance="analyze_features",
                title="Deterministic spatial analysis",
                summary=(
                    "Indicator values computed by the spatial analytics engine. "
                    "Not retrieved from documentation."
                ),
                citations=tuple(citations),
                payload={"status": analysis.status, "science": "engine_computed"},
            )
        )
    if energy_analysis is not None:
        citations = [f"capability:{energy_analysis.capability_id}"]
        analytical.append(
            DeterministicAnalyticalEvidence(
                engine="GeoLoadST",
                provenance="analyze_energy_grid",
                title="Deterministic energy-network analysis",
                summary=(
                    "GeoLoadST results. Not retrieved from documentation and not "
                    "SolarDSS indicators."
                ),
                citations=tuple(citations),
                payload={"science": "engine_computed"},
            )
        )
    if solar_results is not None:
        copied_fields = [
            name
            for name, value in (
                ("decision_mode", solar_results.decision_mode),
                ("project_type", solar_results.project_type),
                ("selected_site", solar_results.selected_site),
                ("suitability_score", solar_results.suitability_score),
                ("ghi", solar_results.ghi),
                ("dni", solar_results.dni),
                ("slope", solar_results.slope),
                ("aspect", solar_results.aspect),
                ("shading_impact", solar_results.shading_impact),
                ("roof_suitability", solar_results.roof_suitability),
                ("available_area", solar_results.available_area),
                ("distance_to_grid", solar_results.distance_to_grid),
                ("distance_to_road", solar_results.distance_to_road),
                ("protected_area_status", solar_results.protected_area_status),
                ("recommended_capacity", solar_results.recommended_capacity),
                ("module_technology", solar_results.module_technology),
                ("tilt", solar_results.tilt),
                ("azimuth", solar_results.azimuth),
                ("expected_annual_energy", solar_results.expected_annual_energy),
                ("specific_yield", solar_results.specific_yield),
            )
            if value is not None
        ]
        citations = []
        if solar_results.capability_id:
            citations.append(f"capability:{solar_results.capability_id}")
        if solar_results.provenance.solardss_version:
            citations.append(f"solardss:{solar_results.provenance.solardss_version}")
        analytical.append(
            DeterministicAnalyticalEvidence(
                engine="SolarDSS",
                provenance="solardss_integration_layer",
                title="Solar decision support indicators (copied from SolarDSS)",
                summary=(
                    "Deterministic SolarDSS indicators copied through the "
                    "integration layer. Not retrieved from documentation and not "
                    "computed by Ariadne."
                ),
                citations=tuple(citations),
                payload={
                    "science": "copied_from_solardss",
                    "copied_fields": copied_fields,
                    "candidate_count": len(solar_results.candidates),
                },
            )
        )
    return EvidencePartition(
        deterministic_analytical=tuple(analytical),
        knowledge_documents=knowledge_from_passages(passages),
    )


def solar_passages_present(passages: Sequence[RetrievedPassage]) -> bool:
    """True when any retrieved passage belongs to the solar-PV corpus."""
    return any(item.domain == SOLAR_PV_KNOWLEDGE_DOMAIN for item in passages)
