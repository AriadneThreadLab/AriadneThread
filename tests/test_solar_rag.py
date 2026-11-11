"""Solar-PV RAG grounding: whitelist, ingest, retrieval, evidence partition."""

from __future__ import annotations

from uuid import uuid4

import pytest
from app.agent.accumulator import ResultAccumulator
from app.agent.contracts import GeoAgentRequest
from app.agent.evidence import partition_evidence
from app.agent.loop import LoopLimits
from app.agent.orchestrator import PlannerExecutorAgent, _model_facing_tools
from app.agent.solar_intent import (
    is_solar_decision_request,
    is_solar_knowledge_request,
    is_solar_request,
)
from app.analytics.datasets import DatasetRegistry
from app.core.errors import KnowledgeNumericContentError
from app.llm.contracts import ToolCall, ToolDefinition
from app.places.contracts import PlaceRegistry
from app.rag.contracts import (
    OSM_KNOWLEDGE_DOMAIN,
    SOLAR_PV_KNOWLEDGE_DOMAIN,
    RetrievedPassage,
)
from app.rag.corpus import (
    OSM_KNOWLEDGE_CORPUS,
    SOLAR_DOCUMENT_KINDS,
    SOLAR_PV_KNOWLEDGE_CORPUS,
    is_allowed_solar_source,
    is_allowed_source,
)
from app.rag.numeric_guard import reject_numeric_calculations
from app.rag.persist import InMemoryKnowledgeRepository
from app.rag.retriever import InMemoryPassageStore, PgVectorKnowledgeRetriever, ScoredChunkRow
from app.rag.solar_ingest import SOLAR_DOCUMENTS_DIR, ingest_solar_knowledge
from app.solar.store import InMemorySolarDecisionStore
from app.tools.context import AnalysisRunState, GroundingState, ToolContext
from app.tools.registry import ToolRegistry
from app.tools.search_solar_knowledge import (
    TOOL_NAME as SEARCH_SOLAR_KNOWLEDGE,
)
from app.tools.search_solar_knowledge import (
    SearchSolarKnowledgeArgs,
    SearchSolarKnowledgeTool,
)
from app.tools.solar_tools import TOOL_NAME as ANALYZE_SOLAR_SITE
from app.tools.solar_tools import AnalyzeSolarSiteTool
from pydantic import ValidationError

from tests.test_orchestrator import ScriptedLLM, _final, _tool_response
from tests.test_retriever import FakeEmbedder

_GHI = "What does GHI mean?"
_SLOPE = "Why is slope important?"
_EXCLUDED = "Why was this site excluded?"
_REGULATION = "What regulation affects this site?"

_SOLAR_PASSAGE = RetrievedPassage(
    content=(
        "Global horizontal irradiance, commonly abbreviated GHI, is the total "
        "solar radiation received on a horizontal surface. Numerical site "
        "values must come from SolarDSS, never from this corpus."
    ),
    document_title="Global Horizontal Irradiance in solar assessment",
    section="Definition",
    source_url=(
        "https://joint-research-centre.ec.europa.eu/"
        "photovoltaic-geographical-information-system-pvgis_en"
    ),
    score=0.91,
    domain=SOLAR_PV_KNOWLEDGE_DOMAIN,
    source_type="scientific_methodology",
)


class FakeSolarRetriever:
    def __init__(self, passages: list[RetrievedPassage] | None = None) -> None:
        self.passages = passages if passages is not None else [_SOLAR_PASSAGE]
        self.seen: list[tuple[str, int, str]] = []

    async def search(
        self, query: str, *, top_k: int, domain: str = OSM_KNOWLEDGE_DOMAIN
    ) -> list[RetrievedPassage]:
        self.seen.append((query, top_k, domain))
        if domain != SOLAR_PV_KNOWLEDGE_DOMAIN:
            return []
        return self.passages


def _ctx(message: str = _GHI) -> ToolContext:
    return ToolContext(
        datasets=DatasetRegistry(),
        analysis=AnalysisRunState(),
        user_message=message,
        places=PlaceRegistry(),
        grounding=GroundingState(),
    )


def test_osm_whitelist_is_unchanged() -> None:
    assert len(OSM_KNOWLEDGE_CORPUS) == 8
    assert not is_allowed_source(SOLAR_PV_KNOWLEDGE_CORPUS[0].citation_url)


def test_solar_corpus_covers_required_document_kinds() -> None:
    kinds = {entry.kind for entry in SOLAR_PV_KNOWLEDGE_CORPUS}
    assert kinds == SOLAR_DOCUMENT_KINDS
    urls = [entry.citation_url for entry in SOLAR_PV_KNOWLEDGE_CORPUS]
    assert len(urls) == len(set(urls))
    for entry in SOLAR_PV_KNOWLEDGE_CORPUS:
        assert entry.domain == SOLAR_PV_KNOWLEDGE_DOMAIN
        assert is_allowed_solar_source(entry.citation_url)
        path = SOLAR_DOCUMENTS_DIR / entry.filename
        assert path.is_file()
        reject_numeric_calculations(path.read_text(encoding="utf-8"), source=entry.filename)


def test_packaged_notes_reject_scientific_numeric_calculations() -> None:
    with pytest.raises(KnowledgeNumericContentError):
        reject_numeric_calculations(
            "Typical GHI = 1200 kWh/m2 and slope > 15.",
            source="fake.md",
        )


def test_example_questions_are_solar_knowledge_not_osm() -> None:
    for question in (_GHI, _SLOPE, _EXCLUDED, _REGULATION):
        assert is_solar_knowledge_request(question) is True
        assert is_solar_request(question) is True
        assert is_solar_decision_request(question) is False
    assert is_solar_knowledge_request("Find public parks in Berlin") is False
    assert is_solar_request("Find good solar farm locations in Vienna") is True
    assert is_solar_decision_request("Find good solar farm locations in Vienna") is True


@pytest.mark.asyncio
async def test_solar_ingest_from_packaged_markdown() -> None:
    repo = InMemoryKnowledgeRepository()
    results = await ingest_solar_knowledge(repo)
    assert len(results) == len(SOLAR_PV_KNOWLEDGE_CORPUS)
    assert {item.status for item in results} == {"inserted"}
    assert all(item.chunk_count >= 1 for item in results)
    keys = set(repo.documents)
    assert all(domain == SOLAR_PV_KNOWLEDGE_DOMAIN for domain, _url in keys)


@pytest.mark.asyncio
async def test_search_solar_knowledge_labels_document_evidence() -> None:
    retriever = FakeSolarRetriever()
    tool = SearchSolarKnowledgeTool(retriever)
    outcome = await tool.execute(SearchSolarKnowledgeArgs(query=_GHI, top_k=3), _ctx())
    assert retriever.seen == [(_GHI, 3, SOLAR_PV_KNOWLEDGE_DOMAIN)]
    assert "solar_documentation" in outcome.observation
    assert "knowledge_document" in outcome.observation
    assert "Never use these passages as replacements" in outcome.observation
    assert "DETERMINISTIC ANALYTICAL EVIDENCE" in outcome.observation
    assert "KNOWLEDGE / DOCUMENT EVIDENCE" in outcome.observation
    assert outcome.payload.domain == SOLAR_PV_KNOWLEDGE_DOMAIN
    assert outcome.payload.passages[0].source_url.startswith("https://")
    assert "geojson" not in outcome.payload.model_dump()


def test_search_solar_knowledge_forbids_indicator_fields() -> None:
    with pytest.raises(ValidationError):
        SearchSolarKnowledgeArgs.model_validate({"query": "GHI", "ghi": 1800})


@pytest.mark.asyncio
async def test_retriever_allows_solar_domain() -> None:
    row = ScoredChunkRow(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="GHI is global horizontal irradiance.",
        section="Definition",
        document_title="GHI definition",
        source_url=SOLAR_PV_KNOWLEDGE_CORPUS[0].citation_url,
        cosine_similarity=0.8,
        domain=SOLAR_PV_KNOWLEDGE_DOMAIN,
        source_type="scientific_methodology",
    )
    store = InMemoryPassageStore(rows=[row], domain=SOLAR_PV_KNOWLEDGE_DOMAIN)
    retriever = PgVectorKnowledgeRetriever(store, FakeEmbedder())
    passages = await retriever.search(_GHI, top_k=1, domain=SOLAR_PV_KNOWLEDGE_DOMAIN)
    assert passages[0].domain == SOLAR_PV_KNOWLEDGE_DOMAIN
    assert passages[0].source_type == "scientific_methodology"
    empty = await retriever.search(_GHI, top_k=1, domain=OSM_KNOWLEDGE_DOMAIN)
    assert empty == []


def test_evidence_partition_keeps_rag_out_of_solardss() -> None:
    evidence = partition_evidence(passages=[_SOLAR_PASSAGE], solar_decision=None)
    assert evidence.deterministic_analytical == ()
    assert len(evidence.knowledge_documents) == 1
    item = evidence.knowledge_documents[0]
    assert item.evidence_class == "knowledge_document"
    assert item.domain == SOLAR_PV_KNOWLEDGE_DOMAIN
    assert item.url == _SOLAR_PASSAGE.source_url


def test_model_facing_tools_for_knowledge_question() -> None:
    defs = [
        ToolDefinition(name="search_osm_knowledge", description="d", parameters_schema={}),
        ToolDefinition(name="search_solar_knowledge", description="d", parameters_schema={}),
        ToolDefinition(name="query_osm", description="d", parameters_schema={}),
        ToolDefinition(name=ANALYZE_SOLAR_SITE, description="d", parameters_schema={}),
    ]
    names = [
        item.name
        for item in _model_facing_tools(
            defs,
            state=ResultAccumulator(),
            user_message=_GHI,
            places=PlaceRegistry(),
        )
    ]
    assert names == [SEARCH_SOLAR_KNOWLEDGE]
    assert "query_osm" not in names
    assert ANALYZE_SOLAR_SITE not in names


def test_model_facing_tools_for_solar_siting_include_rag() -> None:
    defs = [
        ToolDefinition(name="search_osm_knowledge", description="d", parameters_schema={}),
        ToolDefinition(name="search_solar_knowledge", description="d", parameters_schema={}),
        ToolDefinition(name="query_osm", description="d", parameters_schema={}),
        ToolDefinition(name=ANALYZE_SOLAR_SITE, description="d", parameters_schema={}),
    ]
    names = [
        item.name
        for item in _model_facing_tools(
            defs,
            state=ResultAccumulator(),
            user_message="Find good solar farm locations in Vienna",
            places=PlaceRegistry(),
        )
    ]
    assert names == [SEARCH_SOLAR_KNOWLEDGE, ANALYZE_SOLAR_SITE]


@pytest.mark.asyncio
async def test_orchestrator_ghi_question_uses_document_evidence_only() -> None:
    retriever = FakeSolarRetriever()
    registry = ToolRegistry()
    registry.register(SearchSolarKnowledgeTool(retriever))
    registry.register(AnalyzeSolarSiteTool(store=InMemorySolarDecisionStore()))
    llm = ScriptedLLM(
        [
            _tool_response(
                ToolCall(id="1", name=SEARCH_SOLAR_KNOWLEDGE, arguments={"query": _GHI})
            ),
            _final(
                "KNOWLEDGE / DOCUMENT EVIDENCE: GHI is global horizontal irradiance. "
                "DETERMINISTIC ANALYTICAL EVIDENCE: none. No SolarDSS GHI was used."
            ),
        ]
    )
    agent = PlannerExecutorAgent(
        llm,  # type: ignore[arg-type]
        registry,
        LoopLimits(max_tool_rounds=4, max_tool_calls=8),
        solar_store=InMemorySolarDecisionStore(),
    )
    result = await agent.run(GeoAgentRequest(message=_GHI))
    assert result.geojson is None
    assert result.solar_decision is None
    assert result.passages
    assert result.passages[0].domain == SOLAR_PV_KNOWLEDGE_DOMAIN
    assert result.evidence.knowledge_documents
    assert result.evidence.deterministic_analytical == ()
    assert result.evidence.knowledge_documents[0].url == _SOLAR_PASSAGE.source_url
    assert any(source.kind == "solar_documentation" for source in result.sources)


@pytest.mark.asyncio
async def test_orchestrator_does_not_copy_rag_into_solar_indicators() -> None:
    retriever = FakeSolarRetriever()
    store = InMemorySolarDecisionStore()
    registry = ToolRegistry()
    registry.register(SearchSolarKnowledgeTool(retriever))
    registry.register(AnalyzeSolarSiteTool(store=store))
    llm = ScriptedLLM(
        [
            _tool_response(
                ToolCall(
                    id="1",
                    name=ANALYZE_SOLAR_SITE,
                    arguments={
                        "mode": "site_selection",
                        "site_selection_scope": "region_search",
                        "project_type": "utility_scale",
                        "region_name": "Vienna",
                        "capability_id": "solar_site_selection",
                    },
                ),
                ToolCall(id="2", name=SEARCH_SOLAR_KNOWLEDGE, arguments={"query": _EXCLUDED}),
            ),
            _final(
                "DETERMINISTIC ANALYTICAL EVIDENCE: Vienna is routed for site selection. "
                "KNOWLEDGE / DOCUMENT EVIDENCE: exclusion is an engine screen, not RAG."
            ),
        ]
    )
    agent = PlannerExecutorAgent(
        llm,  # type: ignore[arg-type]
        registry,
        LoopLimits(max_tool_rounds=4, max_tool_calls=8),
        solar_store=store,
    )
    result = await agent.run(
        GeoAgentRequest(
            message="Find good solar farm locations in Vienna. Why might a site be excluded?"
        )
    )
    assert result.solar_decision is not None
    problem = result.solar_decision.problem
    assert "ghi" not in problem
    assert "slope" not in problem
    analytical = result.evidence.deterministic_analytical
    knowledge = result.evidence.knowledge_documents
    assert analytical
    assert knowledge
    assert analytical[0].engine == "SolarDSS"
    assert analytical[0].payload.get("science") == "not_computed"
    assert knowledge[0].evidence_class == "knowledge_document"
    dumped = str(analytical[0].payload)
    assert "1200" not in dumped
    assert result.geojson is None
