"""The two agent tools, exercised against fake collaborators."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import pytest
from app.core.errors import EmbeddingError, ToolExecutionError
from app.osm.contracts import OSM_ATTRIBUTION, OverpassElement, OverpassResponse
from app.osm.query_spec import OsmFeatureQuery, PointRadius, TagFilter
from app.rag.contracts import OSM_KNOWLEDGE_DOMAIN, RetrievedPassage
from app.tools.query_osm import QueryOsmTool
from app.tools.search_osm_knowledge import SearchOsmKnowledgeArgs, SearchOsmKnowledgeTool

_PASSAGE = RetrievedPassage(
    content="leisure=park is used for public green areas intended for recreation.",
    document_title="Tag: leisure=park",
    section="Description",
    source_url="https://wiki.openstreetmap.org/wiki/Tag:leisure%3Dpark",
    score=0.91,
)


class FakeRetriever:
    def __init__(self, passages: list[RetrievedPassage] | None = None) -> None:
        self.passages = passages if passages is not None else [_PASSAGE]
        self.seen: list[tuple[str, int, str]] = []

    async def search(
        self, query: str, *, top_k: int, domain: str = OSM_KNOWLEDGE_DOMAIN
    ) -> list[RetrievedPassage]:
        self.seen.append((query, top_k, domain))
        return self.passages


class FailingRetriever:
    async def search(
        self, query: str, *, top_k: int, domain: str = OSM_KNOWLEDGE_DOMAIN
    ) -> list[RetrievedPassage]:
        raise EmbeddingError("model not loaded")


class FakeOverpassClient:
    def __init__(self, elements: tuple[OverpassElement, ...] = (), truncated: bool = False) -> None:
        self._elements = elements
        self._truncated = truncated
        self.queries: list[str] = []

    @property
    def endpoint(self) -> str:
        return "https://overpass.test/api/interpreter"

    async def run(self, query: str) -> OverpassResponse:
        self.queries.append(query)
        return OverpassResponse(
            elements=self._elements,
            query=query,
            endpoint=self.endpoint,
            retrieved_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            response_bytes=len(query),
            truncated=self._truncated,
        )

    async def aclose(self) -> None:
        return None


class FakeEncoder:
    def encode(self, elements: Sequence[OverpassElement]) -> dict[str, object]:
        return {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": element, "geometry": None} for element in elements
            ],
        }


# --- search_osm_knowledge ---


async def test_knowledge_search_returns_passages_and_labels_them_as_documentation():
    retriever = FakeRetriever()
    tool = SearchOsmKnowledgeTool(retriever)
    outcome = await tool.execute(SearchOsmKnowledgeArgs(query="tag for public park", top_k=3))

    assert retriever.seen == [("tag for public park", 3, OSM_KNOWLEDGE_DOMAIN)]
    assert "not live map data" in outcome.observation
    assert "Tag: leisure=park > Description" in outcome.observation
    assert outcome.payload.passage_count == 1
    assert outcome.payload.domain == OSM_KNOWLEDGE_DOMAIN


async def test_empty_knowledge_search_tells_the_model_not_to_guess():
    tool = SearchOsmKnowledgeTool(FakeRetriever(passages=[]))
    outcome = await tool.execute(SearchOsmKnowledgeArgs(query="unknown concept"))
    assert "Do not guess tags" in outcome.observation
    assert outcome.payload.passages == []


async def test_retrieval_failure_becomes_a_tool_execution_error():
    tool = SearchOsmKnowledgeTool(FailingRetriever())
    with pytest.raises(ToolExecutionError, match="knowledge search unavailable"):
        await tool.execute(SearchOsmKnowledgeArgs(query="park"))


def test_knowledge_tool_schema_bounds_top_k():
    schema = SearchOsmKnowledgeArgs.model_json_schema()
    assert schema["properties"]["top_k"]["maximum"] == 20
    assert schema["additionalProperties"] is False


# --- query_osm ---


def _query_tool(client: FakeOverpassClient, *, max_results: int = 1000) -> QueryOsmTool:
    return QueryOsmTool(client, FakeEncoder(), timeout_seconds=30, max_results=max_results)


async def test_query_osm_builds_the_query_and_returns_geojson():
    client = FakeOverpassClient(
        elements=({"type": "node", "id": 1}, {"type": "way", "id": 2}),
    )
    tool = _query_tool(client)
    outcome = await tool.execute(
        OsmFeatureQuery(place="Berlin", tags=[TagFilter(key="leisure", value="park")])
    )

    assert client.queries[0].startswith("[out:json][timeout:30];")
    assert '"leisure"="park"' in client.queries[0]
    assert outcome.payload.feature_count == 2
    assert outcome.payload.geojson["type"] == "FeatureCollection"
    assert outcome.payload.overpass_query == client.queries[0]
    assert outcome.payload.source.attribution == OSM_ATTRIBUTION
    assert "2 feature(s) returned" in outcome.observation
    assert "leisure=park in Berlin" in outcome.observation


async def test_query_osm_reports_an_empty_result_instead_of_inviting_invention():
    tool = _query_tool(FakeOverpassClient(elements=()))
    outcome = await tool.execute(
        OsmFeatureQuery(
            point=PointRadius(lat=52.5219, lon=13.4132, radius_m=2000),
            tags=[TagFilter(key="amenity", value="pharmacy")],
        )
    )
    assert outcome.payload.feature_count == 0
    assert "Do not invent results" in outcome.observation
    assert "within 2000 m of 52.5219,13.4132" in outcome.observation


async def test_query_osm_clamps_the_limit_to_the_configured_maximum():
    client = FakeOverpassClient()
    tool = _query_tool(client, max_results=100)
    await tool.execute(
        OsmFeatureQuery(place="Berlin", tags=[TagFilter(key="leisure", value="park")], limit=1000)
    )
    assert client.queries[0].endswith("out geom 100;")


async def test_query_osm_surfaces_truncation():
    client = FakeOverpassClient(elements=({"type": "node", "id": 1},), truncated=True)
    tool = _query_tool(client)
    outcome = await tool.execute(
        OsmFeatureQuery(place="Berlin", tags=[TagFilter(key="leisure", value="park")], limit=1)
    )
    assert outcome.payload.truncated
    assert "truncated" in outcome.observation


def test_query_osm_arguments_are_the_validated_query_spec():
    tool = _query_tool(FakeOverpassClient())
    assert tool.args_model is OsmFeatureQuery
    schema = tool.args_model.model_json_schema()
    assert schema["additionalProperties"] is False
    assert "overpass_ql" not in schema["properties"]
