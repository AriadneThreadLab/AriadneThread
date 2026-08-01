"""Offline semantic retrieval and tool registration tests."""

from __future__ import annotations

from uuid import uuid4

import pytest
from app.core.errors import EmbeddingError
from app.db.models import EMBEDDING_DIMENSION
from app.embeddings.contracts import Vector
from app.rag.contracts import OSM_KNOWLEDGE_DOMAIN
from app.rag.retriever import (
    InMemoryPassageStore,
    PgVectorKnowledgeRetriever,
    RetrievalError,
    ScoredChunkRow,
    cosine_similarity_from_distance,
    validate_top_k,
)
from app.tools.factory import build_tool_registry
from app.tools.search_osm_knowledge import SearchOsmKnowledgeArgs, SearchOsmKnowledgeTool


def _unit() -> Vector:
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[0] = 1.0
    return vector


class FakeEmbedder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.queries: list[str] = []
        self.model_name = "fake"
        self.dimension = EMBEDDING_DIMENSION

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        return [_unit() for _ in texts]

    async def embed_query(self, text: str) -> Vector:
        self.queries.append(text)
        if self.fail:
            raise EmbeddingError("embed failed")
        return _unit()

    async def aclose(self) -> None:
        return None


def _row(score: float, *, title: str = "Tag: leisure=park") -> ScoredChunkRow:
    return ScoredChunkRow(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="leisure=park is used for public parks.",
        section="Description",
        document_title=title,
        source_url="https://wiki.openstreetmap.org/wiki/Tag:leisure%3Dpark",
        cosine_similarity=score,
    )


def test_score_is_cosine_similarity_not_distance():
    assert cosine_similarity_from_distance(0.2) == pytest.approx(0.8)
    assert cosine_similarity_from_distance(0.0) == pytest.approx(1.0)


def test_top_k_validation_bounds():
    assert validate_top_k(5) == 5
    with pytest.raises(RetrievalError):
        validate_top_k(0)
    with pytest.raises(RetrievalError):
        validate_top_k(100)


async def test_retriever_orders_by_similarity_and_filters_domain():
    rows = [_row(0.4), _row(0.9), _row(0.7)]
    store = InMemoryPassageStore(rows=rows, domain=OSM_KNOWLEDGE_DOMAIN)
    retriever = PgVectorKnowledgeRetriever(store, FakeEmbedder())

    passages = await retriever.search("پارک عمومی", top_k=2)
    assert [passage.score for passage in passages] == [0.9, 0.7]
    assert all(passage.document_title == "Tag: leisure=park" for passage in passages)
    assert passages[0].chunk_id is not None
    assert passages[0].document_id is not None


async def test_retriever_rejects_unsupported_domain():
    retriever = PgVectorKnowledgeRetriever(InMemoryPassageStore(rows=[]), FakeEmbedder())
    with pytest.raises(RetrievalError, match="unsupported knowledge domain"):
        await retriever.search("park", top_k=3, domain="other_domain")


async def test_empty_corpus_returns_empty_list():
    retriever = PgVectorKnowledgeRetriever(InMemoryPassageStore(rows=[]), FakeEmbedder())
    assert await retriever.search("park tags", top_k=5) == []


async def test_provider_failure_propagates():
    retriever = PgVectorKnowledgeRetriever(
        InMemoryPassageStore(rows=[_row(0.5)]), FakeEmbedder(fail=True)
    )
    with pytest.raises(EmbeddingError, match="embed failed"):
        await retriever.search("park", top_k=3)


async def test_search_tool_is_registered_and_returns_documentation_shape():
    store = InMemoryPassageStore(rows=[_row(0.88)])
    retriever = PgVectorKnowledgeRetriever(store, FakeEmbedder())
    registry = build_tool_registry(knowledge_retriever=retriever, rag_top_k=5)

    assert registry.names == ("search_osm_knowledge",)
    definition = registry.definitions()[0]
    assert definition.name == "search_osm_knowledge"
    assert "NOT live map" in definition.description

    tool = SearchOsmKnowledgeTool(retriever)
    outcome = await tool.execute(SearchOsmKnowledgeArgs(query="public park tag", top_k=1))
    assert outcome.payload.passages[0].score == pytest.approx(0.88)
    assert "not live map data" in outcome.observation
