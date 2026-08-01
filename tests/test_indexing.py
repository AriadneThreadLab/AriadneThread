"""Offline tests for BGE-M3 chunk indexing."""

from __future__ import annotations

from uuid import uuid4

import pytest
from app.core.errors import EmbeddingError
from app.db.models import EMBEDDING_DIMENSION
from app.embeddings.contracts import Vector
from app.rag.indexing import (
    ChunkForIndex,
    InMemoryChunkIndexStore,
    index_chunks,
    validate_embedding_vector,
)


def _unit(value: float = 0.0) -> Vector:
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[0] = 1.0 if value == 0.0 else value
    # Keep roughly unit length for the common case.
    if value == 0.0:
        return vector
    return vector


class FakeEmbedder:
    def __init__(self, *, fail: bool = False, bad_dim: bool = False) -> None:
        self.fail = fail
        self.bad_dim = bad_dim
        self.calls: list[list[str]] = []
        self.model_name = "fake"
        self.dimension = EMBEDDING_DIMENSION

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        self.calls.append(list(texts))
        if self.fail:
            raise EmbeddingError("provider down")
        if self.bad_dim:
            return [[0.1] * 8 for _ in texts]
        return [_unit() for _ in texts]

    async def embed_query(self, text: str) -> Vector:
        return (await self.embed_documents([text]))[0]

    async def aclose(self) -> None:
        return None


def test_validate_embedding_vector_requires_1024_dims():
    assert len(validate_embedding_vector(_unit())) == 1024
    with pytest.raises(EmbeddingError, match="1024"):
        validate_embedding_vector([0.1, 0.2])


async def test_index_embeds_null_chunks_in_batches_and_skips_existing():
    pending_id = uuid4()
    embedded_id = uuid4()
    store = InMemoryChunkIndexStore(
        chunks=[
            ChunkForIndex(
                chunk_id=pending_id,
                content="leisure=park marks a public park.",
                content_hash="a",
                has_embedding=False,
            ),
            ChunkForIndex(
                chunk_id=embedded_id,
                content="already embedded",
                content_hash="b",
                has_embedding=True,
            ),
        ],
        embeddings={embedded_id: _unit()},
    )
    embedder = FakeEmbedder()

    first = await index_chunks(store, embedder, batch_size=1, force=False)
    assert first.selected == 1
    assert first.embedded == 1
    assert first.skipped == 1
    assert first.batches == 1
    assert pending_id in store.embeddings
    assert embedder.calls == [["leisure=park marks a public park."]]

    second = await index_chunks(store, embedder, batch_size=8, force=False)
    assert second.selected == 0
    assert second.embedded == 0
    assert second.skipped == 2


async def test_force_reembeds_all_chunks():
    chunk_id = uuid4()
    store = InMemoryChunkIndexStore(
        chunks=[
            ChunkForIndex(
                chunk_id=chunk_id,
                content="changed chunk body",
                content_hash="new",
                has_embedding=True,
            )
        ],
        embeddings={chunk_id: _unit(0.5)},
    )
    embedder = FakeEmbedder()
    result = await index_chunks(store, embedder, force=True)
    assert result.embedded == 1
    assert result.skipped == 0
    assert store.embeddings[chunk_id] == _unit()


async def test_changed_chunk_without_embedding_is_selected():
    # Phase 2 replaces changed chunks with new rows (embedding NULL).
    chunk_id = uuid4()
    store = InMemoryChunkIndexStore(
        chunks=[
            ChunkForIndex(
                chunk_id=chunk_id,
                content="updated content",
                content_hash="changed",
                has_embedding=False,
            )
        ],
        embeddings={},
    )
    result = await index_chunks(store, FakeEmbedder(), force=False)
    assert result.embedded == 1
    assert chunk_id in store.embeddings


async def test_provider_failure_surfaces_as_embedding_error():
    store = InMemoryChunkIndexStore(
        chunks=[
            ChunkForIndex(
                chunk_id=uuid4(),
                content="text",
                content_hash="x",
                has_embedding=False,
            )
        ],
        embeddings={},
    )
    with pytest.raises(EmbeddingError, match="provider down"):
        await index_chunks(store, FakeEmbedder(fail=True))


async def test_bad_dimension_from_provider_is_rejected():
    store = InMemoryChunkIndexStore(
        chunks=[
            ChunkForIndex(
                chunk_id=uuid4(),
                content="text",
                content_hash="x",
                has_embedding=False,
            )
        ],
        embeddings={},
    )
    with pytest.raises(EmbeddingError, match="1024"):
        await index_chunks(store, FakeEmbedder(bad_dim=True))
