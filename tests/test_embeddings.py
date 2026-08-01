"""Embedding provider must stay lazy and offline until explicitly used."""

from __future__ import annotations

import sys

import pytest
from app.core.errors import EmbeddingError
from app.embeddings.bge_m3 import BGE_M3_DIMENSION, BgeM3EmbeddingProvider
from app.embeddings.contracts import EmbeddingProvider


def test_importing_the_provider_does_not_import_torch():
    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


def test_construction_does_not_load_the_model():
    provider = BgeM3EmbeddingProvider(model_name="BAAI/bge-m3", device="cpu")
    assert not provider.is_loaded
    assert provider.model_name == "BAAI/bge-m3"
    assert provider.dimension == BGE_M3_DIMENSION == 1024


def test_provider_satisfies_the_embedding_interface():
    assert isinstance(BgeM3EmbeddingProvider(), EmbeddingProvider)


async def test_embedding_no_documents_short_circuits_before_loading():
    provider = BgeM3EmbeddingProvider()
    assert await provider.embed_documents([]) == []
    assert not provider.is_loaded


async def test_load_failure_is_reported_as_a_domain_error(monkeypatch):
    provider = BgeM3EmbeddingProvider(model_name="does-not-exist", device="cpu")

    def fail() -> None:
        raise EmbeddingError("could not load embedding model 'does-not-exist'")

    monkeypatch.setattr(provider, "_load_model", fail)
    with pytest.raises(EmbeddingError, match="does-not-exist"):
        await provider.embed_query("parks in Berlin")


async def test_encoding_uses_the_loaded_model_and_validates_dimensions(monkeypatch):
    class StubModel:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode(self, texts, **kwargs):
            self.calls.append(list(texts))
            return [[0.1] * BGE_M3_DIMENSION for _ in texts]

    stub = StubModel()
    provider = BgeM3EmbeddingProvider(batch_size=4)
    monkeypatch.setattr(provider, "_load_model", lambda: stub)

    vectors = await provider.embed_documents(["park", "پارک"])
    assert stub.calls == [["park", "پارک"]]
    assert [len(vector) for vector in vectors] == [BGE_M3_DIMENSION, BGE_M3_DIMENSION]
    assert provider.is_loaded

    await provider.aclose()
    assert not provider.is_loaded


async def test_wrong_dimension_output_is_rejected(monkeypatch):
    class BadModel:
        def encode(self, texts, **kwargs):
            return [[0.1] * 768 for _ in texts]

    provider = BgeM3EmbeddingProvider()
    monkeypatch.setattr(provider, "_load_model", lambda: BadModel())
    with pytest.raises(EmbeddingError, match="1024"):
        await provider.embed_query("park")
