"""Embedding provider must stay lazy and offline until explicitly used."""

from __future__ import annotations

import math
import sys

import pytest
from app.core.errors import EmbeddingError
from app.embeddings.bge_m3 import BGE_M3_DIMENSION, BgeM3EmbeddingProvider
from app.embeddings.cache import huggingface_model_cache_dir, is_model_cached, require_model_cached
from app.embeddings.contracts import EmbeddingProvider


def test_importing_the_provider_does_not_import_torch():
    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


def test_construction_does_not_load_the_model():
    provider = BgeM3EmbeddingProvider(model_name="BAAI/bge-m3", device="cpu")
    assert not provider.is_loaded
    assert provider.model_name == "BAAI/bge-m3"
    assert provider.device == "cpu"
    assert provider.dimension == BGE_M3_DIMENSION == 1024


def test_provider_satisfies_the_embedding_interface():
    assert isinstance(BgeM3EmbeddingProvider(), EmbeddingProvider)


def test_cache_helpers_detect_bge_m3_layout():
    # The machine under test is expected to already have the model cached.
    # This assertion documents the environment without downloading anything.
    assert huggingface_model_cache_dir("BAAI/bge-m3").name == "models--BAAI--bge-m3"
    if is_model_cached("BAAI/bge-m3"):
        assert require_model_cached("BAAI/bge-m3").is_dir()
    with pytest.raises(EmbeddingError, match="was not found"):
        require_model_cached("BAAI/definitely-missing-model-xyz")


async def test_embedding_no_documents_short_circuits_before_loading():
    provider = BgeM3EmbeddingProvider()
    assert await provider.embed_documents([]) == []
    assert not provider.is_loaded


async def test_empty_query_is_rejected_before_loading():
    provider = BgeM3EmbeddingProvider()
    with pytest.raises(EmbeddingError, match="empty"):
        await provider.embed_query("   ")
    assert not provider.is_loaded


async def test_load_failure_is_reported_as_a_domain_error(monkeypatch):
    provider = BgeM3EmbeddingProvider(model_name="does-not-exist", device="cpu")

    def fail() -> None:
        raise EmbeddingError("could not load embedding model 'does-not-exist'")

    monkeypatch.setattr(provider, "_load_model", fail)
    with pytest.raises(EmbeddingError, match="does-not-exist"):
        await provider.embed_query("parks in Berlin")


async def test_encoding_uses_the_loaded_model_and_validates_normalised_dimensions(
    monkeypatch,
):
    class StubModel:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode(self, texts, **kwargs):
            self.calls.append(list(texts))
            vector = [0.0] * BGE_M3_DIMENSION
            vector[0] = 1.0
            return [vector for _ in texts]

    stub = StubModel()
    provider = BgeM3EmbeddingProvider(batch_size=4, normalize=True)
    monkeypatch.setattr(provider, "_load_model", lambda: stub)

    vectors = await provider.embed_documents(["park", "پارک"])
    assert stub.calls == [["park", "پارک"]]
    assert [len(vector) for vector in vectors] == [BGE_M3_DIMENSION, BGE_M3_DIMENSION]
    assert math.isclose(math.sqrt(sum(v * v for v in vectors[0])), 1.0, abs_tol=1e-6)
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


async def test_non_normalised_output_is_rejected_when_normalize_enabled(monkeypatch):
    class BadNormModel:
        def encode(self, texts, **kwargs):
            return [[2.0] * BGE_M3_DIMENSION for _ in texts]

    provider = BgeM3EmbeddingProvider(normalize=True)
    monkeypatch.setattr(provider, "_load_model", lambda: BadNormModel())
    with pytest.raises(EmbeddingError, match="normalised"):
        await provider.embed_query("park")
