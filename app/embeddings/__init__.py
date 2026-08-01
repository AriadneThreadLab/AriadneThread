"""Embedding provider boundary."""

from app.embeddings.bge_m3 import BGE_M3_DIMENSION, BgeM3EmbeddingProvider
from app.embeddings.contracts import EmbeddingProvider, Vector

__all__ = [
    "BGE_M3_DIMENSION",
    "BgeM3EmbeddingProvider",
    "EmbeddingProvider",
    "Vector",
]
