"""Construct embedding providers from settings without loading the model."""

from __future__ import annotations

from app.core.config import Settings
from app.embeddings.bge_m3 import BgeM3EmbeddingProvider


def build_bge_m3_provider(settings: Settings) -> BgeM3EmbeddingProvider:
    """Return a lazy BGE-M3 provider configured from settings."""
    return BgeM3EmbeddingProvider(
        model_name=settings.bge_model_name,
        device=settings.bge_device,
        batch_size=settings.bge_batch_size,
        normalize=True,
        local_files_only=True,
    )
