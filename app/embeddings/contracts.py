"""Provider-neutral embedding contracts.

Documents and queries are separate methods because retrieval models may use
different prefixes or pooling for each side. BGE-M3 does not require a query
instruction, but keeping the distinction avoids a breaking change later.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

Vector = list[float]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Dense text embedding interface."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int:
        """Vector width. Must match the database ``VECTOR(n)`` column."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed corpus passages, preserving input order."""
        ...

    async def embed_query(self, text: str) -> Vector:
        """Embed a single search query."""
        ...

    async def aclose(self) -> None:
        """Release model resources (GPU memory, worker threads)."""
        ...
