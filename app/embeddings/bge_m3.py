"""BGE-M3 embedding provider.

The model is multilingual, which is what makes Persian queries retrievable
against an English OSM documentation corpus.

Loading is lazy and guarded by a lock: importing this module must never import
torch, allocate GPU memory, or touch the network. Encoding is blocking, so it
runs in a worker thread.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from app.core.errors import EmbeddingError
from app.embeddings.contracts import Vector

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from sentence_transformers import SentenceTransformer

#: Dense output width of BAAI/bge-m3.
BGE_M3_DIMENSION = 1024


class BgeM3EmbeddingProvider:
    """Sentence-Transformers backed provider for ``BAAI/bge-m3``.

    Requires the optional ``embeddings`` extra. Instantiating this class is
    cheap; the first ``embed_*`` call performs the actual model load.
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 8,
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._model: SentenceTransformer | None = None
        self._load_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return BGE_M3_DIMENSION

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        return await self._encode(texts)

    async def embed_query(self, text: str) -> Vector:
        vectors = await self._encode([text])
        return vectors[0]

    async def aclose(self) -> None:
        self._model = None

    # --- internals ---

    async def _encode(self, texts: list[str]) -> list[Vector]:
        model = await self._ensure_model()
        try:
            raw: Any = await asyncio.to_thread(
                model.encode,
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"BGE-M3 encoding failed: {exc}") from exc
        vectors = [[float(value) for value in row] for row in raw]
        for vector in vectors:
            if len(vector) != BGE_M3_DIMENSION:
                raise EmbeddingError(
                    f"Expected {BGE_M3_DIMENSION}-dim embeddings, got {len(vector)}"
                )
        return vectors

    async def _ensure_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model)
        return self._model

    def _load_model(self) -> SentenceTransformer:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise EmbeddingError(
                "sentence-transformers is not installed; install the 'embeddings' extra"
            ) from exc
        try:
            return SentenceTransformer(self._model_name, device=self._device)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model '{self._model_name}' on '{self._device}': {exc}"
            ) from exc
