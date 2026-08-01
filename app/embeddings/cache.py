"""Local Hugging Face cache inspection for BGE-M3.

The project must not silently download large models. Callers check the cache
before loading and fail with a clear setup message when files are missing.
"""

from __future__ import annotations

from pathlib import Path

from app.core.errors import EmbeddingError

_WEIGHT_NAMES = (
    "pytorch_model.bin",
    "model.safetensors",
    "pytorch_model.safetensors",
)


def huggingface_model_cache_dir(model_name: str) -> Path:
    """Return the expected ``~/.cache/huggingface/hub/models--...`` directory."""
    slug = "models--" + model_name.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / slug


def is_model_cached(model_name: str) -> bool:
    """Whether a usable local snapshot of ``model_name`` appears to exist."""
    root = huggingface_model_cache_dir(model_name)
    if not root.is_dir():
        return False
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return False
    for snapshot in snapshots.iterdir():
        if not snapshot.is_dir():
            continue
        if (snapshot / "config.json").is_file() and any(
            (snapshot / name).is_file() for name in _WEIGHT_NAMES
        ):
            return True
    return False


def require_model_cached(model_name: str) -> Path:
    """Return the cache root, or raise with setup instructions."""
    root = huggingface_model_cache_dir(model_name)
    if is_model_cached(model_name):
        return root
    raise EmbeddingError(
        f"Embedding model '{model_name}' was not found in the local Hugging Face "
        f"cache at {root}. Place the model there before indexing, or restore it "
        "from an existing cache. This project will not download BGE-M3 automatically."
    )
