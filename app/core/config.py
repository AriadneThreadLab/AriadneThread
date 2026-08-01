"""Application settings.

One settings object for the whole application, loaded from environment
variables and an optional local ``.env`` file. Settings are grouped by concern
but kept in a single flat model so that the environment contract stays obvious.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AppEnv = Literal["local", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

#: Dense embedding width of BAAI/bge-m3. The database vector column and the
#: embedding provider must agree on this value.
BGE_M3_EMBEDDING_DIM = 1024


class Settings(BaseSettings):
    """Runtime configuration for the OSM GeoAgent service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Application ---
    app_env: AppEnv = "local"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8100, ge=1, le=65535)
    log_level: LogLevel = "INFO"

    # --- Database ---
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@127.0.0.1:5433/osm_geoagent",
        description="Async SQLAlchemy DSN for this project's own database.",
    )

    # --- LLM (Ollama) ---
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "deepseek-r1:7b"
    ollama_num_ctx: int = Field(default=4096, ge=512, le=131072)
    ollama_request_timeout_seconds: float = Field(default=180.0, gt=0)

    # --- Embeddings ---
    bge_model_name: str = "BAAI/bge-m3"
    # Prefer cpu when Ollama is already using the GPU; set to cuda when VRAM allows.
    bge_device: str = "cpu"
    bge_batch_size: int = Field(default=8, ge=1, le=256)

    # --- Overpass ---
    overpass_base_url: str = "https://overpass-api.de/api/interpreter"
    overpass_timeout_seconds: float = Field(default=60.0, gt=0)
    overpass_max_results: int = Field(default=1000, ge=1, le=10000)
    overpass_max_response_bytes: int = Field(default=20_000_000, ge=1024)

    # --- Retrieval ---
    rag_top_k: int = Field(default=5, ge=1, le=50)

    # --- OSM wiki ingestion ---
    osm_wiki_api_url: str = "https://wiki.openstreetmap.org/w/api.php"
    osm_wiki_user_agent: str = "OSM-GeoAgent/0.1 (+https://github.com/local/osm-geoagent; research)"
    osm_wiki_timeout_seconds: float = Field(default=30.0, gt=0)
    osm_wiki_max_retries: int = Field(default=2, ge=0, le=5)
    rag_chunk_max_chars: int = Field(default=2000, ge=400, le=8000)
    rag_chunk_min_chars: int = Field(default=200, ge=50, le=2000)

    # --- Agent loop bounds ---
    agent_max_tool_rounds: int = Field(default=4, ge=1, le=16)
    agent_max_tool_calls: int = Field(default=8, ge=1, le=32)

    @field_validator("database_url")
    @classmethod
    def _require_async_driver(cls, value: str) -> str:
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError("DATABASE_URL must use the postgresql+asyncpg driver")
        return value

    @property
    def embedding_dim(self) -> int:
        return BGE_M3_EMBEDDING_DIM


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    Cached rather than module-global so that tests can clear the cache after
    changing the environment.
    """
    return Settings()
