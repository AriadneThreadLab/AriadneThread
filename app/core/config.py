"""Application settings.

One settings object for the whole application, loaded from environment
variables and an optional local ``.env`` file. Settings are grouped by concern
but kept in a single flat model so that the environment contract stays obvious.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AppEnv = Literal["local", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LLMProviderName = Literal["ollama", "avalai"]
OllamaToolMode = Literal["prompted", "native"]
AvalAIToolMode = Literal["native", "prompted"]

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
    # Required from the environment / .env. No password-bearing default is
    # committed; see .env.example for the expected shape.
    database_url: str = Field(
        description="Async SQLAlchemy DSN for this project's own database.",
    )

    # --- LLM provider selection ---
    # ``ollama`` keeps the local DeepSeek path unchanged. ``avalai`` uses the
    # OpenAI-compatible AvalAI gateway (Gemini 3.6 Flash by default).
    llm_provider: LLMProviderName = "ollama"

    # --- LLM (Ollama) ---
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "deepseek-r1:7b"
    ollama_num_ctx: int = Field(default=4096, ge=512, le=131072)
    ollama_request_timeout_seconds: float = Field(default=180.0, gt=0)
    # Explicit tool protocol. ``prompted`` is the safe default for deepseek-r1:7b:
    # Ollama may accept native ``tools`` (HTTP 200) without the model emitting
    # reliable native tool_calls. Do not infer this from advertised capabilities.
    ollama_tool_mode: OllamaToolMode = "prompted"

    # --- LLM (AvalAI OpenAI-compatible) ---
    # API key is required only when ``LLM_PROVIDER=avalai``. Never logged.
    avalai_api_key: str | None = Field(default=None, repr=False)
    avalai_base_url: str = "https://api.avalai.ir/v1"
    avalai_model: str = "gemini-3.6-flash"
    avalai_timeout_seconds: float = Field(default=120.0, gt=0)
    # Prefer native OpenAI tool calling; prompted is an explicit fallback.
    avalai_tool_mode: AvalAIToolMode = "native"
    # Total attempts = 1 initial + (max_attempts - 1) retries. Transient only.
    avalai_max_attempts: int = Field(default=2, ge=1, le=3)
    avalai_retry_backoff_seconds: float = Field(default=0.5, gt=0, le=5.0)

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
    # Deterministic client-side retry for transient upstream failures only.
    # Total attempts = 1 initial + (max_attempts - 1) retries. No endpoint fallback.
    overpass_max_attempts: int = Field(default=2, ge=1, le=3)
    overpass_retry_backoff_seconds: float = Field(default=1.5, gt=0, le=30.0)

    # --- Nominatim (trusted place resolution) ---
    nominatim_base_url: str = "https://nominatim.openstreetmap.org/search"
    nominatim_timeout_seconds: float = Field(default=15.0, gt=0, le=60.0)

    # --- Retrieval ---
    rag_top_k: int = Field(default=5, ge=1, le=50)

    # --- OSM wiki ingestion ---
    osm_wiki_api_url: str = "https://wiki.openstreetmap.org/w/api.php"
    osm_wiki_user_agent: str = (
        "Ariadne-Thread/0.1 (+https://github.com/AriadneThreadLab/ariadne-thread; research)"
    )
    osm_wiki_timeout_seconds: float = Field(default=30.0, gt=0)
    osm_wiki_max_retries: int = Field(default=2, ge=0, le=5)
    rag_chunk_max_chars: int = Field(default=2000, ge=400, le=8000)
    rag_chunk_min_chars: int = Field(default=200, ge=50, le=2000)

    # --- Agent loop bounds ---
    # Multi-target comparison needs grounding + resolve + dual query + analyze.
    agent_max_tool_rounds: int = Field(default=6, ge=1, le=16)
    agent_max_tool_calls: int = Field(default=12, ge=1, le=32)
    # Total wall-clock budget for one HTTP agent request (distinct from Ollama /
    # Overpass / per-tool timeouts).
    agent_request_timeout_seconds: float = Field(default=240.0, gt=0)
    # Short timeout for readiness probes only.
    ready_check_timeout_seconds: float = Field(default=2.0, gt=0, le=30.0)

    # --- Execution memory (structured analytical snapshots, not chat history) ---
    # Separate from Active Learning. Never trains or stores chain-of-thought.
    execution_memory_enabled: bool = True
    #: Live OSM datasets older than this are refreshed unless the user asks to keep them.
    execution_memory_osm_ttl_seconds: int = Field(default=86_400, ge=60, le=2_592_000)
    #: Drop executions older than this, except the latest in the conversation.
    execution_memory_max_age_seconds: int = Field(default=604_800, ge=3600, le=31_536_000)
    #: Hard cap of snapshots retained per conversation_id.
    execution_memory_max_per_conversation: int = Field(default=20, ge=2, le=200)

    # --- Active learning (candidate selection for human review) ---
    # Selection only. Nothing in this group can start training, change model
    # weights or promote a model; that happens in the separate finetuning
    # pipeline, which reads exported dataset files.
    active_learning_enabled: bool = True
    #: Below this informativeness score a run is discarded, not stored.
    active_learning_min_score_to_store: float = Field(default=0.20, ge=0.0, le=1.0)
    #: At or above this score a stored candidate joins the human review queue.
    active_learning_min_score_for_review: float = Field(default=0.35, ge=0.0, le=1.0)
    #: Token-overlap level at which two queries count as the same question.
    active_learning_novelty_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    #: Hard cap on stored candidates sharing one task signature.
    active_learning_max_duplicates_per_signature: int = Field(default=5, ge=1, le=1000)
    #: Bounded in-process memory of recent runs so late feedback can still
    #: materialise a candidate for a run that was filtered out.
    active_learning_recent_run_cache: int = Field(default=256, ge=0, le=10000)

    @field_validator("database_url")
    @classmethod
    def _require_async_driver(cls, value: str) -> str:
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError("DATABASE_URL must use the postgresql+asyncpg driver")
        return value

    @model_validator(mode="after")
    def _require_avalai_api_key(self) -> Settings:
        if self.llm_provider == "avalai" and not (self.avalai_api_key or "").strip():
            raise ValueError("AVALAI_API_KEY is required when LLM_PROVIDER=avalai")
        return self

    @property
    def embedding_dim(self) -> int:
        return BGE_M3_EMBEDDING_DIM

    @property
    def llm_model(self) -> str:
        """Model id of the configured chat provider (never a secret)."""
        if self.llm_provider == "avalai":
            return self.avalai_model
        return self.ollama_model

    @property
    def llm_endpoint_host(self) -> str:
        """Hostname:port of the active chat provider. Never includes a key."""
        url = self.avalai_base_url if self.llm_provider == "avalai" else self.ollama_base_url
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if not host:
            return "unknown"
        return f"{host}:{parsed.port}" if parsed.port else host


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    Cached rather than module-global so that tests can clear the cache after
    changing the environment.
    """
    return Settings()
