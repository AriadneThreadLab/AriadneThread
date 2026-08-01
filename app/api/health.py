"""Health endpoint.

Liveness only: it reports what the process is configured to use and must not
depend on PostgreSQL, Ollama or Overpass being reachable.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from app import __version__
from app.api.dependencies import SettingsDep

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    """Service liveness and effective configuration summary."""

    model_config = ConfigDict(frozen=True)

    status: str
    version: str
    app_env: str
    llm_model: str
    embedding_model: str
    embedding_dim: int


@router.get("/health", response_model=HealthResponse, summary="Service liveness")
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        app_env=settings.app_env,
        llm_model=settings.ollama_model,
        embedding_model=settings.bge_model_name,
        embedding_dim=settings.embedding_dim,
    )
