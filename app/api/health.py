"""Liveness and readiness endpoints.

``/health`` is process liveness only. ``/ready`` performs short, explicit
reachability checks and never loads BGE-M3 or runs Overpass/LLM generation.
"""

from __future__ import annotations

import asyncio
import logging
from importlib import import_module
from importlib.util import find_spec

import httpx
from fastapi import APIRouter, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from app import __version__
from app.api.dependencies import DatabaseDep, SettingsDep
from app.db.session import Database

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


class GeoLoadSTHealth(BaseModel):
    """Energy-engine probe. Never imports OSM tools or GeoLoadST algorithms."""

    model_config = ConfigDict(frozen=True)

    geoloadst_available: bool
    version: str | None = None
    capabilities: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service liveness and effective configuration summary."""

    model_config = ConfigDict(frozen=True)

    status: str
    version: str
    app_env: str
    llm_model: str
    llm_provider: str
    embedding_model: str
    embedding_dim: int
    geoloadst: GeoLoadSTHealth


class ReadyCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    ok: bool
    detail: str | None = None


class ReadyResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    checks: list[ReadyCheck] = Field(default_factory=list)


@router.get("/health", response_model=HealthResponse, summary="Service liveness")
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        app_env=settings.app_env,
        llm_model=settings.llm_model,
        llm_provider=settings.llm_provider,
        embedding_model=settings.bge_model_name,
        embedding_dim=settings.embedding_dim,
        geoloadst=geoloadst_health(),
    )


@router.get(
    "/health/geoloadst",
    response_model=GeoLoadSTHealth,
    summary="GeoLoadST engine health",
)
async def geoloadst_health_endpoint() -> GeoLoadSTHealth:
    return geoloadst_health()


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Service readiness (lightweight dependency checks)",
    responses={503: {"description": "One or more required dependencies are unavailable."}},
)
async def ready(
    settings: SettingsDep,
    database: DatabaseDep,
    response: Response,
) -> ReadyResponse:
    checks = [
        await _check_database(database, timeout=settings.ready_check_timeout_seconds),
        await _check_llm_provider(settings),
    ]
    ok = all(check.ok for check in checks)
    response.status_code = 200 if ok else 503
    return ReadyResponse(status="ready" if ok else "not_ready", checks=checks)


def geoloadst_health() -> GeoLoadSTHealth:
    """Read the plugin health payload. Does not import ``geoloadst`` or OSM clients."""
    fallback = GeoLoadSTHealth(
        geoloadst_available=False,
        version=None,
        capabilities=["moran_lisa"],
    )
    if find_spec("ariadne_geoloadst") is None:
        return fallback
    try:
        module = import_module("ariadne_geoloadst")
        reporter = getattr(module, "health_status", None)
        if not callable(reporter):
            return fallback
        payload = reporter()
    except Exception as exc:
        logger.warning("geoloadst health probe failed: %s", type(exc).__name__)
        return fallback
    if not isinstance(payload, dict):
        return fallback
    capabilities = [str(item) for item in payload.get("capabilities", ()) if isinstance(item, str)]
    if "moran_lisa" not in capabilities:
        capabilities = ["moran_lisa", *capabilities]
    version = payload.get("version")
    return GeoLoadSTHealth(
        geoloadst_available=bool(payload.get("geoloadst_available")),
        version=version if isinstance(version, str) and version else None,
        capabilities=capabilities,
    )


async def _check_database(database: Database, *, timeout: float) -> ReadyCheck:
    async def _probe() -> None:
        async with database.session() as session:
            await session.execute(text("SELECT 1"))

    try:
        await asyncio.wait_for(_probe(), timeout=timeout)
    except Exception as exc:
        logger.warning("readiness database check failed: %s", type(exc).__name__)
        return ReadyCheck(name="database", ok=False, detail="unreachable")
    return ReadyCheck(name="database", ok=True)


async def _check_llm_provider(settings: SettingsDep) -> ReadyCheck:
    if settings.llm_provider == "avalai":
        return await _check_avalai(settings)
    return await _check_ollama(settings)


async def _check_ollama(settings: SettingsDep) -> ReadyCheck:
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        async with httpx.AsyncClient(timeout=settings.ready_check_timeout_seconds) as client:
            response = await client.get(url)
            response.raise_for_status()
            body = response.json()
    except Exception as exc:
        logger.warning("readiness ollama check failed: %s", type(exc).__name__)
        return ReadyCheck(name="ollama", ok=False, detail="unreachable")
    models = body.get("models") if isinstance(body, dict) else None
    if not isinstance(models, list):
        return ReadyCheck(name="ollama", ok=False, detail="invalid_response")
    names = {
        str(entry.get("name")) for entry in models if isinstance(entry, dict) and entry.get("name")
    }
    if settings.ollama_model not in names:
        return ReadyCheck(
            name="ollama",
            ok=False,
            detail=f"model_missing:{settings.ollama_model}",
        )
    return ReadyCheck(name="ollama", ok=True)


async def _check_avalai(settings: SettingsDep) -> ReadyCheck:
    """Reachability + auth only. The gateway catalogue may omit routed models."""
    if not (settings.avalai_api_key or "").strip():
        return ReadyCheck(name="avalai", ok=False, detail="missing_api_key")
    url = settings.avalai_base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {settings.avalai_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=settings.ready_check_timeout_seconds) as client:
            response = await client.get(url, headers=headers)
    except Exception as exc:
        logger.warning("readiness avalai check failed: %s", type(exc).__name__)
        return ReadyCheck(name="avalai", ok=False, detail="unreachable")
    if response.status_code in {401, 403}:
        return ReadyCheck(name="avalai", ok=False, detail="authentication_failed")
    if response.status_code >= 500:
        return ReadyCheck(name="avalai", ok=False, detail="unavailable")
    if response.status_code >= 400:
        return ReadyCheck(name="avalai", ok=False, detail=f"http_{response.status_code}")
    return ReadyCheck(name="avalai", ok=True)
