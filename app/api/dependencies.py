"""FastAPI dependencies.

Shared objects are read from ``app.state``, which the lifespan handler owns, so
no request handler reaches for a module-level singleton.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.active_learning.service import ActiveLearningService
from app.agent.contracts import GeoAgent
from app.core.config import Settings
from app.db.session import Database


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_active_learning(request: Request) -> ActiveLearningService | None:
    service: ActiveLearningService | None = getattr(request.app.state, "active_learning", None)
    return service


def get_database(request: Request) -> Database:
    database: Database = request.app.state.database
    return database


def get_geo_agent(request: Request) -> GeoAgent:
    agent: GeoAgent = request.app.state.geo_agent
    return agent


SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[Database, Depends(get_database)]
GeoAgentDep = Annotated[GeoAgent, Depends(get_geo_agent)]
ActiveLearningDep = Annotated[ActiveLearningService | None, Depends(get_active_learning)]
