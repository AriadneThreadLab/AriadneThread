"""Compose Execution Memory for the application process."""

from __future__ import annotations

from app.core.config import Settings
from app.db.session import Database
from app.execution_memory.repository import (
    InMemoryExecutionMemoryRepository,
    PostgresExecutionMemoryRepository,
)
from app.execution_memory.service import ExecutionMemoryService
from app.llm.contracts import LLMProvider


def build_execution_memory_service(
    settings: Settings,
    database: Database | None,
    llm: LLMProvider | None = None,
    *,
    in_memory: bool = False,
) -> ExecutionMemoryService | None:
    if not settings.execution_memory_enabled:
        return None
    if in_memory or database is None:
        repo: InMemoryExecutionMemoryRepository | PostgresExecutionMemoryRepository = (
            InMemoryExecutionMemoryRepository()
        )
    else:
        repo = PostgresExecutionMemoryRepository(database)
    return ExecutionMemoryService(repo, settings, llm)
