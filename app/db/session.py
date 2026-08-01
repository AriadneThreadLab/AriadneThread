"""Async engine and session lifecycle.

The engine is owned by the application (created in the lifespan handler and
stored on ``app.state``) rather than by a module-level global, so tests and
scripts can create and dispose their own without side effects at import time.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Connection pool settings."""

    url: str
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 5
    pool_pre_ping: bool = True


class Database:
    """Owns one async engine and its session factory."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._engine: AsyncEngine = create_async_engine(
            config.url,
            echo=config.echo,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_pre_ping=config.pool_pre_ping,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            autoflush=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session, committing on success and rolling back on error."""
        async with self._session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            else:
                await session.commit()

    async def dispose(self) -> None:
        await self._engine.dispose()
