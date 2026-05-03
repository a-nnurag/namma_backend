"""
Async SQLAlchemy engine and session factory.

get_db() is a FastAPI dependency that yields a PostgreSQLAdapter per request.
The session is committed on success and rolled back on any exception.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import settings
from core.exceptions import DBConnectionError
from core.logging import get_logger
from db.adapter import PostgreSQLAdapter

log = get_logger(__name__)

_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

_AsyncSessionFactory = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[PostgreSQLAdapter, None]:
    """FastAPI dependency — yields a PostgreSQLAdapter for the current request."""
    async with _AsyncSessionFactory() as session:
        try:
            adapter = PostgreSQLAdapter(session)
            yield adapter
            await session.commit()
            log.debug("DB session committed successfully")
        except Exception:
            await session.rollback()
            log.warning("DB session rolled back due to exception")
            raise


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[PostgreSQLAdapter, None]:
    """Context manager variant for use outside FastAPI (background tasks, startup)."""
    async with _AsyncSessionFactory() as session:
        try:
            adapter = PostgreSQLAdapter(session)
            yield adapter
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def verify_db_connection() -> None:
    """Called at startup to confirm DB is reachable. Raises DBConnectionError on failure."""
    from sqlalchemy import text

    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        log.info("Database connection verified")
    except Exception as exc:
        log.critical("Database connection failed at startup", exc_info=True)
        raise DBConnectionError(str(exc)) from exc


async def close_db() -> None:
    """Dispose connection pool — called on app shutdown."""
    await _engine.dispose()
    log.info("Database connection pool disposed")
