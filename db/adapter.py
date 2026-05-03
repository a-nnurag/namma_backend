"""
Database Adapter pattern for NammaKelsa backend.

DatabaseAdapter is the abstract interface. All route handlers and service code
depend ONLY on DatabaseAdapter — never on SQLAlchemy internals directly.
PostgreSQLAdapter is the concrete implementation backed by an AsyncSession.

Swapping the database engine = swap the adapter, zero changes in business logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Type, TypeVar
from uuid import UUID

from sqlalchemy import select, update as sa_update, delete as sa_delete
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from core.exceptions import DBIntegrityError, DBQueryError
from core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class DatabaseAdapter(ABC):
    """Abstract interface for all database operations."""

    @abstractmethod
    async def get(self, model: Type[T], id: Any) -> T | None:
        """Fetch a single row by primary key. Returns None if not found."""

    @abstractmethod
    async def get_by(self, model: Type[T], **filters: Any) -> T | None:
        """Fetch first row matching all keyword filters. Returns None if not found."""

    @abstractmethod
    async def list_by(self, model: Type[T], **filters: Any) -> list[T]:
        """Fetch all rows matching all keyword filters."""

    @abstractmethod
    async def create(self, instance: T) -> T:
        """Persist a new ORM instance and return it (with DB-generated fields set)."""

    @abstractmethod
    async def update(self, instance: T, **fields: Any) -> T:
        """Update fields on an already-loaded ORM instance and persist."""

    @abstractmethod
    async def delete(self, instance: T) -> None:
        """Delete an ORM instance."""

    @abstractmethod
    async def execute_query(self, stmt: Any) -> Any:
        """Execute a raw SQLAlchemy statement and return the result."""

    @abstractmethod
    async def flush(self) -> None:
        """Flush pending changes to the DB (within current transaction)."""

    @abstractmethod
    async def refresh(self, instance: T) -> T:
        """Reload an instance from DB to pick up generated fields."""

    @abstractmethod
    async def exists(self, model: Type[T], **filters: Any) -> bool:
        """Return True if at least one row matches all keyword filters."""


class PostgreSQLAdapter(DatabaseAdapter):
    """
    Concrete DB adapter backed by an SQLAlchemy AsyncSession.

    Every public method wraps SQLAlchemy operations in consistent error handling:
      - IntegrityError   → DBIntegrityError
      - SQLAlchemyError  → DBQueryError
    Both are subclasses of DatabaseError so callers catch a single type.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, model: Type[T], id: Any) -> T | None:
        try:
            return await self._session.get(model, id)
        except SQLAlchemyError as exc:
            log.error(
                "DB get failed",
                model=model.__name__,
                id=str(id),
                exc_info=True,
            )
            raise DBQueryError("get", str(exc)) from exc

    async def get_by(self, model: Type[T], **filters: Any) -> T | None:
        try:
            stmt = select(model).filter_by(**filters)
            result = await self._session.execute(stmt)
            return result.scalars().first()
        except SQLAlchemyError as exc:
            log.error(
                "DB get_by failed",
                model=model.__name__,
                filters=filters,
                exc_info=True,
            )
            raise DBQueryError("get_by", str(exc)) from exc

    async def list_by(self, model: Type[T], **filters: Any) -> list[T]:
        try:
            stmt = select(model).filter_by(**filters)
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            log.error(
                "DB list_by failed",
                model=model.__name__,
                filters=filters,
                exc_info=True,
            )
            raise DBQueryError("list_by", str(exc)) from exc

    async def create(self, instance: T) -> T:
        try:
            self._session.add(instance)
            await self._session.flush()
            await self._session.refresh(instance)
            log.debug(
                "DB row created",
                model=type(instance).__name__,
                id=str(getattr(instance, "id", "?")),
            )
            return instance
        except IntegrityError as exc:
            await self._session.rollback()
            log.warning(
                "DB integrity error on create",
                model=type(instance).__name__,
                exc_info=True,
            )
            raise DBIntegrityError(str(exc.orig)) from exc
        except SQLAlchemyError as exc:
            await self._session.rollback()
            log.error(
                "DB create failed",
                model=type(instance).__name__,
                exc_info=True,
            )
            raise DBQueryError("create", str(exc)) from exc

    async def update(self, instance: T, **fields: Any) -> T:
        try:
            for key, value in fields.items():
                setattr(instance, key, value)
            await self._session.flush()
            await self._session.refresh(instance)
            log.debug(
                "DB row updated",
                model=type(instance).__name__,
                id=str(getattr(instance, "id", "?")),
                fields=list(fields.keys()),
            )
            return instance
        except IntegrityError as exc:
            await self._session.rollback()
            raise DBIntegrityError(str(exc.orig)) from exc
        except SQLAlchemyError as exc:
            await self._session.rollback()
            log.error(
                "DB update failed",
                model=type(instance).__name__,
                exc_info=True,
            )
            raise DBQueryError("update", str(exc)) from exc

    async def delete(self, instance: T) -> None:
        try:
            await self._session.delete(instance)
            await self._session.flush()
            log.debug(
                "DB row deleted",
                model=type(instance).__name__,
                id=str(getattr(instance, "id", "?")),
            )
        except SQLAlchemyError as exc:
            await self._session.rollback()
            log.error("DB delete failed", model=type(instance).__name__, exc_info=True)
            raise DBQueryError("delete", str(exc)) from exc

    async def execute_query(self, stmt: Any) -> Any:
        try:
            return await self._session.execute(stmt)
        except SQLAlchemyError as exc:
            log.error("DB execute_query failed", exc_info=True)
            raise DBQueryError("execute_query", str(exc)) from exc

    async def flush(self) -> None:
        try:
            await self._session.flush()
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise DBQueryError("flush", str(exc)) from exc

    async def refresh(self, instance: T) -> T:
        try:
            await self._session.refresh(instance)
            return instance
        except SQLAlchemyError as exc:
            raise DBQueryError("refresh", str(exc)) from exc

    async def exists(self, model: Type[T], **filters: Any) -> bool:
        try:
            stmt = select(model).filter_by(**filters).limit(1)
            result = await self._session.execute(stmt)
            return result.scalars().first() is not None
        except SQLAlchemyError as exc:
            raise DBQueryError("exists", str(exc)) from exc
