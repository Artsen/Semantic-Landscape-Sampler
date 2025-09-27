"""Database engine and session utilities.

Attributes:
    engine (AsyncEngine): Primary SQLModel async engine connected to SQLite.
    SessionLocal (async_sessionmaker): Factory for yielding AsyncSession objects.

Functions:
    init_db(): Create database tables and ensure SQLite schema patches are applied.
    get_session(): Dependency that yields an AsyncSession for request handlers.
"""

from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import get_settings

_settings = get_settings()

_db_path = Path(_settings.database_url.replace("sqlite+aiosqlite:///", "")).resolve()
if _db_path.parent.name:
    _db_path.parent.mkdir(parents=True, exist_ok=True)

engine: AsyncEngine = create_async_engine(
    _settings.database_url,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {},
)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        if _settings.database_url.startswith("sqlite"):
            await _ensure_sqlite_schema(conn)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


async def _ensure_sqlite_schema(conn) -> None:
    """Apply lightweight, idempotent schema patches for SQLite."""

    result = await conn.exec_driver_sql("PRAGMA table_info(clusters)")
    columns = {row[1] for row in result.fetchall()}
    if "outlier_score" not in columns:
        await conn.exec_driver_sql("ALTER TABLE clusters ADD COLUMN outlier_score FLOAT")

