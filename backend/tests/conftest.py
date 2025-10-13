import asyncio
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.main import app
from app.db.session import get_session
from app.models import Cluster, Embedding, Projection, Response, Run


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture()
async def session() -> AsyncGenerator[AsyncSession, None]:
    engine: AsyncEngine = create_async_engine("sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async_session = AsyncSession(engine, expire_on_commit=False)
    try:
        yield async_session
    finally:
        await async_session.close()
        await engine.dispose()


@pytest_asyncio.fixture()
async def client(session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    async def _override_session():
        yield session

    app.dependency_overrides[get_session] = _override_session
    async with AsyncClient(app=app, base_url="http://testserver") as test_client:
        yield test_client
    app.dependency_overrides.clear()
