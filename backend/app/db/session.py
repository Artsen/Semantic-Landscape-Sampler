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
    connect_args=(
        {"check_same_thread": False}
        if _settings.database_url.startswith("sqlite")
        else {}
    ),
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

    try:
        await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        await conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
        await conn.exec_driver_sql("PRAGMA mmap_size=268435456")
    except Exception:
        pass

    result = await conn.exec_driver_sql("PRAGMA table_info(clusters)")

    columns = {row[1] for row in result.fetchall()}

    if "outlier_score" not in columns:

        await conn.exec_driver_sql(
            "ALTER TABLE clusters ADD COLUMN outlier_score FLOAT"
        )

    result = await conn.exec_driver_sql("PRAGMA table_info(runs)")

    run_columns = {row[1] for row in result.fetchall()}

    if "progress_stage" not in run_columns:

        await conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN progress_stage TEXT")

    if "progress_message" not in run_columns:

        await conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN progress_message TEXT")

    if "progress_percent" not in run_columns:

        await conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN progress_percent FLOAT")

    if "progress_metadata" not in run_columns:

        await conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN progress_metadata TEXT")

    if "use_cache" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN use_cache BOOLEAN DEFAULT 1"
        )

    if "preproc_version" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN preproc_version TEXT DEFAULT 'legacy-v0'"
        )

    if "umap_n_neighbors" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN umap_n_neighbors INTEGER DEFAULT 30"
        )

    if "umap_min_dist" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN umap_min_dist FLOAT DEFAULT 0.3"
        )

    if "umap_metric" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN umap_metric TEXT DEFAULT 'cosine'"
        )

    if "umap_seed" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN umap_seed INTEGER"
        )

    if "random_state_seed_source" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN random_state_seed_source TEXT DEFAULT 'default'"
        )

    if "trustworthiness_2d" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN trustworthiness_2d FLOAT"
        )

    if "trustworthiness_3d" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN trustworthiness_3d FLOAT"
        )

    if "continuity_2d" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN continuity_2d FLOAT"
        )

    if "continuity_3d" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN continuity_3d FLOAT"
        )

    if "cluster_algo" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN cluster_algo TEXT DEFAULT 'hdbscan'"
        )

    if "hdbscan_min_cluster_size" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN hdbscan_min_cluster_size INTEGER"
        )

    if "hdbscan_min_samples" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN hdbscan_min_samples INTEGER"
        )

    if "processing_time_ms" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN processing_time_ms FLOAT"
        )

    if "timings_json" not in run_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE runs ADD COLUMN timings_json TEXT"
        )

    result = await conn.exec_driver_sql("PRAGMA table_info(response_segments)")

    segment_columns = {row[1] for row in result.fetchall()}

    if "text_hash" not in segment_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE response_segments ADD COLUMN text_hash TEXT"
        )

    if "is_cached" not in segment_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE response_segments ADD COLUMN is_cached BOOLEAN DEFAULT 0"
        )

    if "is_duplicate" not in segment_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE response_segments ADD COLUMN is_duplicate BOOLEAN DEFAULT 0"
        )

    if "simhash64" not in segment_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE response_segments ADD COLUMN simhash64 BIGINT"
        )

    if "chunk_overlap" not in run_columns:

        await conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN chunk_overlap INTEGER")

    result = await conn.exec_driver_sql("PRAGMA table_info(embedding_cache)")

    cache_columns = {row[1] for row in result.fetchall()}

    if "preproc_version" not in cache_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE embedding_cache ADD COLUMN preproc_version TEXT DEFAULT 'legacy-v0'"
        )

    if "provider" not in cache_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE embedding_cache ADD COLUMN provider TEXT"
        )

    if "model_revision" not in cache_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE embedding_cache ADD COLUMN model_revision TEXT"
        )

    if "vector_dtype" not in cache_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE embedding_cache ADD COLUMN vector_dtype TEXT DEFAULT 'float32'"
        )

    if "vector_norm" not in cache_columns:

        await conn.exec_driver_sql(
            "ALTER TABLE embedding_cache ADD COLUMN vector_norm FLOAT"
        )

    await conn.exec_driver_sql(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_embedding_cache_hash_model_preproc ON embedding_cache (text_hash, model_id, preproc_version)"
    )

