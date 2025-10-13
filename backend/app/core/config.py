"""Runtime configuration helpers.

Classes:
    Settings: Pydantic settings model capturing environment-driven defaults.

Functions:
    get_settings(): Return a cached Settings instance for dependency injection.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Semantic Landscape Sampler API"
    database_url: str = "sqlite+aiosqlite:///./data/semantic_sampler.db"
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_fallback_model: str = "text-embedding-3-small"
    default_temperature: float = 1.0
    default_top_p: float = 1.0
    default_max_tokens: int | None = None
    discourse_tagging_model: str | None = None
    enable_discourse_tagging: bool = True
    segment_similarity_threshold: float = 0.85
    segment_keyword_axes: list[str] = Field(default_factory=list)
    segment_word_window: int = 3
    segment_word_overlap: int = 1
    projection_min_dist: float = 0.25
    embedding_preproc_version: str = "norm-nfkc-v1"
    umap_default_n_neighbors: int = 30
    umap_default_min_dist: float = 0.3
    umap_default_metric: str = "cosine"
    umap_default_seed: int = 42
    tsne_preview_threshold: int = 12000
    tsne_preview_size: int = 10000
    default_env_label: str | None = None
    ann_index_dir: str = "./data/indexes"
    cluster_default_algo: str = "hdbscan"
    hdbscan_default_min_cluster_size: int = 30
    hdbscan_default_min_samples: int = 5


@lru_cache()
def get_settings() -> Settings:
    return Settings()



