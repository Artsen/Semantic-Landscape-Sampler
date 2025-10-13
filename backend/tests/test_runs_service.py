import json
from uuid import UUID

import numpy as np
import pytest

from sqlalchemy import select

from app.models import AnnIndex, RunStatus
from app.schemas import (
    RunCreateRequest,
    RunUpdateRequest,
    SampleRequest,
    UMAPParamsRequest,
)
from app.services.openai_client import ChatSample, EmbeddingBatch
from app.services.runs import RunService, load_run_with_details
from app.utils.text import normalise_for_embedding


class FakeOpenAIService:
    def __init__(self) -> None:
        self.embed_calls: int = 0
        self.embed_payloads: list[list[str]] = []
        self.sample_calls: int = 0

    @property
    def is_configured(self) -> bool:
        return False

    async def sample_chat(self, **_: object) -> list[ChatSample]:
        self.sample_calls += 1
        base_texts = [
            "Answer variant one with unique framing.",
            "Another perspective to contrast.",
            "Third angle bringing extra nuance.",
        ]
        return [
            ChatSample(index=index, text=text, tokens=120 + index, finish_reason="stop", usage={"completion_tokens": 120})
            for index, text in enumerate(base_texts)
        ]

    async def embed_texts(self, texts, **_: object) -> EmbeddingBatch:
        self.embed_calls += 1
        self.embed_payloads.append(list(texts))
        vectors = []
        for index, _ in enumerate(texts):
            vectors.append([0.1 * (index + 1), -0.2 * (index + 1), 0.05 * (index + 1)])
        return EmbeddingBatch(vectors=vectors, model="fake-embedding", dim=3)

    async def discourse_tag_segments(self, segments, **_: object):
        return [None for _ in segments]


@pytest.mark.asyncio
async def test_run_service_persists_sampling_artifacts(session):
    service = RunService(openai_service=FakeOpenAIService())
    payload = RunCreateRequest(prompt="Test me", n=3, model="gpt-4.1-mini", temperature=0.9, top_p=1.0, seed=42, max_tokens=256)
    run = await service.create_run(session, payload)
    await service.sample_run(session, run_id=run.id, sample_request=SampleRequest())

    details = await load_run_with_details(session, run.id)
    assert details is not None
    (
        run_loaded,
        responses,
        projections,
        clusters,
        embeddings,
        segments,
        edges,
        hulls,
        cluster_metrics,
        provenance,
    ) = details

    assert run_loaded.status == RunStatus.COMPLETED
    assert run_loaded.progress_stage == "completed"
    assert run_loaded.progress_message
    assert run_loaded.progress_percent == pytest.approx(1.0)
    assert run_loaded.umap_n_neighbors >= 5
    assert run_loaded.random_state_seed_source in {"default", "ui", "env"}
    assert run_loaded.trustworthiness_2d is not None
    assert len(responses) == 3
    assert len(embeddings) == 3
    assert len(segments) >= 3
    assert len({projection.dim for projection in projections}) == 2
    assert any(cluster.label >= -1 for cluster in clusters)
    assert all(embedding.dim == 3 for embedding in embeddings)
    assert provenance is not None
    assert cluster_metrics is not None
    assert cluster_metrics.algo in {"hdbscan", "kmeans"}
    assert provenance.umap_params_json is not None
    params = json.loads(provenance.umap_params_json)
    assert params["n_neighbors"] == run_loaded.umap_n_neighbors

    coords = np.array(
        [
            (projection.x, projection.y, projection.z or 0.0)
            for projection in projections
            if projection.dim == 3
        ]
    )
    assert coords.shape[0] == 3
    assert isinstance(edges, list)
    assert isinstance(hulls, list)

    await session.refresh(run_loaded)
    assert run_loaded.processing_time_ms is not None
    assert run_loaded.processing_time_ms >= 0
    timings_snapshot = json.loads(run_loaded.timings_json or "{}")
    assert isinstance(timings_snapshot, dict)
    stages = timings_snapshot.get("stages", [])
    assert isinstance(stages, list)
    assert any(stage.get("name") == "request-completions" for stage in stages)
    ann_records = await session.exec(select(AnnIndex).where(AnnIndex.run_id == run_loaded.id))
    ann_entry = ann_records.first()
    assert ann_entry is not None
    params = json.loads(ann_entry.params_json or "{}")
    assert params.get("method")
    input_space = json.loads(provenance.input_space_json or "{}")
    assert input_space.get("metric") == run_loaded.umap_metric


@pytest.mark.asyncio
async def test_update_run_notes(session):
    service = RunService(openai_service=FakeOpenAIService())
    payload = RunCreateRequest(prompt="Note test", n=1, model="gpt-4.1-mini", temperature=0.7, top_p=1.0)
    run = await service.create_run(session, payload)
    update = RunUpdateRequest(notes="  keep this insight  ")
    updated_run = await service.update_run(session, run_id=run.id, payload=update)
    assert updated_run.notes == "keep this insight"

    cleared = await service.update_run(session, run_id=run.id, payload=RunUpdateRequest(notes="   "))
    assert cleared.notes is None


@pytest.mark.asyncio
async def test_run_create_persists_preproc_version(session):
    service = RunService(openai_service=FakeOpenAIService())
    payload = RunCreateRequest(
        prompt="Hash me",
        n=1,
        model="gpt-4.1-mini",
        temperature=0.5,
        top_p=1.0,
        preproc_version="norm-nfkc-v1",
    )
    run = await service.create_run(session, payload)
    assert run.preproc_version == "norm-nfkc-v1"


@pytest.mark.asyncio
async def test_embedding_cache_hits_across_runs(session):
    openai = FakeOpenAIService()
    service = RunService(openai_service=openai)
    payload = RunCreateRequest(
        prompt="Cache test",
        n=2,
        model="gpt-4.1-mini",
        temperature=0.6,
        top_p=1.0,
        embedding_model="fake-embedding",
        use_cache=True,
    )
    first_run = await service.create_run(session, payload)
    await service.sample_run(session, run_id=first_run.id, sample_request=SampleRequest())
    details_one = await load_run_with_details(session, first_run.id)
    assert details_one is not None
    segments_first = details_one[5]
    assert segments_first
    assert any(not getattr(segment, "is_cached", False) for segment in segments_first)

    second_run = await service.create_run(session, payload)
    await service.sample_run(session, run_id=second_run.id, sample_request=SampleRequest())
    details_two = await load_run_with_details(session, second_run.id)
    assert details_two is not None
    segments_second = details_two[5]
    assert segments_second
    assert all(
        getattr(segment, "is_cached", False)
        for segment in segments_second
        if getattr(segment, "text_hash", None)
    )


class DuplicateOpenAIService(FakeOpenAIService):
    async def sample_chat(self, **_: object) -> list[ChatSample]:
        self.sample_calls += 1
        text = "Repeated insight across completions."
        return [
            ChatSample(index=index, text=text, tokens=60, finish_reason="stop", usage={"completion_tokens": 60})
            for index in range(3)
        ]


@pytest.mark.asyncio
async def test_duplicate_flagging_when_cache_disabled(session):
    service = RunService(openai_service=DuplicateOpenAIService())
    payload = RunCreateRequest(
        prompt="Duplicate scan",
        n=3,
        model="gpt-4.1-mini",
        temperature=0.8,
        top_p=1.0,
        embedding_model="fake-embedding",
        use_cache=False,
    )
    run = await service.create_run(session, payload)
    await service.sample_run(session, run_id=run.id, sample_request=SampleRequest())
    details = await load_run_with_details(session, run.id)
    assert details is not None
    segments = details[5]
    assert any(getattr(segment, "is_duplicate", False) for segment in segments)
    assert any(segment.text_hash for segment in segments)


def test_normalise_for_embedding_is_platform_stable():
    text_unix = "Line one\nLine two"
    text_windows = "Line one\r\nLine two"
    collapsed_a, normalised_a = normalise_for_embedding(text_unix)
    collapsed_b, normalised_b = normalise_for_embedding(text_windows)
    assert collapsed_a == collapsed_b
    assert normalised_a == normalised_b


@pytest.mark.asyncio
async def test_umap_validation_rejects_invalid_neighbors(session):
    service = RunService(openai_service=FakeOpenAIService())
    payload = RunCreateRequest(
        prompt="Invalid",
        n=10,
        model="gpt-4.1-mini",
        temperature=0.7,
        top_p=1.0,
        umap=UMAPParamsRequest(n_neighbors=1),
    )
    with pytest.raises(ValueError):
        await service.create_run(session, payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        UMAPParamsRequest(min_dist=-0.1),
        UMAPParamsRequest(min_dist=1.5),
        UMAPParamsRequest(metric="invalid"),
    ],
)
async def test_umap_validation_rejects_out_of_range_values(session, params):
    service = RunService(openai_service=FakeOpenAIService())
    payload = RunCreateRequest(
        prompt="Invalid",
        n=20,
        model="gpt-4.1-mini",
        temperature=0.7,
        top_p=1.0,
        umap=params,
    )
    with pytest.raises(ValueError):
        await service.create_run(session, payload)


@pytest.mark.asyncio
async def test_umap_seed_source_records_ui_override(session):
    service = RunService(openai_service=FakeOpenAIService())
    payload = RunCreateRequest(
        prompt="Seed overrides",
        n=12,
        model="gpt-4.1-mini",
        temperature=0.8,
        top_p=1.0,
        umap=UMAPParamsRequest(n_neighbors=25, min_dist=0.2, seed=123),
    )
    run = await service.create_run(session, payload)
    assert run.umap_seed == 123
    assert run.random_state_seed_source == "ui"


@pytest.mark.asyncio
async def test_umap_coordinates_deterministic_with_seed(session):
    service = RunService(openai_service=FakeOpenAIService())
    payload = RunCreateRequest(
        prompt="Deterministic",
        n=3,
        model="gpt-4.1-mini",
        temperature=0.7,
        top_p=1.0,
        use_cache=False,
        umap=UMAPParamsRequest(n_neighbors=15, min_dist=0.2, seed=42),
    )

    run_first = await service.create_run(session, payload)
    await service.sample_run(session, run_id=run_first.id, sample_request=SampleRequest(force_refresh=True))
    details_first = await load_run_with_details(session, run_first.id)
    projections_first = {
        proj.response_id: (round(proj.x, 6), round(proj.y, 6), round((proj.z or 0.0), 6))
        for proj in details_first[2]
        if proj.dim == 3
    }

    run_second = await service.create_run(session, payload)
    await service.sample_run(session, run_id=run_second.id, sample_request=SampleRequest(force_refresh=True))
    details_second = await load_run_with_details(session, run_second.id)
    projections_second = {
        proj.response_id: (round(proj.x, 6), round(proj.y, 6), round((proj.z or 0.0), 6))
        for proj in details_second[2]
        if proj.dim == 3
    }

    assert len(projections_first) == len(projections_second)
    first_values = sorted(projections_first.values())
    second_values = sorted(projections_second.values())
    assert first_values == second_values


@pytest.mark.asyncio
async def test_cached_neighbors_load_from_index(session):
    service = RunService(openai_service=FakeOpenAIService())
    payload = RunCreateRequest(
        prompt="Neighbor cache",
        n=3,
        model="gpt-4.1-mini",
        temperature=0.7,
        top_p=1.0,
        use_cache=False,
        umap=UMAPParamsRequest(n_neighbors=10, min_dist=0.2, seed=11),
    )
    run = await service.create_run(session, payload)
    await service.sample_run(session, run_id=run.id, sample_request=SampleRequest(force_refresh=True))
    ann_query = await session.exec(select(AnnIndex).where(AnnIndex.run_id == run.id))
    ann_entry = ann_query.scalar_one()

    cached_map = service._load_cached_neighbors(ann_entry, requested_k=5)
    assert cached_map
    any_lengths = {len(values) for values in cached_map.values()}
    assert any_lengths  # not empty set
    assert max(any_lengths) >= 1


@pytest.mark.asyncio
async def test_get_projection_variant_uses_cache(async_session, openai_stub):
    service = RunService(openai_service=openai_stub)
    payload = RunCreateRequest(prompt="Cache me", n=4, model="gpt-4.1-mini", temperature=0.8, top_p=0.9, seed=11, max_tokens=200)
    run = await service.create_run(async_session, payload)
    await service.sample_run(async_session, run_id=run.id, sample_request=SampleRequest(include_segments=False))

    first_payload, first_cache = await service.get_projection_variant(
        async_session, run_id=run.id, method="umap", params=None
    )
    assert first_cache is False
    second_payload, second_cache = await service.get_projection_variant(
        async_session, run_id=run.id, method="umap", params=None
    )
    assert second_cache is True
    assert first_payload.method == "umap"
    assert np.allclose(first_payload.coords_2d, second_payload.coords_2d)
    assert np.allclose(first_payload.coords_3d, second_payload.coords_3d)
