from uuid import UUID

import numpy as np
import pytest

from app.models import RunStatus
from app.schemas import RunCreateRequest, RunUpdateRequest, SampleRequest
from app.services.openai_client import ChatSample, EmbeddingBatch
from app.services.runs import RunService, load_run_with_details


class FakeOpenAIService:
    @property
    def is_configured(self) -> bool:
        return False

    async def sample_chat(self, **_: object) -> list[ChatSample]:
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
    run_loaded, responses, projections, clusters, embeddings, segments, edges, hulls = details

    assert run_loaded.status == RunStatus.COMPLETED
    assert run_loaded.progress_stage == "completed"
    assert run_loaded.progress_message
    assert run_loaded.progress_percent == pytest.approx(1.0)
    assert len(responses) == 3
    assert len(embeddings) == 3
    assert len(segments) >= 3
    assert len({projection.dim for projection in projections}) == 2
    assert any(cluster.label >= -1 for cluster in clusters)
    assert all(embedding.dim == 3 for embedding in embeddings)

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
