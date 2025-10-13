"""Tests for the compare runs service and endpoint."""

from __future__ import annotations

from typing import Sequence
from uuid import UUID

import numpy as np
import pytest

from app.models import Response, ResponseSegment, Run
from app.schemas import CompareRunsRequest
from app.services.compare import CompareService


async def _create_run(
    session,
    *,
    prompt: str,
    coords: Sequence[np.ndarray],
    hashes: Sequence[str | None],
    cluster_labels: Sequence[int],
    run_kwargs: dict | None = None,
) -> tuple[Run, list[UUID]]:
    run_kwargs = run_kwargs or {}
    run = Run(
        prompt=prompt,
        n=1,
        model="gpt-test",
        temperature=0.1,
        **run_kwargs,
    )
    session.add(run)
    await session.flush()

    response = Response(run_id=run.id, index=0, raw_text=f"response for {prompt}")
    session.add(response)
    await session.flush()

    segment_ids: list[UUID] = []
    for idx, coord in enumerate(coords):
        coord2 = coord[:2] if coord.shape[0] >= 2 else np.array([coord[0], 0.0])
        embedding = np.array([1.0, float(idx + 1), 0.0], dtype=np.float32)
        segment = ResponseSegment(
            response_id=response.id,
            position=idx,
            text=f"segment {idx} for {prompt}",
            role="assistant",
            text_hash=hashes[idx],
            cluster_label=cluster_labels[idx],
            cluster_probability=0.9,
            cluster_similarity=0.8,
            prompt_similarity=0.7,
            coord_x=float(coord[0]),
            coord_y=float(coord[1]),
            coord_z=float(coord[2]) if coord.shape[0] >= 3 else 0.0,
            coord2_x=float(coord2[0]),
            coord2_y=float(coord2[1]),
            embedding_dim=int(embedding.shape[0]),
            embedding_vector=embedding.astype(np.float16).tobytes(),
        )
        session.add(segment)
        await session.flush()
        segment_ids.append(segment.id)
    return run, segment_ids


async def _setup_shared_runs(session):
    left_coords = [
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    ]
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([0.0, 1.0, 0.0])
    right_coords = [rotation @ coord + translation for coord in left_coords]
    hashes = ["hash-a", "hash-b"]

    left_run, _ = await _create_run(
        session,
        prompt="left",
        coords=left_coords,
        hashes=hashes,
        cluster_labels=[0, 1],
    )
    right_run, _ = await _create_run(
        session,
        prompt="right",
        coords=right_coords,
        hashes=hashes,
        cluster_labels=[0, 1],
    )
    await session.commit()
    return left_run, right_run


async def _setup_nn_runs(session):
    left_coords = [
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.8, 0.2, 0.0], dtype=np.float64),
        np.array([1.6, 0.4, 0.0], dtype=np.float64),
    ]
    right_coords = [coord + np.array([0.05, -0.05, 0.0]) for coord in left_coords]
    unique_hashes = [None, None, None]

    left_run, _ = await _create_run(
        session,
        prompt="left-nn",
        coords=left_coords,
        hashes=unique_hashes,
        cluster_labels=[0, 1, 2],
    )
    right_run, _ = await _create_run(
        session,
        prompt="right-nn",
        coords=right_coords,
        hashes=unique_hashes,
        cluster_labels=[0, 1, 2],
    )
    await session.commit()
    return left_run, right_run


@pytest.mark.anyio
async def test_compare_runs_with_shared_hash_alignment(session):
    left_run, right_run = await _setup_shared_runs(session)
    service = CompareService(session)
    payload = CompareRunsRequest(
        left_run_id=left_run.id,
        right_run_id=right_run.id,
        min_shared=1,
        save=True,
    )
    result = await service.compare_runs(payload)

    assert result.transforms.anchor_kind == "shared_hash"
    assert result.metrics.shared_segment_count == 2
    assert result.run_pair_id is not None

    exact_links = [link for link in result.links if link.link_type == "exact_hash"]
    assert len(exact_links) == 2
    for link in exact_links:
        assert link.movement_distance is not None
        assert link.movement_distance == pytest.approx(0.0, abs=1e-5)


@pytest.mark.anyio
async def test_compare_runs_fallback_to_feature_nn(session):
    left_run, right_run = await _setup_nn_runs(session)
    service = CompareService(session)
    payload = CompareRunsRequest(
        left_run_id=left_run.id,
        right_run_id=right_run.id,
        min_shared=5,
        save=False,
    )
    result = await service.compare_runs(payload)

    assert result.transforms.anchor_kind == "feature_nn"
    assert result.metrics.shared_segment_count == 0
    assert any(link.link_type == "nn" for link in result.links)


@pytest.mark.anyio
async def test_compare_endpoint_returns_payload(session, client):
    left_run, right_run = await _setup_shared_runs(session)
    payload = {
        "left_run_id": str(left_run.id),
        "right_run_id": str(right_run.id),
        "min_shared": 1,
        "save": True,
    }
    response = await client.post("/compare", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]["shared_segment_count"] == 2
    assert body["transforms"]["anchor_kind"] == "shared_hash"
