"""Integration tests for the fine-grained export endpoint."""

from __future__ import annotations

import hashlib
import io
import json
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from app.api.routes import run as run_routes
from app.models import (
    Cluster,
    Embedding,
    Projection,
    Response,
    ResponseSegment,
    Run,
    RunProvenance,
    RunStatus,
    SegmentInsight,
)


async def seed_run(session) -> tuple[Run, list[Response], list[ResponseSegment]]:
    run = Run(
        prompt="How will climate change reshape coastal cities?",
        n=2,
        model="gpt-4.1-mini",
        temperature=0.8,
        top_p=0.9,
        status=RunStatus.COMPLETED,
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    response_a = Response(
        run_id=run.id,
        index=0,
        raw_text="Coastal infrastructure must adapt with surge barriers and floating districts.",
        tokens=128,
        usage_json={"prompt_tokens": 64, "completion_tokens": 64},
    )
    response_b = Response(
        run_id=run.id,
        index=1,
        raw_text="Soft engineering and wetland restoration will buffer storm surges.",
        tokens=96,
        usage_json={"prompt_tokens": 48, "completion_tokens": 48},
    )
    session.add_all([response_a, response_b])
    await session.commit()
    await session.refresh(response_a)
    await session.refresh(response_b)

    projections = [
        Projection(response_id=response_a.id, method="umap", dim=3, x=0.1, y=0.2, z=0.3),
        Projection(response_id=response_a.id, method="umap", dim=2, x=0.15, y=0.25, z=None),
        Projection(response_id=response_b.id, method="umap", dim=3, x=-0.2, y=0.05, z=-0.1),
        Projection(response_id=response_b.id, method="umap", dim=2, x=-0.18, y=0.07, z=None),
    ]
    clusters = [
        Cluster(response_id=response_a.id, method="hdbscan", label=1, probability=0.92, similarity=0.81),
        Cluster(response_id=response_b.id, method="hdbscan", label=-1, probability=0.35, similarity=None),
    ]
    session.add_all(projections + clusters)

    embedding_a = Embedding(
        response_id=response_a.id,
        dim=3,
        vector=np.asarray([0.1, 0.2, 0.3], dtype=np.float32).tobytes(),
    )
    embedding_b = Embedding(
        response_id=response_b.id,
        dim=3,
        vector=np.asarray([0.4, 0.1, -0.2], dtype=np.float32).tobytes(),
    )
    session.add_all([embedding_a, embedding_b])

    seg_a = ResponseSegment(
        id=uuid4(),
        response_id=response_a.id,
        position=0,
        text="Harbor walls must flex with tidal rhythms.",
        role="assistant",
        tokens=48,
        prompt_similarity=0.74,
        silhouette_score=0.58,
        cluster_label=5,
        cluster_probability=0.9,
        cluster_similarity=0.86,
        coord_x=0.11,
        coord_y=0.22,
        coord_z=0.02,
        coord2_x=0.1,
        coord2_y=0.21,
        text_hash=hashlib.sha256(b"Harbor walls must flex with tidal rhythms.").hexdigest(),
        embedding_dim=4,
        embedding_vector=np.asarray([0.02, 0.04, 0.08, 0.16], dtype=np.float32).tobytes(),
    )
    seg_b = ResponseSegment(
        id=uuid4(),
        response_id=response_b.id,
        position=1,
        text="Mangrove buffers absorb storm energy.",
        role="assistant",
        tokens=32,
        prompt_similarity=0.61,
        silhouette_score=0.49,
        cluster_label=2,
        cluster_probability=0.65,
        cluster_similarity=0.52,
        coord_x=-0.19,
        coord_y=0.06,
        coord_z=-0.03,
        coord2_x=-0.17,
        coord2_y=0.05,
        text_hash=hashlib.sha256(b"Mangrove buffers absorb storm energy.").hexdigest(),
        embedding_dim=4,
        embedding_vector=np.asarray([0.05, -0.01, 0.02, -0.04], dtype=np.float32).tobytes(),
    )
    session.add_all([seg_a, seg_b])

    insight_a = SegmentInsight(
        segment_id=seg_a.id,
        top_terms_json=json.dumps([{"term": "harbor", "weight": 0.44}, {"term": "walls", "weight": 0.31}]),
        neighbors_json=json.dumps([{"segment_id": str(seg_b.id), "similarity": 0.48}]),
        metrics_json=json.dumps({"local_density": 0.72}),
    )
    insight_b = SegmentInsight(
        segment_id=seg_b.id,
        top_terms_json=json.dumps([{"term": "mangrove", "weight": 0.55}]),
        neighbors_json=json.dumps([]),
        metrics_json=json.dumps({"local_density": 0.41}),
    )
    session.add_all([insight_a, insight_b])

    provenance = RunProvenance(
        run_id=run.id,
        python_version="3.11.4",
        lib_versions_json=json.dumps({"numpy": "1.26.4"}),
        feature_weights_json=json.dumps({"tfidf": 0.8, "prompt_similarity": 0.2}),
        input_space_json=json.dumps({"blend": "openai+tfidf"}),
    )
    session.add(provenance)

    await session.commit()

    return run, [response_a, response_b], [seg_a, seg_b]


@pytest.mark.asyncio
async def test_export_selection_matches_segment_ids(client, session):
    run, _, segments = await seed_run(session)
    selected = segments[0]

    response = await client.post(
        f"/run/{run.id}/export?scope=selection&mode=segments&format=json",
        json={"selection_ids": [str(selected.id)]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == run_routes._EXPORT_SCHEMA_VERSION
    assert payload["count"] == 1
    row = payload["rows"][0]
    assert row["segment_id"] == str(selected.id)
    assert row["run_id"] == str(run.id)
    assert row["kind"] == "segment"


@pytest.mark.asyncio
async def test_export_cluster_csv(client, session):
    run, responses, _ = await seed_run(session)

    response = await client.get(
        f"/run/{run.id}/export?scope=cluster&mode=responses&cluster_id=1&format=csv"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    content = response.content.decode("utf-8").splitlines()
    assert "schema_version" in content[0]
    # Ensure the clustered response is present and the noise sample is excluded
    assert str(responses[0].id) in "".join(content[1:])
    assert str(responses[1].id) not in "".join(content[1:])


@pytest.mark.asyncio
async def test_export_parquet_includes_vectors_and_provenance(client, session):
    run, _, segments = await seed_run(session)

    response = await client.get(
        f"/run/{run.id}/export?scope=run&mode=segments&format=parquet&include=vectors,provenance"
    )

    assert response.status_code == 200
    dataframe = pd.read_parquet(io.BytesIO(response.content))
    assert dataframe.shape[0] == len(segments)
    assert "embedding" in dataframe.columns
    assert "provenance" in dataframe.columns
    sample_embedding = json.loads(dataframe.loc[0, "embedding"])
    assert "vector" in sample_embedding
    assert len(sample_embedding["vector"]) >= 4
    provenance_payload = json.loads(dataframe.loc[0, "provenance"])
    assert provenance_payload.get("python_version") == "3.11.4"
