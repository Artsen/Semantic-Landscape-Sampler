import pytest

from app.schemas import RunCreateRequest, SampleRequest
from app.services.runs import RunService

from .test_runs_service import FakeOpenAIService


@pytest.mark.asyncio
async def test_metrics_endpoint_reports_cache_hits(client, session):
    service = RunService(openai_service=FakeOpenAIService())
    payload = RunCreateRequest(
        prompt="Metric coverage",
        n=3,
        model="gpt-4.1-mini",
        temperature=0.8,
        top_p=1.0,
        embedding_model="fake-embedding",
        use_cache=True,
    )
    run_first = await service.create_run(session, payload)
    await service.sample_run(session, run_id=run_first.id, sample_request=SampleRequest())

    run_second = await service.create_run(session, payload)
    await service.sample_run(session, run_id=run_second.id, sample_request=SampleRequest())

    response = await client.get(f"/run/{run_second.id}/metrics")
    assert response.status_code == 200
    payload_json = response.json()
    assert payload_json["run_id"] == str(run_second.id)
    assert payload_json["total_segments"] > 0
    assert payload_json["cached_segments"] == payload_json["total_segments"]
    assert pytest.approx(payload_json["cache_hit_rate"], rel=0.01) == 100.0
    assert "processing_time_ms" in payload_json
    assert payload_json["processing_time_ms"] >= 0
    assert isinstance(payload_json.get("stage_timings"), list)


@pytest.mark.asyncio
async def test_graph_endpoint_returns_edges(client, session):
    service = RunService(openai_service=FakeOpenAIService())
    payload = RunCreateRequest(
        prompt="Graph coverage",
        n=3,
        model="gpt-4.1-mini",
        temperature=0.7,
        top_p=1.0,
        embedding_model="fake-embedding",
        use_cache=False,
    )
    run = await service.create_run(session, payload)
    await service.sample_run(session, run_id=run.id, sample_request=SampleRequest())

    full = await client.get(f"/run/{run.id}/graph?mode=full&k=12&sim=0.0")
    assert full.status_code == 200
    full_payload = full.json()
    assert full_payload["mode"] == "full"
    assert full_payload["node_count"] > 0
    assert isinstance(full_payload["edges"], list)
    if full_payload["node_count"] > 1:
        assert full_payload["edges"], "expected graph to return edges"
        for edge in full_payload["edges"]:
            assert edge["similarity"] >= 0.0

    simplified = await client.get(f"/run/{run.id}/graph?mode=simplified&k=8&sim=0.0")
    assert simplified.status_code == 200
    simplified_payload = simplified.json()
    assert simplified_payload["mode"] == "simplified"
    assert simplified_payload["node_count"] == full_payload["node_count"]
    assert isinstance(simplified_payload["auto_simplified"], bool)


@pytest.mark.asyncio
async def test_provenance_endpoint_returns_payload(client, session):
    service = RunService(openai_service=FakeOpenAIService())
    payload = RunCreateRequest(
        prompt="Provenance coverage",
        n=2,
        model="gpt-4.1-mini",
        temperature=0.8,
        top_p=1.0,
        embedding_model="fake-embedding",
        use_cache=True,
    )
    run = await service.create_run(session, payload)
    await service.sample_run(session, run_id=run.id, sample_request=SampleRequest())

    response = await client.get(f"/run/{run.id}/provenance")
    assert response.status_code == 200
    payload_json = response.json()
    assert payload_json["run_id"] == str(run.id)
    assert payload_json["embedding_model"] == "fake-embedding"
    assert isinstance(payload_json.get("lib_versions"), dict)
    assert isinstance(payload_json.get("feature_weights"), dict)
    assert isinstance(payload_json.get("umap_params"), dict)
    assert payload_json["umap_params"].get("n_neighbors") == 30


@pytest.mark.asyncio
async def test_cluster_metrics_endpoint_returns_summary(client, session):
    service = RunService(openai_service=FakeOpenAIService())
    payload = RunCreateRequest(
        prompt="Metrics detail",
        n=3,
        model="gpt-4.1-mini",
        temperature=0.7,
        top_p=1.0,
        embedding_model="fake-embedding",
        use_cache=False,
    )
    run = await service.create_run(session, payload)
    await service.sample_run(session, run_id=run.id, sample_request=SampleRequest())

    response = await client.get(f"/run/{run.id}/cluster-metrics")
    assert response.status_code == 200
    payload_json = response.json()
    assert payload_json["run_id"] == str(run.id)
    assert payload_json["algo"] in {"hdbscan", "kmeans"}
    assert isinstance(payload_json.get("params"), dict)


@pytest.mark.asyncio
async def test_recompute_clusters_endpoint_updates_params(client, session):
    service = RunService(openai_service=FakeOpenAIService())
    payload = RunCreateRequest(
        prompt="Recluster detail",
        n=3,
        model="gpt-4.1-mini",
        temperature=0.6,
        top_p=1.0,
        embedding_model="fake-embedding",
        use_cache=False,
    )
    run = await service.create_run(session, payload)
    await service.sample_run(session, run_id=run.id, sample_request=SampleRequest())

    response = await client.get(
        f"/run/{run.id}/clusters?min_cluster_size=2&min_samples=1"
    )
    assert response.status_code == 200
    payload_json = response.json()
    assert payload_json["run"].get("id") == str(run.id)
    metrics = payload_json.get("cluster_metrics")
    assert metrics is not None
    assert metrics.get("params", {}).get("min_cluster_size") == 2

