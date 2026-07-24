from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport

from champollion_agents._acp import RunStore, emit, make_acp_router
from fastapi import FastAPI


@pytest.mark.unit
async def test_analyst_server_health(monkeypatch):
    monkeypatch.setenv("CHAMPOLLION_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CHAMPOLLION_OUTPUT_DIR", "/tmp/test-analyst")

    with patch("champollion_agents.analyst.server.lifespan"):
        from champollion_agents.analyst.server import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200


@pytest.mark.unit
async def test_analyst_index_endpoint(monkeypatch, tmp_output_dir, fake_embeddings_csv):
    monkeypatch.setenv("CHAMPOLLION_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CHAMPOLLION_OUTPUT_DIR", str(tmp_output_dir))

    with (
        patch("champollion_agents.analyst.server.lifespan"),
        patch(
            "champollion_agents.analyst.server.index_pipeline_outputs",
            new=AsyncMock(return_value=3),
        ),
    ):
        from champollion_agents.analyst import server as srv
        srv._output_dir = str(tmp_output_dir)

        async with AsyncClient(transport=ASGITransport(app=srv.app), base_url="http://test") as client:
            resp = await client.post(
                "/index",
                params={"combined_embeddings_path": str(fake_embeddings_csv), "run_id": "run-001"},
            )
            assert resp.status_code == 200
            assert resp.json()["indexed"] == 3


@pytest.mark.unit
async def test_analyst_run_endpoint():
    fake_run_store: RunStore = {}

    async def fake_execute(run_id, message, queue, session_id):
        await emit(queue, "message", {"text": "analysis complete"})
        await emit(queue, "done", {"status": "succeeded"})
        fake_run_store[run_id].status = "succeeded"

    app = FastAPI()
    app.include_router(make_acp_router(fake_run_store, fake_execute))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/runs", json={"message": "find similar subjects"})
        assert resp.status_code == 202
        data = resp.json()
        assert "run_id" in data
