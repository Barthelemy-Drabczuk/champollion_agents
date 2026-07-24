from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.mark.unit
async def test_technician_server_health(monkeypatch):
    monkeypatch.setenv("CHAMPOLLION_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CHAMPOLLION_MCP_DIR", "/fake/mcp")
    monkeypatch.setenv("CHAMPOLLION_PIPELINE_DIR", "/fake/pipeline")

    with patch("champollion_agents.technician.server.lifespan"):
        from champollion_agents.technician.server import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200


@pytest.mark.unit
async def test_create_run_returns_run_id(monkeypatch):
    monkeypatch.setenv("CHAMPOLLION_LLM_API_KEY", "test-key")

    from champollion_agents._acp import RunStore, RunEntry, emit, make_acp_router
    from fastapi import FastAPI

    fake_run_store: RunStore = {}

    async def fake_execute(run_id, message, queue, session_id):
        await emit(queue, "message", {"text": "done"})
        await emit(queue, "done", {"status": "succeeded"})
        fake_run_store[run_id].status = "succeeded"

    app = FastAPI()
    app.include_router(make_acp_router(fake_run_store, fake_execute))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/runs", json={"message": "run the pipeline"})
        assert resp.status_code == 202
        assert "run_id" in resp.json()
