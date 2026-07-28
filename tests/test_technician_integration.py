"""Integration tests for the Pipeline Technician agent.

Uses the claude-code-sdk (OAuth auth) + real MCP subprocess. Does NOT launch
any actual pipeline stage — only exercises read-only MCP tools (preflight,
get_pipeline_info) via the agent's reasoning loop.

Run with:
    pixi run test-integration
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

MCP_DIR = os.environ.get(
    "CHAMPOLLION_MCP_DIR",
    str(Path(__file__).parent.parent.parent / "champollion_sulcal_mcp"),
)
PIPELINE_DIR = os.environ.get(
    "CHAMPOLLION_PIPELINE_DIR",
    str(Path(__file__).parent.parent.parent / "champollion_pipeline"),
)


@pytest.fixture(scope="module")
def mcp_dir():
    if not Path(MCP_DIR).is_dir():
        pytest.skip(f"MCP server not found at {MCP_DIR}")
    return MCP_DIR


@pytest.fixture(scope="module")
def pipeline_dir():
    if not Path(PIPELINE_DIR).is_dir():
        pytest.skip(f"Pipeline dir not found at {PIPELINE_DIR}")
    return PIPELINE_DIR


@pytest.mark.integration
async def test_technician_runs_preflight(sdk_available, mcp_dir, pipeline_dir):
    """Technician agent calls preflight_check and returns a coherent result."""
    from champollion_agents.technician.agent import build_technician_agent

    agent = build_technician_agent(mcp_dir, pipeline_dir, analyst_url="http://localhost:8002")
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Run a preflight check on the pipeline."}]},
    )

    messages = result["messages"]
    assert len(messages) >= 2
    final = messages[-1]["content"]
    assert len(final) > 20, f"Response too short: {final!r}"
    lower = final.lower()
    assert any(word in lower for word in ("preflight", "check", "ok", "pass", "script", "pipeline")), (
        f"Response doesn't mention preflight: {final[:300]}"
    )


@pytest.mark.integration
async def test_technician_reports_pipeline_info(sdk_available, mcp_dir, pipeline_dir):
    """Technician agent can report pipeline version information."""
    from champollion_agents.technician.agent import build_technician_agent

    agent = build_technician_agent(mcp_dir, pipeline_dir, analyst_url="http://localhost:8002")
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What version is the Champollion pipeline?"}]},
    )

    final = result["messages"][-1]["content"]
    assert len(final) > 10, f"Response too short: {final!r}"
