"""Integration tests for the champollion-sulcal MCP server.

These tests start the real MCP subprocess via `pixi run run` and exercise
the live tools. They require the champollion_sulcal_mcp project to be
present as a sibling directory and its pixi environment to be installed.

Run with:
    pixi run test-integration
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
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


@asynccontextmanager
async def _mcp_tools(mcp_dir: str, pipeline_dir: str):
    """Start the champollion MCP subprocess and yield its LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    env = {**os.environ, "CHAMPOLLION_PIPELINE_DIR": pipeline_dir}
    client = MultiServerMCPClient(
        {
            "champollion-sulcal": {
                "command": "pixi",
                "args": ["run", "run"],
                "cwd": mcp_dir,
                "env": env,
                "transport": "stdio",
            }
        }
    )
    yield await client.get_tools()


@pytest.mark.integration
async def test_mcp_tool_discovery(mcp_dir, pipeline_dir):
    """MCP server exposes >10 tools including preflight_check."""
    async with _mcp_tools(mcp_dir, pipeline_dir) as tools:
        names = [t.name for t in tools]
        assert len(tools) > 10, f"Expected >10 tools, got {len(tools)}: {names}"
        assert "preflight_check" in names
        assert "get_pipeline_info" in names


@pytest.mark.integration
async def test_mcp_get_pipeline_info(mcp_dir, pipeline_dir):
    """get_pipeline_info returns version and tool list."""
    async with _mcp_tools(mcp_dir, pipeline_dir) as tools:
        tool = next(t for t in tools if t.name == "get_pipeline_info")
        result = await tool.ainvoke({})
        result_str = str(result)
        assert "version" in result_str.lower() or "0.1" in result_str


@pytest.mark.integration
async def test_mcp_preflight_ok(mcp_dir, pipeline_dir):
    """preflight_check reports all required scripts are present."""
    async with _mcp_tools(mcp_dir, pipeline_dir) as tools:
        tool = next(t for t in tools if t.name == "preflight_check")
        result = await tool.ainvoke({})
        result_str = str(result).lower()
        assert "ok" in result_str or "true" in result_str or "pass" in result_str


@pytest.mark.integration
async def test_mcp_list_jobs_empty(mcp_dir, pipeline_dir, tmp_path):
    """list_jobs on a fresh temp directory returns no jobs."""
    async with _mcp_tools(mcp_dir, pipeline_dir) as tools:
        tool = next(t for t in tools if t.name == "list_jobs")
        result = await tool.ainvoke({"output_dir": str(tmp_path)})
        result_str = str(result).strip()
        assert result_str in ("[]", "") or "no jobs" in result_str.lower() or result_str == "[]"
