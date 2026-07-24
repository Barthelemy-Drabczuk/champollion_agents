from __future__ import annotations

import pytest
from langchain_core.tools import tool as lc_tool


@pytest.mark.unit
def test_build_technician_agent_with_mock_tools(monkeypatch):
    monkeypatch.setenv("CHAMPOLLION_LLM_API_KEY", "test-key")

    from champollion_agents.technician.agent import build_technician_agent

    @lc_tool
    def fake_tool(x: str) -> str:
        """Fake MCP tool for testing."""
        return x

    agent = build_technician_agent(mcp_tools=[fake_tool], analyst_url="http://localhost:8002")

    assert hasattr(agent, "ainvoke")
    assert hasattr(agent, "astream")


@pytest.mark.unit
def test_build_technician_agent_includes_local_tools(monkeypatch):
    monkeypatch.setenv("CHAMPOLLION_LLM_API_KEY", "test-key")

    from champollion_agents.technician.agent import build_technician_agent

    agent = build_technician_agent(mcp_tools=[], analyst_url="http://localhost:8002")
    tool_node = agent.nodes.get("tools")
    assert tool_node is not None
