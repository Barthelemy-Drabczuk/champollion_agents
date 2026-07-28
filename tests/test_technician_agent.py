from __future__ import annotations

import pytest


@pytest.mark.unit
def test_build_technician_agent_returns_sdk_agent():
    from champollion_agents.sdk_agent import SdkAgent
    from champollion_agents.technician.agent import build_technician_agent

    agent = build_technician_agent(
        mcp_dir="/tmp/fake_mcp",
        pipeline_dir="/tmp/fake_pipeline",
        analyst_url="http://localhost:8002",
    )

    assert isinstance(agent, SdkAgent)
    assert hasattr(agent, "ainvoke")
    assert hasattr(agent, "astream")


@pytest.mark.unit
def test_build_technician_agent_has_champollion_sulcal_server():
    from champollion_agents.technician.agent import build_technician_agent

    agent = build_technician_agent(
        mcp_dir="/tmp/fake_mcp",
        pipeline_dir="/tmp/fake_pipeline",
    )

    servers = agent.options.mcp_servers
    assert servers is not None
    assert "champollion_sulcal" in servers
