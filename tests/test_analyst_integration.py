"""Integration tests for the Data Analyst agent.

Uses real ChromaDB (on-disk, tmp_path) and the real Claude Haiku model
via the Anthropic API. Requires CHAMPOLLION_API_KEY (sk-ant-...) to be
set in the environment or in champollion_agents/.env.

Run with:
    pixi run test-integration
"""
from __future__ import annotations

import pytest


@pytest.mark.integration
async def test_analyst_indexes_and_answers(anthropic_api_key, fake_embeddings_csv, tmp_output_dir):
    """End-to-end: index fake CSV embeddings then query the analyst agent."""
    from champollion_agents.analyst.agent import build_analyst_agent
    from champollion_agents.analyst.indexer import index_pipeline_outputs

    output_dir = str(tmp_output_dir)
    combined_dir = str(fake_embeddings_csv)

    count = await index_pipeline_outputs(combined_dir, output_dir, run_id="integration-test-run")
    assert count == 3, f"Expected 3 docs indexed, got {count}"

    agent = build_analyst_agent(output_dir)

    config = {"configurable": {"thread_id": "integration-1"}}
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "List all indexed regions"}]},
        config=config,
    )

    final = result["messages"][-1].content
    assert "SC-sylv" in final, f"Expected 'SC-sylv' in response, got: {final[:300]}"


@pytest.mark.integration
async def test_analyst_subject_stats(anthropic_api_key, fake_embeddings_csv, tmp_output_dir):
    """Analyst agent can retrieve per-subject embedding statistics."""
    from champollion_agents.analyst.agent import build_analyst_agent
    from champollion_agents.analyst.indexer import index_pipeline_outputs

    output_dir = str(tmp_output_dir)
    await index_pipeline_outputs(str(fake_embeddings_csv), output_dir, run_id="stats-test-run")

    agent = build_analyst_agent(output_dir)
    config = {"configurable": {"thread_id": "integration-2"}}
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Get embedding stats for subject sub-001"}]},
        config=config,
    )

    final = result["messages"][-1].content
    assert "sub-001" in final or "SC-sylv" in final, f"Unexpected response: {final[:300]}"
