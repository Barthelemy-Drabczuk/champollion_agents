from __future__ import annotations

import pytest


@pytest.fixture
async def indexed_dir(tmp_output_dir, fake_embeddings_csv):
    from champollion_agents.analyst.indexer import index_pipeline_outputs

    await index_pipeline_outputs(str(fake_embeddings_csv), str(tmp_output_dir), "run-001")
    return tmp_output_dir


@pytest.mark.unit
def test_list_indexed_regions(indexed_dir):
    from champollion_agents.analyst.tools import make_analyst_tools

    tools = {t.name: t for t in make_analyst_tools(str(indexed_dir))}
    result = tools["list_indexed_regions"].invoke({"output_dir_": ""})
    assert "SC-sylv_left" in result or "SC" in result


@pytest.mark.unit
def test_list_indexed_regions_empty(tmp_output_dir):
    from champollion_agents.analyst.tools import make_analyst_tools

    tools = {t.name: t for t in make_analyst_tools(str(tmp_output_dir))}
    result = tools["list_indexed_regions"].invoke({"output_dir_": ""})
    assert "No embeddings indexed" in result or "empty" in result.lower()


@pytest.mark.unit
def test_get_subject_embedding_stats(indexed_dir):
    from champollion_agents.analyst.tools import make_analyst_tools

    tools = {t.name: t for t in make_analyst_tools(str(indexed_dir))}
    result = tools["get_subject_embedding_stats"].invoke({"subject_id": "sub-001"})
    assert "sub-001" in result
    assert "mean=" in result


@pytest.mark.unit
def test_search_similar_subjects(indexed_dir):
    from champollion_agents.analyst.tools import make_analyst_tools

    tools = {t.name: t for t in make_analyst_tools(str(indexed_dir))}
    result = tools["search_similar_subjects"].invoke(
        {"region": "SC-sylv", "hemisphere": "left", "subject_id": "sub-001", "n_results": 2}
    )
    assert "sub-001" in result or "similar" in result.lower()
