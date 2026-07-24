from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path


@pytest.mark.unit
def test_get_chroma_client_creates_persistent_dir(tmp_output_dir):
    from champollion_agents.analyst.indexer import get_chroma_client
    client = get_chroma_client(str(tmp_output_dir))
    import chromadb
    assert isinstance(client, chromadb.ClientAPI)
    assert (tmp_output_dir / ".chromadb").exists()


@pytest.mark.unit
async def test_index_pipeline_outputs_populates_collection(tmp_output_dir, fake_embeddings_csv):
    from champollion_agents.analyst.indexer import index_pipeline_outputs, get_chroma_client, REGION_EMBEDDINGS

    n = await index_pipeline_outputs(
        combined_embeddings_path=str(fake_embeddings_csv),
        output_dir=str(tmp_output_dir),
        run_id="run-001",
    )
    assert n == 3

    client = get_chroma_client(str(tmp_output_dir))
    coll = client.get_collection(REGION_EMBEDDINGS)
    results = coll.get()
    assert len(results["ids"]) == 3
    assert any("SC-sylv" in id_ for id_ in results["ids"])


@pytest.mark.unit
async def test_index_is_idempotent(tmp_output_dir, fake_embeddings_csv):
    from champollion_agents.analyst.indexer import index_pipeline_outputs, get_chroma_client, REGION_EMBEDDINGS

    await index_pipeline_outputs(str(fake_embeddings_csv), str(tmp_output_dir), "run-001")
    await index_pipeline_outputs(str(fake_embeddings_csv), str(tmp_output_dir), "run-001")

    client = get_chroma_client(str(tmp_output_dir))
    coll = client.get_collection(REGION_EMBEDDINGS)
    assert len(coll.get()["ids"]) == 3
