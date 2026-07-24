from __future__ import annotations

import pytest


@pytest.fixture
def tmp_output_dir(tmp_path):
    d = tmp_path / "derivatives"
    d.mkdir()
    return d


@pytest.fixture
def fake_embeddings_csv(tmp_output_dir):
    """Create a minimal full_embeddings.csv for indexer tests."""
    import pandas as pd

    region_dir = tmp_output_dir / "combined_embeddings" / "SC-sylv_left"
    region_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {f"dim{i}": [float(i + j) for j in range(3)] for i in range(128)},
        index=["sub-001", "sub-002", "sub-003"],
    )
    df.index.name = "subject_id"
    csv_path = region_dir / "full_embeddings.csv"
    df.to_csv(csv_path)
    return tmp_output_dir / "combined_embeddings"
