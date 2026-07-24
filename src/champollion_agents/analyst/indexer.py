from __future__ import annotations

import re
from pathlib import Path

import chromadb
import pandas as pd

REGION_EMBEDDINGS = "region_embeddings"
SUBJECT_METADATA = "subject_metadata"
PIPELINE_RUNS = "pipeline_runs"

_REGION_RE = re.compile(r"(.+)_(left|right)$", re.IGNORECASE)


def get_chroma_client(output_dir: str) -> chromadb.ClientAPI:
    chroma_path = Path(output_dir) / ".chromadb"
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_path))


def _parse_region_hemisphere(folder_name: str) -> tuple[str, str]:
    m = _REGION_RE.search(folder_name)
    if m:
        return m.group(1), m.group(2).lower()
    return folder_name, "unknown"


async def index_pipeline_outputs(
    combined_embeddings_path: str,
    output_dir: str,
    run_id: str,
) -> int:
    client = get_chroma_client(output_dir)
    collection = client.get_or_create_collection(REGION_EMBEDDINGS, embedding_function=None)

    total = 0
    for csv_path in Path(combined_embeddings_path).rglob("full_embeddings.csv"):
        region, hemisphere = _parse_region_hemisphere(csv_path.parent.name)
        df = pd.read_csv(csv_path, index_col=0)
        dim_cols = [c for c in df.columns if c.startswith("dim")]
        if not dim_cols:
            continue

        ids = [f"{subj}__{region}__{hemisphere}" for subj in df.index]
        embeddings = df[dim_cols].values.tolist()
        metadatas = [
            {"subject_id": subj, "region": region, "hemisphere": hemisphere, "run_id": run_id}
            for subj in df.index
        ]
        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        total += len(ids)

    runs_coll = client.get_or_create_collection(PIPELINE_RUNS, embedding_function=None)
    runs_coll.upsert(
        ids=[run_id],
        embeddings=[[0.0]],
        metadatas=[{"run_id": run_id, "combined_embeddings_path": combined_embeddings_path}],
    )
    return total
