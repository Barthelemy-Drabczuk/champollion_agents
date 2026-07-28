"""Stdio MCP server exposing ChromaDB analyst tools.

Run with:
    python -m champollion_agents.analyst.mcp_server

The CHAMPOLLION_OUTPUT_DIR environment variable sets the ChromaDB root.
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from champollion_agents.analyst.indexer import REGION_EMBEDDINGS, get_chroma_client

_OUTPUT_DIR = os.environ.get("CHAMPOLLION_OUTPUT_DIR", "./outputs")

mcp = FastMCP("analyst-tools")


@mcp.tool()
def search_similar_subjects(region: str, hemisphere: str, subject_id: str, n_results: int = 5) -> str:
    """Find subjects with similar embeddings for a given region/hemisphere."""
    client = get_chroma_client(_OUTPUT_DIR)
    try:
        coll = client.get_collection(REGION_EMBEDDINGS, embedding_function=None)
    except Exception:
        return f"No embeddings indexed yet in {_OUTPUT_DIR}"

    doc_id = f"{subject_id}__{region}__{hemisphere}"
    existing = coll.get(ids=[doc_id], include=["embeddings"])
    if not existing["embeddings"]:
        return f"Subject {subject_id} not found for {region} {hemisphere}"

    results = coll.query(
        query_embeddings=[existing["embeddings"][0]],
        n_results=n_results + 1,
        where={"$and": [{"region": {"$eq": region}}, {"hemisphere": {"$eq": hemisphere}}]},
        include=["metadatas", "distances"],
    )
    lines = [f"Top {n_results} subjects similar to {subject_id} ({region} {hemisphere}):"]
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        if meta["subject_id"] != subject_id:
            lines.append(f"  {meta['subject_id']}  distance={dist:.4f}")
    return "\n".join(lines)


@mcp.tool()
def list_indexed_regions() -> str:
    """List all regions and hemispheres currently indexed in ChromaDB."""
    client = get_chroma_client(_OUTPUT_DIR)
    try:
        coll = client.get_collection(REGION_EMBEDDINGS, embedding_function=None)
    except Exception:
        return "No embeddings indexed yet."

    all_meta = coll.get(include=["metadatas"])["metadatas"]
    regions = sorted({f"{m['region']}_{m['hemisphere']}" for m in all_meta})
    return ("Indexed regions:\n" + "\n".join(f"  {r}" for r in regions)) if regions else "Collection is empty."


@mcp.tool()
def get_subject_embedding_stats(subject_id: str) -> str:
    """Return summary statistics for all embeddings of a given subject."""
    import numpy as np

    client = get_chroma_client(_OUTPUT_DIR)
    try:
        coll = client.get_collection(REGION_EMBEDDINGS, embedding_function=None)
    except Exception:
        return f"No embeddings indexed yet in {_OUTPUT_DIR}"

    results = coll.get(where={"subject_id": subject_id}, include=["embeddings", "metadatas"])
    if not results["embeddings"]:
        return f"No embeddings found for subject {subject_id}"

    lines = [f"Subject {subject_id} — {len(results['embeddings'])} region(s):"]
    for emb, meta in zip(results["embeddings"], results["metadatas"]):
        arr = np.array(emb)
        lines.append(
            f"  {meta['region']}_{meta['hemisphere']}  "
            f"mean={arr.mean():.4f}  std={arr.std():.4f}  dim={len(arr)}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
