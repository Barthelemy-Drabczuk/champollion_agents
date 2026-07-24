from __future__ import annotations

from langchain_core.tools import tool

from champollion_agents.analyst.indexer import REGION_EMBEDDINGS, get_chroma_client


def make_analyst_tools(output_dir: str):

    @tool
    def search_similar_subjects(region: str, hemisphere: str, subject_id: str, n_results: int = 5) -> str:
        """Find subjects with similar embeddings for a given region/hemisphere."""
        client = get_chroma_client(output_dir)
        try:
            coll = client.get_collection(REGION_EMBEDDINGS, embedding_function=None)
        except Exception:
            return f"No embeddings indexed yet in {output_dir}"

        doc_id = f"{subject_id}__{region}__{hemisphere}"
        existing = coll.get(ids=[doc_id], include=["embeddings"])
        if existing["embeddings"] is None or len(existing["embeddings"]) == 0:
            return f"Subject {subject_id} not found for {region} {hemisphere}"

        query_embedding = existing["embeddings"][0]
        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=n_results + 1,
            where={"$and": [{"region": {"$eq": region}}, {"hemisphere": {"$eq": hemisphere}}]},
            include=["metadatas", "distances"],
        )
        lines = [f"Top {n_results} subjects similar to {subject_id} ({region} {hemisphere}):"]
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            sid = meta["subject_id"]
            if sid != subject_id:
                lines.append(f"  {sid}  distance={dist:.4f}")
        return "\n".join(lines)

    @tool
    def list_indexed_regions(output_dir_: str = "") -> str:
        """List all regions and hemispheres currently indexed in ChromaDB."""
        dir_to_use = output_dir_ or output_dir
        client = get_chroma_client(dir_to_use)
        try:
            coll = client.get_collection(REGION_EMBEDDINGS, embedding_function=None)
        except Exception:
            return "No embeddings indexed yet."

        all_meta = coll.get(include=["metadatas"])["metadatas"]
        regions = sorted({f"{m['region']}_{m['hemisphere']}" for m in all_meta})
        if not regions:
            return "Collection is empty."
        return "Indexed regions:\n" + "\n".join(f"  {r}" for r in regions)

    @tool
    def get_subject_embedding_stats(subject_id: str) -> str:
        """Return summary statistics for all embeddings of a given subject."""
        import numpy as np

        client = get_chroma_client(output_dir)
        try:
            coll = client.get_collection(REGION_EMBEDDINGS, embedding_function=None)
        except Exception:
            return f"No embeddings indexed yet in {output_dir}"

        results = coll.get(
            where={"subject_id": subject_id},
            include=["embeddings", "metadatas"],
        )
        if results["embeddings"] is None or len(results["embeddings"]) == 0:
            return f"No embeddings found for subject {subject_id}"

        lines = [f"Subject {subject_id} — {len(results['embeddings'])} region(s):"]
        for emb, meta in zip(results["embeddings"], results["metadatas"]):
            arr = np.array(emb)
            lines.append(
                f"  {meta['region']}_{meta['hemisphere']}  "
                f"mean={arr.mean():.4f}  std={arr.std():.4f}  dim={len(arr)}"
            )
        return "\n".join(lines)

    return [search_similar_subjects, list_indexed_regions, get_subject_embedding_stats]
