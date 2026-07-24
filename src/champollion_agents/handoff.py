from __future__ import annotations

import httpx
from pydantic import BaseModel


class HandoffPayload(BaseModel):
    event: str = "pipeline_complete"
    output_dir: str
    combined_embeddings: str
    run_id: str
    dataset: str
    n_subjects: int


async def post_handoff(payload: HandoffPayload, analyst_url: str = "http://localhost:8002") -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{analyst_url}/runs",
            json={"message": payload.model_dump()},
        )
        resp.raise_for_status()
        return resp.json()
