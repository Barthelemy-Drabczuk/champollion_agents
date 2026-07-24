from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable

from langchain_core.tools import tool

from champollion_agents.handoff import HandoffPayload, post_handoff


async def poll_until_done(job_id: str, output_dir: str, interval: float = 10) -> dict:
    job_file = Path(output_dir) / ".mcp_jobs" / f"{job_id}.json"
    while True:
        if job_file.exists():
            try:
                data = json.loads(job_file.read_text())
                if data.get("status") in ("succeeded", "failed", "cancelled"):
                    return data
            except json.JSONDecodeError:
                pass
        await asyncio.sleep(interval)


def read_job_log_tail(job_id: str, output_dir: str, n: int = 50) -> str:
    log_file = Path(output_dir) / ".mcp_jobs" / f"{job_id}.log"
    if not log_file.exists():
        return ""
    lines = log_file.read_text().splitlines()
    return "\n".join(lines[-n:])


def make_handoff_tool(analyst_url: str = "http://localhost:8002") -> Callable:
    @tool
    async def handoff_to_analyst(
        output_dir: str,
        combined_embeddings: str,
        run_id: str,
        dataset: str,
        n_subjects: int,
    ) -> dict:
        """Hand off completed pipeline outputs to the Data Analyst agent for indexing and analysis."""
        payload = HandoffPayload(
            output_dir=output_dir,
            combined_embeddings=combined_embeddings,
            run_id=run_id,
            dataset=dataset,
            n_subjects=n_subjects,
        )
        return await post_handoff(payload, analyst_url=analyst_url)

    return handoff_to_analyst
