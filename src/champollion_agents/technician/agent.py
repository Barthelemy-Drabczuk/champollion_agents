from __future__ import annotations

import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from champollion_agents.llm import build_llm
from champollion_agents.technician.tools import make_handoff_tool, poll_until_done, read_job_log_tail

SYSTEM_PROMPT = """You are the Champollion Pipeline Technician. Your job is to:
1. Run preflight_check first on every request.
2. Adapt parameters: if no CUDA, inject cpu=True; if njobs unset and >8 cores, set njobs=cpu_count//2.
3. Launch pipeline stages sequentially via the MCP tools, calling poll_until_done after each launch.
4. On combine success, call handoff_to_analyst with the output paths.
5. On any failure, read the job log and report the root cause.

Never guess paths — ask the user for any missing required path.
Always return job_id and output path immediately after launching each stage."""


def build_technician_agent(mcp_tools: list, analyst_url: str = "http://localhost:8002"):
    @tool
    async def poll_job(job_id: str, output_dir: str) -> str:
        """Poll a job until it reaches a terminal state (succeeded/failed/cancelled). Returns final status dict."""
        result = await poll_until_done(job_id, output_dir, interval=10)
        return str(result)

    @tool
    def tail_job_log(job_id: str, output_dir: str, n: int = 50) -> str:
        """Return the last N lines of a job's log file."""
        return read_job_log_tail(job_id, output_dir, n=n)

    handoff_tool = make_handoff_tool(analyst_url=analyst_url)
    local_tools = [poll_job, tail_job_log, handoff_tool]

    llm = build_llm()
    checkpointer = MemorySaver()
    return create_react_agent(
        llm,
        mcp_tools + local_tools,
        checkpointer=checkpointer,
        prompt=SYSTEM_PROMPT,
    )


@asynccontextmanager
async def load_mcp_tools(
    mcp_dir: str,
    pipeline_dir: str,
    extra_env: dict | None = None,
) -> AsyncGenerator[list, None]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    env = {**os.environ, "CHAMPOLLION_PIPELINE_DIR": pipeline_dir}
    if extra_env:
        env.update(extra_env)

    client = MultiServerMCPClient(
        {
            "champollion-sulcal": {
                "command": "pixi",
                "args": ["run", "run"],
                "cwd": mcp_dir,
                "env": env,
                "transport": "stdio",
            }
        }
    )
    yield await client.get_tools()
