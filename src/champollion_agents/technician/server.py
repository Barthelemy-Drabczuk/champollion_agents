from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from champollion_agents._acp import RunStore, emit, make_acp_router
from champollion_agents.technician.agent import build_technician_agent

run_store: RunStore = {}
_agent = None  # set during lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    mcp_dir = os.environ.get("CHAMPOLLION_MCP_DIR", "../champollion_sulcal_mcp")
    pipeline_dir = os.environ.get("CHAMPOLLION_PIPELINE_DIR", "../champollion_pipeline")
    analyst_url = os.environ.get("CHAMPOLLION_ANALYST_URL", "http://localhost:8002")
    _agent = build_technician_agent(mcp_dir, pipeline_dir, analyst_url=analyst_url)
    yield


async def execute_run(run_id: str, message: str | dict, queue: asyncio.Queue, session_id: str) -> None:
    try:
        user_text = message if isinstance(message, str) else str(message)
        async for chunk in _agent.astream(
            {"messages": [HumanMessage(content=user_text)]},
            config={"configurable": {"thread_id": session_id}},
            stream_mode="messages",
        ):
            msg, metadata = chunk
            if isinstance(msg, AIMessageChunk) and msg.content:
                await emit(queue, "message", {"text": msg.content})
            elif isinstance(msg, ToolMessage):
                await emit(queue, "log", {"line": str(msg.content)[:500]})
        await emit(queue, "done", {"status": "succeeded"})
        run_store[run_id].status = "succeeded"
    except Exception as exc:
        await emit(queue, "done", {"status": "failed", "error": str(exc)})
        run_store[run_id].status = "failed"


app = FastAPI(title="Champollion Pipeline Technician", lifespan=lifespan)
app.include_router(make_acp_router(run_store, execute_run))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "agent_ready": _agent is not None}
