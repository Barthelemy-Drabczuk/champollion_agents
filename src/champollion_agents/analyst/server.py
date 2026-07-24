from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from champollion_agents._acp import RunStore, emit, make_acp_router
from champollion_agents.analyst.agent import build_analyst_agent
from champollion_agents.analyst.indexer import index_pipeline_outputs

run_store: RunStore = {}
_agent = None
_output_dir: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _output_dir
    _output_dir = os.environ.get("CHAMPOLLION_OUTPUT_DIR", "./outputs")
    _agent = build_analyst_agent(_output_dir)
    yield


async def execute_run(run_id: str, message: str | dict, queue: asyncio.Queue, session_id: str) -> None:
    try:
        user_text = message if isinstance(message, str) else str(message)
        async for chunk in _agent.astream(
            {"messages": [HumanMessage(content=user_text)]},
            config={"configurable": {"thread_id": session_id}},
            stream_mode="messages",
        ):
            msg, _ = chunk
            if isinstance(msg, AIMessageChunk) and msg.content:
                await emit(queue, "message", {"text": msg.content})
            elif isinstance(msg, ToolMessage):
                await emit(queue, "log", {"line": str(msg.content)[:500]})
        await emit(queue, "done", {"status": "succeeded"})
        run_store[run_id].status = "succeeded"
    except Exception as exc:
        await emit(queue, "done", {"status": "failed", "error": str(exc)})
        run_store[run_id].status = "failed"


app = FastAPI(title="Champollion Data Analyst", lifespan=lifespan)
app.include_router(make_acp_router(run_store, execute_run))


@app.post("/index")
async def index_embeddings(combined_embeddings_path: str, run_id: str) -> dict:
    n = await index_pipeline_outputs(combined_embeddings_path, _output_dir, run_id)
    return {"indexed": n, "run_id": run_id}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "agent_ready": _agent is not None}
