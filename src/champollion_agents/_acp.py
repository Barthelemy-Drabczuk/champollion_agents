from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette import EventSourceResponse


@dataclass
class RunEntry:
    status: str
    queue: asyncio.Queue
    session_id: str


RunStore = dict[str, RunEntry]


class RunRequest(BaseModel):
    message: str | dict
    session_id: str | None = None


async def emit(queue: asyncio.Queue, event_type: str, payload: dict[str, Any]) -> None:
    await queue.put({"type": event_type, "payload": payload})


async def sse_generator(queue: asyncio.Queue) -> AsyncGenerator[dict, None]:
    while True:
        event = await queue.get()
        yield {"event": event["type"], "data": json.dumps(event["payload"])}
        if event["type"] == "done":
            break


ExecuteFn = Callable[[str, str | dict, asyncio.Queue, str], Awaitable[None]]


def make_acp_router(run_store: RunStore, execute_fn: ExecuteFn) -> APIRouter:
    router = APIRouter()

    @router.post("/runs", status_code=202)
    async def create_run(body: RunRequest) -> dict:
        run_id = str(uuid.uuid4())
        session_id = body.session_id or run_id
        queue: asyncio.Queue = asyncio.Queue()
        run_store[run_id] = RunEntry(status="running", queue=queue, session_id=session_id)
        asyncio.create_task(execute_fn(run_id, body.message, queue, session_id))
        return {"run_id": run_id}

    @router.get("/runs/{run_id}/stream")
    async def stream_run(run_id: str):
        if run_id not in run_store:
            return JSONResponse({"detail": "Run not found"}, status_code=404)
        return EventSourceResponse(sse_generator(run_store[run_id].queue))

    @router.get("/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        if run_id not in run_store:
            return JSONResponse({"detail": "Run not found"}, status_code=404)
        return {"run_id": run_id, "status": run_store[run_id].status}

    @router.post("/sessions", status_code=201)
    async def create_session() -> dict:
        return {"session_id": str(uuid.uuid4())}

    @router.post("/sessions/{session_id}/runs", status_code=202)
    async def add_turn(session_id: str, body: RunRequest) -> dict:
        body.session_id = session_id
        run_id = str(uuid.uuid4())
        queue: asyncio.Queue = asyncio.Queue()
        run_store[run_id] = RunEntry(status="running", queue=queue, session_id=session_id)
        asyncio.create_task(execute_fn(run_id, body.message, queue, session_id))
        return {"run_id": run_id}

    return router
