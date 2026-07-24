from __future__ import annotations

import asyncio
import pytest

from champollion_agents._acp import emit, sse_generator, RunStore, RunEntry


@pytest.mark.unit
async def test_emit_puts_event_in_queue():
    queue: asyncio.Queue = asyncio.Queue()
    await emit(queue, "message", {"text": "hello"})
    event = queue.get_nowait()
    assert event == {"type": "message", "payload": {"text": "hello"}}


@pytest.mark.unit
async def test_sse_generator_stops_on_done():
    queue: asyncio.Queue = asyncio.Queue()
    await emit(queue, "thought", {"text": "thinking"})
    await emit(queue, "message", {"text": "result"})
    await emit(queue, "done", {"status": "succeeded"})

    events = []
    async for item in sse_generator(queue):
        events.append(item)

    assert len(events) == 3
    assert events[0]["event"] == "thought"
    assert events[2]["event"] == "done"


@pytest.mark.unit
def test_run_store_entry():
    queue: asyncio.Queue = asyncio.Queue()
    entry = RunEntry(status="running", queue=queue, session_id="sess-1")
    assert entry.status == "running"
