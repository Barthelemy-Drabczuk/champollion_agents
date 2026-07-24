from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from champollion_agents.technician.tools import poll_until_done, read_job_log_tail


@pytest.mark.unit
async def test_poll_until_done_returns_when_succeeded(tmp_output_dir):
    jobs_dir = tmp_output_dir / ".mcp_jobs"
    jobs_dir.mkdir()
    job_file = jobs_dir / "job-abc.json"
    job_file.write_text(json.dumps({"status": "running", "job_id": "job-abc"}))

    async def flip_status():
        await asyncio.sleep(0.05)
        job_file.write_text(json.dumps({"status": "succeeded", "job_id": "job-abc", "returncode": 0}))

    asyncio.create_task(flip_status())
    result = await poll_until_done("job-abc", str(tmp_output_dir), interval=0.02)
    assert result["status"] == "succeeded"


@pytest.mark.unit
async def test_poll_until_done_returns_on_failed(tmp_output_dir):
    jobs_dir = tmp_output_dir / ".mcp_jobs"
    jobs_dir.mkdir()
    job_file = jobs_dir / "job-xyz.json"
    job_file.write_text(json.dumps({"status": "failed", "job_id": "job-xyz"}))
    result = await poll_until_done("job-xyz", str(tmp_output_dir), interval=0.01)
    assert result["status"] == "failed"


@pytest.mark.unit
def test_read_job_log_tail_returns_last_n_lines(tmp_output_dir):
    jobs_dir = tmp_output_dir / ".mcp_jobs"
    jobs_dir.mkdir()
    log_file = jobs_dir / "job-log.log"
    lines = [f"line {i}\n" for i in range(100)]
    log_file.write_text("".join(lines))
    tail = read_job_log_tail("job-log", str(tmp_output_dir), n=5)
    assert tail.strip().endswith("line 99")
    assert len(tail.strip().splitlines()) == 5


@pytest.mark.unit
def test_read_job_log_tail_missing_file(tmp_output_dir):
    tail = read_job_log_tail("nonexistent", str(tmp_output_dir), n=10)
    assert "not found" in tail.lower() or tail == ""
