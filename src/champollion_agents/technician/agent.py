from __future__ import annotations

import os

from claude_code_sdk import ClaudeCodeOptions

from champollion_agents.sdk_agent import SdkAgent

SYSTEM_PROMPT = """You are the Champollion Pipeline Technician. Your job is to:
1. Run preflight_check first on every request.
2. Adapt parameters: if no CUDA, inject cpu=True; if njobs unset and >8 cores, set njobs=cpu_count//2.
3. Launch pipeline stages sequentially via the MCP tools.
4. After launching each job, poll for completion by checking its status file every 10s:
   Job files are at <output_dir>/.mcp_jobs/<job_id>.json
   Use Bash: `while true; do status=$(python3 -c "import json,sys; d=json.load(open('$FILE')); print(d['status'])" 2>/dev/null); [ "$status" = succeeded ] || [ "$status" = failed ] || [ "$status" = cancelled ] && break; sleep 10; done; cat $FILE`
5. On failure, read the log with: tail -50 <output_dir>/.mcp_jobs/<job_id>.log
6. On combine success, hand off by calling the analyst REST API:
   curl -s -X POST "<analyst_url>/index?combined_embeddings_path=<path>&run_id=<id>"

Never guess paths — ask the user for any missing required path.
Always return job_id and output path immediately after launching each stage."""


def build_technician_agent(
    mcp_dir: str,
    pipeline_dir: str,
    analyst_url: str = "http://localhost:8002",
) -> SdkAgent:
    env = {k: str(v) for k, v in os.environ.items()}
    env["CHAMPOLLION_PIPELINE_DIR"] = pipeline_dir

    system_prompt = SYSTEM_PROMPT.replace("<analyst_url>", analyst_url)

    options = ClaudeCodeOptions(
        system_prompt=system_prompt,
        mcp_servers={
            "champollion_sulcal": {
                "command": "pixi",
                "args": ["run", "--manifest-path", f"{mcp_dir}/pixi.toml", "run"],
                "env": env,
            }
        },
        model=os.environ.get("CHAMPOLLION_LLM_MODEL", "claude-haiku-4-5-20251001"),
        permission_mode="bypassPermissions",
    )
    return SdkAgent(options)
