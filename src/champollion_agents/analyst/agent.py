from __future__ import annotations

import os
import sys

from claude_code_sdk import ClaudeCodeOptions

from champollion_agents.sdk_agent import SdkAgent

_SYSTEM_PROMPT = (
    "You are the Champollion Data Analyst. "
    "You have access to ChromaDB embeddings from the sulcal pipeline. "
    "Help researchers explore subject similarities, region statistics, and embedding distributions. "
    "Always report region names with hemisphere (e.g. SC-sylv_left)."
)


def build_analyst_agent(output_dir: str) -> SdkAgent:
    env = {k: str(v) for k, v in os.environ.items()}
    env["CHAMPOLLION_OUTPUT_DIR"] = output_dir

    options = ClaudeCodeOptions(
        system_prompt=_SYSTEM_PROMPT,
        mcp_servers={
            "analyst_tools": {
                "command": sys.executable,
                "args": ["-m", "champollion_agents.analyst.mcp_server"],
                "env": env,
            }
        },
        model=os.environ.get("CHAMPOLLION_LLM_MODEL", "claude-haiku-4-5-20251001"),
        permission_mode="bypassPermissions",
    )
    return SdkAgent(options)
