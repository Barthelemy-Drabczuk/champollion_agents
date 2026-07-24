from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from champollion_agents.llm import build_llm
from champollion_agents.analyst.tools import make_analyst_tools


def build_analyst_agent(output_dir: str):
    llm = build_llm()
    tools = make_analyst_tools(output_dir)
    memory = MemorySaver()
    return create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        prompt=(
            "You are the Champollion Data Analyst. "
            "You have access to ChromaDB embeddings from the sulcal pipeline. "
            "Help researchers explore subject similarities, region statistics, and embedding distributions. "
            "Always report region names with hemisphere (e.g. SC-sylv_left)."
        ),
    )
