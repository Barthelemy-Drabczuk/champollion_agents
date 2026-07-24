from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("CHAMPOLLION_LLM_MODEL", "gpt-4o"),
        base_url=os.environ.get("CHAMPOLLION_LLM_BASE_URL") or None,
        api_key=os.environ.get("CHAMPOLLION_LLM_API_KEY", "ollama"),
        temperature=0,
        streaming=True,
    )
