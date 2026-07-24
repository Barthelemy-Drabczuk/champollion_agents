from __future__ import annotations

import pytest


@pytest.mark.unit
def test_build_llm_returns_openai_instance(monkeypatch):
    monkeypatch.setenv("CHAMPOLLION_LLM_API_KEY", "test-key")
    monkeypatch.delenv("CHAMPOLLION_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("CHAMPOLLION_LLM_MODEL", raising=False)

    from champollion_agents.llm import build_llm
    llm = build_llm()

    from langchain_openai import ChatOpenAI
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o"
    assert llm.temperature == 0
    assert llm.streaming is True


@pytest.mark.unit
def test_build_llm_respects_env_overrides(monkeypatch):
    monkeypatch.setenv("CHAMPOLLION_LLM_API_KEY", "my-key")
    monkeypatch.setenv("CHAMPOLLION_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("CHAMPOLLION_LLM_MODEL", "llama3.1:70b")

    from champollion_agents.llm import build_llm
    llm = build_llm()

    assert llm.model_name == "llama3.1:70b"
    assert str(llm.openai_api_base) == "http://localhost:11434/v1"
