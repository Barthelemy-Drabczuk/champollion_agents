from __future__ import annotations

import pytest


@pytest.mark.unit
def test_detect_strategy_anthropic():
    from champollion_agents.llm import AnthropicStrategy, detect_strategy

    strategy = detect_strategy("sk-ant-api03-xxx", None, "claude-haiku-4-5-20251001")
    assert isinstance(strategy, AnthropicStrategy)
    assert strategy.model == "claude-haiku-4-5-20251001"
    assert strategy.api_key == "sk-ant-api03-xxx"


@pytest.mark.unit
def test_detect_strategy_openai_compatible():
    from champollion_agents.llm import OpenAICompatibleStrategy, detect_strategy

    strategy = detect_strategy("my-key", "http://localhost:11434/v1", "llama3.1:8b")
    assert isinstance(strategy, OpenAICompatibleStrategy)
    assert strategy.model == "llama3.1:8b"
    assert strategy.base_url == "http://localhost:11434/v1"


@pytest.mark.unit
def test_detect_strategy_empty_key_defaults_to_openai_compatible():
    from champollion_agents.llm import OpenAICompatibleStrategy, detect_strategy

    strategy = detect_strategy("", None, "some-model")
    assert isinstance(strategy, OpenAICompatibleStrategy)
    assert strategy.api_key == "ollama"


@pytest.mark.unit
def test_build_llm_returns_anthropic_instance(monkeypatch):
    monkeypatch.setenv("CHAMPOLLION_LLM_API_KEY", "sk-ant-api03-test")
    monkeypatch.delenv("CHAMPOLLION_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("CHAMPOLLION_LLM_MODEL", raising=False)

    from champollion_agents.llm import build_llm

    llm = build_llm()

    from langchain_anthropic import ChatAnthropic

    assert isinstance(llm, ChatAnthropic)
    assert llm.model == "claude-haiku-4-5-20251001"


@pytest.mark.unit
def test_build_llm_returns_openai_instance(monkeypatch):
    monkeypatch.setenv("CHAMPOLLION_LLM_API_KEY", "test-key")
    monkeypatch.delenv("CHAMPOLLION_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("CHAMPOLLION_LLM_MODEL", "gpt-4o")

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
