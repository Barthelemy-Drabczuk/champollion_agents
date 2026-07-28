from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel


class LLMStrategy(ABC):
    @abstractmethod
    def build(self) -> BaseChatModel: ...


@dataclass
class AnthropicStrategy(LLMStrategy):
    model: str
    api_key: str

    def build(self) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=self.model, api_key=self.api_key, streaming=True)


@dataclass
class OpenAICompatibleStrategy(LLMStrategy):
    model: str
    api_key: str
    base_url: str | None

    def build(self) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=0,
            streaming=True,
        )


def detect_strategy(api_key: str, base_url: str | None, model: str) -> LLMStrategy:
    if api_key.startswith("sk-ant-"):
        return AnthropicStrategy(model=model, api_key=api_key)
    return OpenAICompatibleStrategy(model=model, api_key=api_key or "ollama", base_url=base_url)


def build_llm() -> BaseChatModel:
    api_key = os.environ.get("CHAMPOLLION_LLM_API_KEY") or os.environ.get("CHAMPOLLION_API_KEY", "")
    base_url = os.environ.get("CHAMPOLLION_LLM_BASE_URL")
    model = os.environ.get("CHAMPOLLION_LLM_MODEL", "claude-haiku-4-5-20251001")
    return detect_strategy(api_key, base_url, model).build()
