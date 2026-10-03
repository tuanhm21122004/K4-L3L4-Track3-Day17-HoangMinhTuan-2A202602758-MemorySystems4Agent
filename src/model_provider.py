from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Configuration for chat model providers.

    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider name aliases into standard names."""
    val = value.strip().lower()
    mapping = {
        "openai": "openai",
        "chatgpt": "openai",
        "gpt": "openai",
        "custom": "custom",
        "local": "custom",
        "vllm": "custom",
        "openai-compatible": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "google-genai": "gemini",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "anthorpic": "anthropic",  # common typo handled
        "ollama": "ollama",
        "openrouter": "openrouter",
        "open-router": "openrouter",
    }
    if val in mapping:
        return mapping[val]
    return val


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate chat model based on provider configuration."""
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    elif provider == "custom":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "base_url": config.base_url or "http://localhost:8000/v1",
            "api_key": config.api_key or "EMPTY",
        }
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model_name": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["anthropic_api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    elif provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter

            kwargs = {
                "model_name": config.model_name,
                "temperature": config.temperature,
            }
            if config.api_key:
                kwargs["openrouter_api_key"] = config.api_key
            if config.base_url:
                kwargs["openrouter_api_base"] = config.base_url
            return ChatOpenRouter(**kwargs)
        except (ImportError, Exception):
            # Fallback to OpenAI-compatible ChatOpenAI with openrouter base url
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": config.model_name,
                "temperature": config.temperature,
                "base_url": config.base_url or "https://openrouter.ai/api/v1",
                "api_key": config.api_key,
            }
            return ChatOpenAI(**kwargs)

    else:
        raise ValueError(
            f"Unsupported provider: {config.provider}. Supported: openai, custom, gemini, anthropic, ollama, openrouter."
        )
