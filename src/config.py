from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the memory systems lab.

    Attributes:
        base_dir: Root directory of the repository.
        data_dir: Directory containing benchmark dataset JSON files.
        state_dir: Directory where persistent state (e.g., profiles/User.md) is saved.
        compact_threshold_tokens: Token count threshold to trigger compaction.
        compact_keep_messages: Number of recent messages to preserve during compaction.
        model: ProviderConfig for the main conversational agent.
        judge_model: ProviderConfig for the judge/evaluation model.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load configuration from environment variables and defaults.

    1. Resolve repository root.
    2. Load `.env` file if available.
    3. Ensure `state/` and `state/profiles/` exist.
    4. Populate and return a LabConfig instance.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load environment variables from .env in repository root
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    # Data and State directories
    data_dir = root / "data"
    state_dir = root / "state"
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    # Compaction settings
    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "600"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    # Provider configuration
    provider_raw = os.getenv("LLM_PROVIDER", "openai")
    provider = normalize_provider(provider_raw)
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Resolve API keys & base URLs per provider
    api_key: str | None = None
    base_url: str | None = None

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
    elif provider == "custom":
        api_key = os.getenv("CUSTOM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        base_url = os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    main_model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    # Judge model configuration (optional override via JUDGE_* env vars)
    judge_provider_raw = os.getenv("JUDGE_PROVIDER", provider)
    judge_provider = normalize_provider(judge_provider_raw)
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)
    judge_api_key = os.getenv("JUDGE_API_KEY", api_key)
    judge_base_url = os.getenv("JUDGE_BASE_URL", base_url)

    judge_model = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=main_model,
        judge_model=judge_model,
    )
