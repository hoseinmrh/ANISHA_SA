"""Environment-driven LLM provider selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .base import LLMClient


DEFAULT_CODING_MODEL = "gpt-5.4-mini-2026-03-17"


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "openai"
    model: str = DEFAULT_CODING_MODEL
    ollama_host: str = "http://127.0.0.1:11434"
    context_length: int = 8_192
    max_output_tokens: int = 512
    temperature: Optional[float] = 0.2
    keep_alive: str = "10m"
    think: bool = False
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "LLMSettings":
        """Load the legacy coding/default configuration.

        New agents should use :meth:`for_agent`, which adds per-agent overrides
        while retaining these variables as backwards-compatible fallbacks.
        """

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        return cls(
            provider=os.getenv("ANISHA_LLM_PROVIDER", "openai").strip().lower(),
            model=os.getenv("ANISHA_CODING_MODEL", DEFAULT_CODING_MODEL).strip(),
            ollama_host=os.getenv(
                "ANISHA_OLLAMA_HOST", "http://127.0.0.1:11434"
            ).strip(),
            context_length=_positive_int("ANISHA_CONTEXT_LENGTH", 8_192),
            max_output_tokens=_positive_int("ANISHA_MAX_OUTPUT_TOKENS", 512),
            temperature=_float_value("ANISHA_TEMPERATURE", 0.2),
            keep_alive=os.getenv("ANISHA_KEEP_ALIVE", "10m").strip(),
            think=_bool_value("ANISHA_OLLAMA_THINK", False),
            api_key=os.getenv("ANISHA_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("ANISHA_LLM_BASE_URL"),
        )

    @classmethod
    def for_agent(
        cls,
        agent_id: str,
        *,
        default_model: Optional[str] = None,
        default_context_length: int = 8_192,
        default_max_output_tokens: int = 512,
        default_temperature: float = 0.2,
    ) -> "LLMSettings":
        """Load provider settings with ``ANISHA_<AGENT>_*`` overrides.

        ``writing_agent`` becomes the environment prefix ``ANISHA_WRITING``.
        Values fall back to the shared ``ANISHA_LLM_*`` settings and then to
        the supplied defaults, so each agent may use a different provider,
        endpoint, API key, or local model without changing its implementation.
        """

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        prefix = _agent_env_prefix(agent_id)
        model_default = default_model or DEFAULT_CODING_MODEL
        return cls(
            provider=_agent_value(
                prefix, "LLM_PROVIDER", "ANISHA_LLM_PROVIDER", "openai"
            ).lower(),
            model=_agent_value(
                prefix, "MODEL", "ANISHA_LLM_MODEL", model_default
            ),
            ollama_host=_agent_value(
                prefix,
                "OLLAMA_HOST",
                "ANISHA_OLLAMA_HOST",
                "http://127.0.0.1:11434",
            ),
            context_length=_agent_positive_int(
                prefix,
                "CONTEXT_LENGTH",
                "ANISHA_CONTEXT_LENGTH",
                default_context_length,
            ),
            max_output_tokens=_agent_positive_int(
                prefix,
                "MAX_OUTPUT_TOKENS",
                "ANISHA_MAX_OUTPUT_TOKENS",
                default_max_output_tokens,
            ),
            temperature=_agent_float(
                prefix,
                "TEMPERATURE",
                "ANISHA_TEMPERATURE",
                default_temperature,
            ),
            keep_alive=_agent_value(
                prefix, "KEEP_ALIVE", "ANISHA_KEEP_ALIVE", "10m"
            ),
            think=_agent_bool(
                prefix, "OLLAMA_THINK", "ANISHA_OLLAMA_THINK", False
            ),
            api_key=(
                os.getenv(f"{prefix}_LLM_API_KEY")
                or os.getenv("ANISHA_LLM_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            ),
            base_url=(
                os.getenv(f"{prefix}_LLM_BASE_URL")
                or os.getenv("ANISHA_LLM_BASE_URL")
            ),
        )


def create_llm_client(settings: Optional[LLMSettings] = None) -> LLMClient:
    config = settings or LLMSettings.from_env()
    if not config.model:
        raise ValueError("LLM model must not be empty")

    if config.provider == "ollama":
        from .ollama import OllamaLLMClient

        if config.temperature is None:
            raise ValueError("Ollama requires an explicit temperature")

        return OllamaLLMClient(
            model=config.model,
            host=config.ollama_host,
            context_length=config.context_length,
            max_output_tokens=config.max_output_tokens,
            temperature=config.temperature,
            keep_alive=config.keep_alive,
            think=config.think,
        )

    if config.provider in {"openai", "openai-compatible"}:
        from .openai_compatible import OpenAICompatibleLLMClient

        if not config.api_key:
            raise RuntimeError("ANISHA_LLM_API_KEY is required for this provider")
        return OpenAICompatibleLLMClient(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            max_tokens_parameter=(
                "max_completion_tokens"
                if config.provider == "openai"
                else "max_tokens"
            ),
        )

    raise ValueError(f"Unsupported ANISHA_LLM_PROVIDER: {config.provider}")


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _float_value(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not 0 <= value <= 2:
        raise ValueError(f"{name} must be between 0 and 2")
    return value


def _bool_value(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{name} must be a boolean")
    return normalized in {"true", "1", "yes"}


def _agent_env_prefix(agent_id: str) -> str:
    normalized = agent_id.strip().lower()
    if normalized.endswith("_agent"):
        normalized = normalized[: -len("_agent")]
    normalized = "_".join(part for part in normalized.split("_") if part)
    if not normalized or not normalized.replace("_", "").isalnum():
        raise ValueError("agent_id must contain letters, numbers, or underscores")
    return f"ANISHA_{normalized.upper()}"


def _agent_value(
    prefix: str, suffix: str, fallback_name: str, default: str
) -> str:
    return os.getenv(f"{prefix}_{suffix}", os.getenv(fallback_name, default)).strip()


def _agent_positive_int(
    prefix: str, suffix: str, fallback_name: str, default: int
) -> int:
    name = f"{prefix}_{suffix}"
    value = int(os.getenv(name, os.getenv(fallback_name, str(default))))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _agent_float(
    prefix: str, suffix: str, fallback_name: str, default: float
) -> float:
    name = f"{prefix}_{suffix}"
    value = float(os.getenv(name, os.getenv(fallback_name, str(default))))
    if not 0 <= value <= 2:
        raise ValueError(f"{name} must be between 0 and 2")
    return value


def _agent_bool(
    prefix: str, suffix: str, fallback_name: str, default: bool
) -> bool:
    name = f"{prefix}_{suffix}"
    raw = os.getenv(name, os.getenv(fallback_name))
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{name} must be a boolean")
    return normalized in {"true", "1", "yes"}
