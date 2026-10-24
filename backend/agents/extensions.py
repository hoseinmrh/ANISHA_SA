"""Discovery and runtime context for externally supplied specialist agents."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from backend.llm import LLMClient, LLMSettings, create_llm_client

from .base import AgentDescriptor, AgentExecutor


@dataclass(frozen=True)
class AgentRuntimeContext:
    """Infrastructure passed to an extension's ``create_agent`` factory."""

    workspace_root: Path
    workspace_roots: Mapping[str, Path]
    llm_timeout_seconds: int

    def workspace_for(self, project_id: str) -> Path:
        return self.workspace_roots.get(project_id, self.workspace_root)

    def create_llm(
        self,
        agent_id: str,
        *,
        default_model: Optional[str] = None,
        default_context_length: int = 8_192,
        default_max_output_tokens: int = 512,
        default_temperature: float = 0.2,
        settings: Optional[LLMSettings] = None,
    ) -> LLMClient:
        """Create a provider-neutral client configured for one agent."""

        configured = settings or LLMSettings.for_agent(
            agent_id,
            default_model=default_model,
            default_context_length=default_context_length,
            default_max_output_tokens=default_max_output_tokens,
            default_temperature=default_temperature,
        )
        return create_llm_client(configured)

    @staticmethod
    def override_llm(settings: LLMSettings, **changes) -> LLMSettings:
        """Return a copy of immutable LLM settings with explicit overrides."""

        return replace(settings, **changes)


def load_agent_extensions(
    module_names: str,
    context: AgentRuntimeContext,
) -> list[AgentExecutor]:
    """Load comma-separated trusted Python modules from ``ANISHA_AGENT_MODULES``.

    Every module must expose ``create_agent(context)`` and return one object that
    implements :class:`AgentExecutor`. Imports are intentionally explicit rather
    than scanning the filesystem, keeping startup deterministic and auditable.
    """

    agents: list[AgentExecutor] = []
    for module_name in (value.strip() for value in module_names.split(",")):
        if not module_name:
            continue
        module = importlib.import_module(module_name)
        factory = getattr(module, "create_agent", None)
        if not callable(factory):
            raise RuntimeError(
                f"Agent extension {module_name!r} must define create_agent(context)"
            )
        agent = factory(context)
        if not isinstance(agent, AgentExecutor):
            raise TypeError(
                f"Agent extension {module_name!r} did not return an AgentExecutor"
            )
        if not isinstance(getattr(agent, "descriptor", None), AgentDescriptor):
            raise TypeError(
                f"Agent extension {module_name!r} must expose an AgentDescriptor"
            )
        agents.append(agent)
    return agents
