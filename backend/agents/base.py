"""Stable, agent-neutral contracts used by the orchestrator."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from backend.workspace.models import AgentOutcomeReport, ProjectTask, WorkspaceEvent


AgentEventHandler = Callable[[WorkspaceEvent], Awaitable[None]]
AgentReportHandler = Callable[[AgentOutcomeReport], Awaitable[None]]


@dataclass(frozen=True)
class AgentDescriptor:
    """Routing metadata exposed by an agent without leaking its implementation."""

    agent_id: str
    description: str
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*_agent", self.agent_id):
            raise ValueError(
                "agent_id must be lower snake case and end with '_agent'"
            )
        if not self.description.strip():
            raise ValueError("agent description must not be empty")


@runtime_checkable
class AgentToolbox(Protocol):
    """Optional provider-neutral shape for model-callable agent tools."""

    @property
    def definitions(self) -> Sequence[Mapping[str, Any]]:
        """Return OpenAI-compatible function tool definitions."""
        ...

    def execute(self, name: str, arguments: Mapping[str, Any]) -> Any:
        """Run a named tool and return an agent-owned result value."""
        ...


@runtime_checkable
class AgentExecutor(Protocol):
    """The complete execution surface available to the orchestrator.

    Every specialist receives the same :class:`ProjectTask` and emits only
    :class:`WorkspaceEvent` and :class:`AgentOutcomeReport` values. Internal
    prompts, model providers, tools, and task models stay behind this boundary.
    """

    agent_id: str

    async def submit(
        self,
        task: ProjectTask,
        *,
        on_event: Optional[AgentEventHandler] = None,
        on_report: Optional[AgentReportHandler] = None,
    ) -> None:
        """Start an assigned task without exposing agent-specific state."""
        ...

    async def cancel(self, task_id: str) -> None:
        """Cancel an agent task if it is still active."""
        ...



@runtime_checkable
class ApprovalCapableAgent(Protocol):
    """Optional surface for specialists that can pause on consequential work."""

    async def approve(self, task_id: str, approval_id: str) -> None:
        """Approve one exact agent-generated consequential action."""
        ...
