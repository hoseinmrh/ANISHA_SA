"""Coding-agent configuration for the reusable service adapter."""

from __future__ import annotations

from backend.workspace.models import ProjectTask

from .adapter import ServiceAgentAdapter
from .base import AgentDescriptor
from .models import CodingTask
from .service import CodingAgentService


CODING_AGENT = AgentDescriptor(
    agent_id="coding_agent",
    description="Implements, debugs, tests, and inspects source-code projects.",
    capabilities=("code", "debugging", "tests", "repository inspection"),
)


class CodingAgentAdapter(ServiceAgentAdapter[CodingTask]):

    def __init__(self, service: CodingAgentService) -> None:
        super().__init__(
            service=service,
            descriptor=CODING_AGENT,
            task_factory=_coding_task,
        )


def _coding_task(task: ProjectTask) -> CodingTask:
    return CodingTask(
        project_id=task.project_id,
        task_id=task.task_id,
        goal=task.goal,
        constraints=task.constraints,
        context=task.context,
        context_bundle=task.context_bundle,
    )
