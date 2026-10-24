"""Writing-agent configuration for the reusable service adapter."""

from __future__ import annotations

from backend.workspace.models import ProjectTask

from .adapter import ServiceAgentAdapter
from .base import AgentDescriptor
from .writing import WritingTask
from .writing_service import WritingAgentService


WRITING_AGENT = AgentDescriptor(
    agent_id="writing_agent",
    description="Creates and revises grounded Markdown documents from project sources.",
    capabilities=("writing", "summarization", "document revision", "Markdown"),
)


class WritingAgentAdapter(ServiceAgentAdapter[WritingTask]):

    def __init__(self, service: WritingAgentService) -> None:
        super().__init__(
            service=service,
            descriptor=WRITING_AGENT,
            task_factory=_writing_task,
        )


def _writing_task(task: ProjectTask) -> WritingTask:
    return WritingTask(
        project_id=task.project_id,
        task_id=task.task_id,
        goal=task.goal,
        constraints=task.constraints,
        source_paths=task.source_paths,
        context=task.context,
        context_bundle=task.context_bundle,
        target_path=task.target_path,
        target_mode=task.target_mode,
        finish_after_first_section=task.finish_after_first_section,
    )
