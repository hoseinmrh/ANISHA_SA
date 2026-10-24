"""A small background executor base for new standard-contract agents."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional
from uuid import uuid4

from backend.workspace.models import (
    AgentOutcomeReport,
    ProjectTask,
    TaskState,
    WorkspaceEvent,
)

from .base import AgentDescriptor, AgentEventHandler, AgentReportHandler


class DuplicateAgentTaskError(ValueError):
    pass


class UnknownAgentTaskError(KeyError):
    pass


class BackgroundAgentExecutor(ABC):
    """Manage async lifecycle while a specialist implements only ``run``.

    ``submit`` returns after scheduling the job. Events and the terminal report
    are delivered through the same callbacks used by every other agent.
    """

    descriptor: AgentDescriptor

    def __init__(self) -> None:
        self.agent_id = self.descriptor.agent_id
        self._tasks: dict[str, ProjectTask] = {}
        self._jobs: dict[str, asyncio.Task[None]] = {}
        self._event_handlers: dict[str, AgentEventHandler] = {}
        self._report_handlers: dict[str, AgentReportHandler] = {}

    async def submit(
        self,
        task: ProjectTask,
        *,
        on_event: Optional[AgentEventHandler] = None,
        on_report: Optional[AgentReportHandler] = None,
    ) -> None:
        if task.assigned_agent != self.agent_id:
            raise ValueError(
                f"task is assigned to {task.assigned_agent}, not {self.agent_id}"
            )
        if task.task_id in self._tasks:
            raise DuplicateAgentTaskError(task.task_id)
        self._tasks[task.task_id] = task.model_copy(deep=True)
        if on_event is not None:
            self._event_handlers[task.task_id] = on_event
        if on_report is not None:
            self._report_handlers[task.task_id] = on_report
        self._jobs[task.task_id] = asyncio.create_task(self._execute(task.task_id))

    async def cancel(self, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if task is None:
            raise UnknownAgentTaskError(task_id)
        job = self._jobs[task_id]
        if job.done():
            return
        job.cancel()
        try:
            await job
        except asyncio.CancelledError:
            pass
        await self._emit(task, TaskState.CANCELLED, "Task cancelled")
        await self._report(
            AgentOutcomeReport(
                report_id=f"report-{uuid4()}",
                project_id=task.project_id,
                task_id=task.task_id,
                agent_id=self.agent_id,
                outcome=TaskState.CANCELLED,
                summary="Task cancelled before completion.",
                unresolved_issues=[task.goal],
                recommended_next_action="Create a new task if the work is still needed.",
            )
        )
        self._clear_handlers(task_id)

    @abstractmethod
    async def run(self, task: ProjectTask) -> AgentOutcomeReport:
        """Perform one task and return exactly one terminal standard report."""

    async def emit(
        self, task: ProjectTask, state: TaskState, message: str
    ) -> None:
        """Publish standard progress from a subclass's ``run`` method."""

        await self._emit(task, state, message)

    async def _execute(self, task_id: str) -> None:
        task = self._tasks[task_id]
        was_cancelled = False
        try:
            report = await self.run(task.model_copy(deep=True))
            if report.task_id != task_id or report.agent_id != self.agent_id:
                raise ValueError("agent returned a report for a different task or agent")
            await self._report(report)
        except asyncio.CancelledError:
            was_cancelled = True
            raise
        except Exception as exc:
            await self._emit(task, TaskState.FAILED, f"Agent execution failed: {exc}")
            await self._report(
                AgentOutcomeReport(
                    report_id=f"report-{uuid4()}",
                    project_id=task.project_id,
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    outcome=TaskState.FAILED,
                    summary=f"Agent execution failed: {exc}",
                    unresolved_issues=[task.goal],
                    recommended_next_action="Inspect the failure and retry the task.",
                )
            )
        finally:
            if not was_cancelled:
                self._clear_handlers(task_id)

    async def _emit(
        self, task: ProjectTask, state: TaskState, message: str
    ) -> None:
        handler = self._event_handlers.get(task.task_id)
        if handler is not None:
            await handler(
                WorkspaceEvent(
                    event_id=f"event-{uuid4()}",
                    project_id=task.project_id,
                    task_id=task.task_id,
                    actor=self.agent_id,
                    state=state,
                    message=message,
                )
            )

    async def _report(self, report: AgentOutcomeReport) -> None:
        handler = self._report_handlers.get(report.task_id)
        if handler is not None:
            await handler(report)

    def _clear_handlers(self, task_id: str) -> None:
        self._event_handlers.pop(task_id, None)
        self._report_handlers.pop(task_id, None)
