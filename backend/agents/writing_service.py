"""In-memory asynchronous task service for the writing agent."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from backend.llm.base import LLMClient
from backend.tools.writing import WritingToolbox

from .models import TaskStatus
from .writing import (
    WritingAgent,
    WritingAgentEvent,
    WritingAgentReport,
    WritingTask,
    WritingTaskRecord,
)


class DuplicateWritingTaskError(ValueError):
    pass


class UnknownWritingTaskError(KeyError):
    pass


EventHandler = Callable[[WritingAgentEvent], Awaitable[None]]
ReportHandler = Callable[[WritingAgentReport], Awaitable[None]]


class WritingAgentService:
    """Queues writing tasks and retains their state and outcome report."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        workspace_root: Path | str,
        workspace_roots: Optional[Mapping[str, Path | str]] = None,
        workspace_read_only_paths: Optional[
            Mapping[str, Sequence[str]]
        ] = None,
        llm_timeout_seconds: int = 120,
        max_calls: int = WritingAgent.MAX_CALLS,
        toolbox_factory: Callable[..., WritingToolbox] = WritingToolbox,
    ) -> None:
        self.llm = llm
        self.workspace_root = Path(workspace_root).resolve(strict=True)
        self.workspace_roots = {
            project_id: Path(root).resolve(strict=True)
            for project_id, root in (workspace_roots or {}).items()
        }
        self.workspace_read_only_paths = {
            project_id: tuple(paths)
            for project_id, paths in (workspace_read_only_paths or {}).items()
        }
        if any(not project_id.strip() for project_id in self.workspace_roots):
            raise ValueError("project workspace identifiers must not be empty")
        if any(
            project_id not in self.workspace_roots
            for project_id in self.workspace_read_only_paths
        ):
            raise ValueError(
                "read-only project inputs require an explicit project workspace"
            )
        if not self.workspace_root.is_dir() or any(
            not root.is_dir() for root in self.workspace_roots.values()
        ):
            raise ValueError("writing workspace roots must be directories")
        if llm_timeout_seconds < 1:
            raise ValueError("llm_timeout_seconds must be positive")
        if not 1 <= max_calls <= WritingAgent.MAX_CALLS:
            raise ValueError(
                f"writing max_calls must be between 1 and {WritingAgent.MAX_CALLS}"
            )
        self.llm_timeout_seconds = llm_timeout_seconds
        self.max_calls = max_calls
        self.toolbox_factory = toolbox_factory
        self._records: dict[str, WritingTaskRecord] = {}
        self._jobs: dict[str, asyncio.Task[None]] = {}
        self._event_handlers: dict[str, EventHandler] = {}
        self._report_handlers: dict[str, ReportHandler] = {}
        self._model_lock = asyncio.Lock()

    async def submit(
        self,
        task: WritingTask,
        *,
        on_event: Optional[EventHandler] = None,
        on_report: Optional[ReportHandler] = None,
    ) -> WritingTaskRecord:
        if task.task_id in self._records:
            raise DuplicateWritingTaskError(task.task_id)
        record = WritingTaskRecord(
            task=task,
            events=[
                WritingAgentEvent(
                    task_id=task.task_id,
                    status=TaskStatus.QUEUED,
                    message="Writing task queued",
                )
            ],
        )
        self._records[task.task_id] = record
        if on_event is not None:
            self._event_handlers[task.task_id] = on_event
        if on_report is not None:
            self._report_handlers[task.task_id] = on_report
        self._jobs[task.task_id] = asyncio.create_task(self._execute(task.task_id))
        return record.model_copy(deep=True)

    def get(self, task_id: str) -> WritingTaskRecord:
        record = self._records.get(task_id)
        if record is None:
            raise UnknownWritingTaskError(task_id)
        return record.model_copy(deep=True)

    async def cancel(self, task_id: str) -> WritingTaskRecord:
        record = self._records.get(task_id)
        if record is None:
            raise UnknownWritingTaskError(task_id)
        job = self._jobs.get(task_id)
        if job and not job.done():
            job.cancel()
            try:
                await job
            except asyncio.CancelledError:
                pass
        if record.report is None:
            event = WritingAgentEvent(
                task_id=task_id,
                status=TaskStatus.CANCELLED,
                message="Writing task cancelled before completion",
            )
            record.events.append(event)
            record.status = TaskStatus.CANCELLED
            record.report = WritingAgentReport(
                report_id=f"report-{uuid4()}",
                project_id=record.task.project_id,
                task_id=task_id,
                model=self.llm.model,
                outcome=TaskStatus.CANCELLED,
                summary=event.message,
                verification="not_run",
                unresolved_issues=[record.task.goal],
                recommended_next_action="Create a new writing task if the draft is still needed.",
            )
            try:
                await self._notify_event(task_id, event)
                await self._notify_report(task_id, record.report)
            finally:
                self._clear_handlers(task_id)
        return self.get(task_id)

    async def _execute(self, task_id: str) -> None:
        record = self._records[task_id]

        async def on_event(event: WritingAgentEvent) -> None:
            record.status = event.status
            record.events.append(event)
            await self._notify_event(task_id, event)

        root = self.workspace_roots.get(record.task.project_id, self.workspace_root)
        agent = WritingAgent(
            llm=self.llm,
            toolbox=self.toolbox_factory(
                root,
                read_only_paths=self.workspace_read_only_paths.get(
                    record.task.project_id, ()
                ),
            ),
            llm_timeout_seconds=self.llm_timeout_seconds,
            max_calls=self.max_calls,
            on_event=on_event,
        )
        try:
            async with self._model_lock:
                report = await agent.run(record.task)
            record.report = report
            record.status = report.outcome
            await self._notify_report(task_id, report)
        finally:
            self._clear_handlers(task_id)

    async def _notify_event(self, task_id: str, event: WritingAgentEvent) -> None:
        handler = self._event_handlers.get(task_id)
        if handler is not None:
            await handler(event)

    async def _notify_report(self, task_id: str, report: WritingAgentReport) -> None:
        handler = self._report_handlers.get(task_id)
        if handler is not None:
            await handler(report)

    def _clear_handlers(self, task_id: str) -> None:
        self._event_handlers.pop(task_id, None)
        self._report_handlers.pop(task_id, None)
