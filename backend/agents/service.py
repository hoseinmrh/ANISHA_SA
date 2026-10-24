"""In-memory asynchronous task service for coding-agent version 1.0.0."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Mapping
from pathlib import Path
from typing import Callable, Optional, Sequence
from uuid import uuid4

from backend.llm.base import LLMClient
from backend.tools.coding import CodingToolbox

from .coding import CodingAgent
from .models import (
    AgentEvent,
    AgentReport,
    CodingTask,
    CodingTaskRecord,
    TaskStatus,
    VerificationSummary,
)
from .python_environment import PythonEnvironmentManager


class DuplicateTaskError(ValueError):
    pass


class UnknownTaskError(KeyError):
    pass


EventHandler = Callable[[AgentEvent], Awaitable[None]]
ReportHandler = Callable[[AgentReport], Awaitable[None]]


class CodingAgentService:
    """Queues coding tasks and retains their current state and outcome report."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        workspace_root: Path | str,
        workspace_roots: Optional[Mapping[str, Path | str]] = None,
        workspace_read_only_paths: Optional[Mapping[str, Sequence[str]]] = None,
        max_steps: int = 20,
        llm_timeout_seconds: int = 120,
        toolbox_factory: Callable[..., CodingToolbox] = CodingToolbox,
        environment_manager_factory: Callable[
            [Path], PythonEnvironmentManager
        ] = PythonEnvironmentManager,
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
            raise ValueError("coding workspace roots must be directories")
        self.max_steps = max_steps
        self.llm_timeout_seconds = llm_timeout_seconds
        self.toolbox_factory = toolbox_factory
        self.environment_manager_factory = environment_manager_factory
        self._records: dict[str, CodingTaskRecord] = {}
        self._jobs: dict[str, asyncio.Task[None]] = {}
        self._event_handlers: dict[str, EventHandler] = {}
        self._report_handlers: dict[str, ReportHandler] = {}
        self._preserved_changed_files: dict[str, set[str]] = {}
        self._model_lock = asyncio.Lock()

    async def submit(
        self,
        task: CodingTask,
        *,
        on_event: Optional[EventHandler] = None,
        on_report: Optional[ReportHandler] = None,
    ) -> CodingTaskRecord:
        if task.task_id in self._records:
            raise DuplicateTaskError(task.task_id)
        record = CodingTaskRecord(
            task=task,
            events=[
                AgentEvent(
                    task_id=task.task_id,
                    status=TaskStatus.QUEUED,
                    message="Coding task queued",
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

    def get(self, task_id: str) -> CodingTaskRecord:
        record = self._records.get(task_id)
        if record is None:
            raise UnknownTaskError(task_id)
        return record.model_copy(deep=True)

    async def cancel(self, task_id: str) -> CodingTaskRecord:
        record = self._records.get(task_id)
        if record is None:
            raise UnknownTaskError(task_id)
        job = self._jobs.get(task_id)
        if job and not job.done():
            job.cancel()
            try:
                await job
            except asyncio.CancelledError:
                pass
        created_event: Optional[AgentEvent] = None
        created_report: Optional[AgentReport] = None
        if record.status not in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            event = AgentEvent(
                task_id=task_id,
                status=TaskStatus.CANCELLED,
                message="Coding task cancelled by the user",
            )
            record.events.append(event)
            record.status = TaskStatus.CANCELLED
            record.report = AgentReport(
                report_id=(
                    f"report-{task_id}"
                    if record.report is None
                    else f"report-{task_id}-{uuid4()}"
                ),
                project_id=record.task.project_id,
                task_id=task_id,
                model=self.llm.model,
                outcome=TaskStatus.CANCELLED,
                summary=event.message,
                verification=VerificationSummary(
                    result="not_run",
                    limitations=["The task was cancelled before it could run."],
                ),
                unresolved_issues=[record.task.goal],
                recommended_next_action="Create a new task if the work is still needed.",
            )
            created_event = event
            created_report = record.report
        if created_event is not None:
            await self._notify_event(task_id, created_event)
        if created_report is not None:
            await self._notify_report(task_id, created_report)
        self._preserved_changed_files.pop(task_id, None)
        self._clear_handlers(task_id)
        return self.get(task_id)

    async def approve(self, task_id: str, approval_id: str) -> CodingTaskRecord:
        """Approve one exact pending Python environment request and resume."""

        record = self._records.get(task_id)
        if record is None:
            raise UnknownTaskError(task_id)
        request = record.report.approval_request if record.report is not None else None
        if (
            record.status != TaskStatus.AWAITING_APPROVAL
            or request is None
            or request.approval_id != approval_id
        ):
            raise ValueError("approval does not match the task's pending request")

        self._preserved_changed_files[task_id] = set(record.report.artefacts_created)
        record.report = None
        event = AgentEvent(
            task_id=task_id,
            status=TaskStatus.QUEUED,
            message=(
                "Python environment setup approved by the user; preparation queued"
            ),
        )
        record.status = event.status
        record.events.append(event)
        await self._notify_event(task_id, event)
        self._jobs[task_id] = asyncio.create_task(
            self._prepare_environment_and_resume(task_id, request.packages)
        )
        return self.get(task_id)

    async def _prepare_environment_and_resume(
        self, task_id: str, packages: Sequence[str]
    ) -> None:
        record = self._records[task_id]
        event = AgentEvent(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            message="Preparing approved project-local Python environment",
        )
        record.status = event.status
        record.events.append(event)
        await self._notify_event(task_id, event)
        root = self.workspace_roots.get(record.task.project_id, self.workspace_root)
        try:
            manager = self.environment_manager_factory(root)
            await asyncio.to_thread(manager.prepare, packages)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            setup_error = _safe_setup_error(exc)
            failed_event = AgentEvent(
                task_id=task_id,
                status=TaskStatus.FAILED,
                message=f"Python environment setup failed: {setup_error}",
            )
            record.status = failed_event.status
            record.events.append(failed_event)
            await self._notify_event(task_id, failed_event)
            report = AgentReport(
                report_id=f"report-{task_id}-{uuid4()}",
                project_id=record.task.project_id,
                task_id=task_id,
                model=self.llm.model,
                outcome=TaskStatus.FAILED,
                summary=f"Approved Python environment setup failed: {setup_error}",
                verification=VerificationSummary(
                    result="failed",
                    limitations=[setup_error],
                ),
                unresolved_issues=["The project dependencies are not installed."],
                recommended_next_action=(
                    "Review the environment setup error, then retry the task."
                ),
            )
            record.report = report
            record.status = report.outcome
            try:
                await self._notify_report(task_id, report)
            finally:
                self._preserved_changed_files.pop(task_id, None)
                self._clear_handlers(task_id)
            return

        ready = AgentEvent(
            task_id=task_id,
            status=TaskStatus.QUEUED,
            message="Approved Python environment is ready; coding task resumed",
        )
        record.status = ready.status
        record.events.append(ready)
        await self._notify_event(task_id, ready)
        await self._execute(task_id)

    async def _execute(self, task_id: str) -> None:
        record = self._records[task_id]

        async def on_event(event: AgentEvent) -> None:
            record.status = event.status
            record.events.append(event)
            await self._notify_event(task_id, event)

        toolbox = self.toolbox_factory(
            self.workspace_roots.get(record.task.project_id, self.workspace_root),
            read_only_paths=self.workspace_read_only_paths.get(
                record.task.project_id, ()
            ),
        )
        toolbox.changed_files.update(
            self._preserved_changed_files.get(task_id, set())
        )
        agent = CodingAgent(
            llm=self.llm,
            toolbox=toolbox,
            max_steps=self.max_steps,
            llm_timeout_seconds=self.llm_timeout_seconds,
            on_event=on_event,
        )
        async with self._model_lock:
            report = await agent.run(record.task)
        record.report = report
        record.status = report.outcome
        try:
            await self._notify_report(task_id, report)
        finally:
            if report.outcome != TaskStatus.AWAITING_APPROVAL:
                self._preserved_changed_files.pop(task_id, None)
                self._clear_handlers(task_id)

    async def _notify_event(self, task_id: str, event: AgentEvent) -> None:
        handler = self._event_handlers.get(task_id)
        if handler is not None:
            await handler(event)

    async def _notify_report(self, task_id: str, report: AgentReport) -> None:
        handler = self._report_handlers.get(task_id)
        if handler is not None:
            await handler(report)

    def _clear_handlers(self, task_id: str) -> None:
        self._event_handlers.pop(task_id, None)
        self._report_handlers.pop(task_id, None)


def _safe_setup_error(exc: Exception) -> str:
    """Keep setup failures actionable without exposing credential-bearing URLs."""

    detail = f"{type(exc).__name__}: {exc}"
    detail = re.sub(r"https?://\S+", "[redacted-url]", detail)
    return " ".join(detail.split())[-500:]
