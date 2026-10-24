"""Reusable adapter from agent-specific services to the public agent contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, Optional, TypeVar
from uuid import uuid4

from backend.workspace.models import (
    AgentOutcomeReport,
    ApprovalRequest,
    ProjectTask,
    TaskState,
    WorkspaceEvent,
)

from .base import AgentDescriptor, AgentEventHandler, AgentReportHandler


TaskT = TypeVar("TaskT")


class ServiceAgentAdapter(Generic[TaskT]):
    """Expose any asynchronous agent service through ANISHA's standard I/O.

    A specialist only supplies a task mapper. The common event/report mapping,
    assignment check, cancellation, and descriptor behavior live here.
    """

    def __init__(
        self,
        *,
        service: Any,
        descriptor: AgentDescriptor,
        task_factory: Callable[[ProjectTask], TaskT],
    ) -> None:
        self._service = service
        self.descriptor = descriptor
        self.agent_id = descriptor.agent_id
        self._task_factory = task_factory

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

        async def forward_event(event: Any) -> None:
            if on_event is None:
                return
            await on_event(
                WorkspaceEvent(
                    event_id=f"event-{uuid4()}",
                    project_id=task.project_id,
                    task_id=event.task_id,
                    actor=self.agent_id,
                    state=TaskState(_enum_value(event.status)),
                    message=event.message,
                    created_at=event.created_at,
                )
            )

        async def forward_report(report: Any) -> None:
            if on_report is None:
                return
            verification = report.verification
            if not isinstance(verification, str):
                verification = verification.result
            approval_request = None
            report_approval = getattr(report, "approval_request", None)
            if report_approval is not None:
                approval_request = ApprovalRequest.model_validate(
                    report_approval.model_dump(mode="python")
                )
            await on_report(
                AgentOutcomeReport(
                    report_id=report.report_id,
                    project_id=report.project_id,
                    task_id=report.task_id,
                    agent_id=self.agent_id,
                    outcome=TaskState(_enum_value(report.outcome)),
                    summary=report.summary,
                    artefact_refs=list(report.artefacts_created),
                    verification=verification,
                    unresolved_issues=list(report.unresolved_issues),
                    recommended_next_action=report.recommended_next_action,
                    approval_request=approval_request,
                    created_at=report.completed_at,
                )
            )

        await self._service.submit(
            self._task_factory(task),
            on_event=forward_event,
            on_report=forward_report,
        )

    async def cancel(self, task_id: str) -> None:
        await self._service.cancel(task_id)

    async def approve(self, task_id: str, approval_id: str) -> None:
        approve = getattr(self._service, "approve", None)
        if approve is None:
            raise ValueError(f"{self.agent_id} does not support approval actions")
        await approve(task_id, approval_id)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))
