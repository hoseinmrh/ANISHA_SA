"""In-memory implementation of the Shared Project Workspace."""

from __future__ import annotations

from .models import (
    AgentOutcomeReport,
    ClarificationRecord,
    ProjectTask,
    TaskAttention,
    TaskState,
    WorkspaceEvent,
    utc_now,
)


class DuplicateRecordError(ValueError):
    pass


class UnknownTaskError(KeyError):
    pass


class UnknownClarificationError(KeyError):
    pass


class InMemoryWorkspace:
    """Stores task state, append-only events, and append-only reports."""

    def __init__(self) -> None:
        self._tasks: dict[tuple[str, str], ProjectTask] = {}
        self._events: list[WorkspaceEvent] = []
        self._reports: list[AgentOutcomeReport] = []
        self._event_ids: set[str] = set()
        self._report_ids: set[str] = set()
        self._clarifications: dict[tuple[str, str], ClarificationRecord] = {}

    def add_clarification(
        self, clarification: ClarificationRecord
    ) -> ClarificationRecord:
        key = self._clarification_key(
            clarification.project_id, clarification.clarification_id
        )
        if key in self._clarifications:
            raise DuplicateRecordError(clarification.clarification_id)
        self._clarifications[key] = clarification.model_copy(deep=True)
        return self.get_clarification(
            clarification.project_id, clarification.clarification_id
        )

    def get_clarification(
        self, project_id: str, clarification_id: str
    ) -> ClarificationRecord:
        clarification = self._clarifications.get(
            self._clarification_key(project_id, clarification_id)
        )
        if clarification is None:
            raise UnknownClarificationError(clarification_id)
        return clarification.model_copy(deep=True)

    def update_clarification(
        self, clarification: ClarificationRecord
    ) -> ClarificationRecord:
        key = self._clarification_key(
            clarification.project_id, clarification.clarification_id
        )
        if key not in self._clarifications:
            raise UnknownClarificationError(clarification.clarification_id)
        stored = clarification.model_copy(
            deep=True, update={"updated_at": utc_now()}
        )
        self._clarifications[key] = stored
        return stored.model_copy(deep=True)

    def add_task(self, task: ProjectTask) -> ProjectTask:
        key = self._task_key(task.project_id, task.task_id)
        if key in self._tasks:
            raise DuplicateRecordError(task.task_id)
        self._tasks[key] = task.model_copy(deep=True)
        return self.get_task(task.project_id, task.task_id)

    def get_task(self, project_id: str, task_id: str) -> ProjectTask:
        task = self._tasks.get(self._task_key(project_id, task_id))
        if task is None:
            raise UnknownTaskError(task_id)
        return task.model_copy(deep=True)

    def list_tasks(self, project_id: str) -> list[ProjectTask]:
        return [
            task.model_copy(deep=True)
            for (stored_project_id, _), task in self._tasks.items()
            if stored_project_id == project_id
        ]

    def append_event(self, event: WorkspaceEvent) -> WorkspaceEvent:
        if event.event_id in self._event_ids:
            raise DuplicateRecordError(event.event_id)
        task = self._require_matching_task(event.project_id, event.task_id)
        stored_event = event.model_copy(deep=True)
        self._events.append(stored_event)
        self._event_ids.add(event.event_id)
        task.state = event.state
        if event.state != TaskState.AWAITING_APPROVAL:
            task.pending_approval = None
        task.updated_at = utc_now()
        return stored_event.model_copy(deep=True)

    def list_events(self, project_id: str, task_id: str) -> list[WorkspaceEvent]:
        self._require_matching_task(project_id, task_id)
        return [
            event.model_copy(deep=True)
            for event in self._events
            if event.project_id == project_id and event.task_id == task_id
        ]

    def append_report(self, report: AgentOutcomeReport) -> AgentOutcomeReport:
        if report.report_id in self._report_ids:
            raise DuplicateRecordError(report.report_id)
        task = self._require_matching_task(report.project_id, report.task_id)
        if task.assigned_agent != report.agent_id:
            raise ValueError("report agent does not match the task assignment")
        stored_report = report.model_copy(deep=True)
        self._reports.append(stored_report)
        self._report_ids.add(report.report_id)
        task.state = report.outcome
        task.pending_approval = report.approval_request
        task.updated_at = utc_now()
        return stored_report.model_copy(deep=True)

    def update_attention(
        self, project_id: str, task_id: str, attention: TaskAttention
    ) -> ProjectTask:
        task = self._require_matching_task(project_id, task_id)
        task.attention = attention.model_copy(deep=True)
        task.updated_at = utc_now()
        return task.model_copy(deep=True)

    def clear_attention(self, project_id: str, task_id: str) -> ProjectTask:
        task = self._require_matching_task(project_id, task_id)
        task.attention = None
        task.updated_at = utc_now()
        return task.model_copy(deep=True)

    def clear_pending_approval(self, project_id: str, task_id: str) -> ProjectTask:
        task = self._require_matching_task(project_id, task_id)
        task.pending_approval = None
        task.updated_at = utc_now()
        return task.model_copy(deep=True)

    def list_reports(
        self, project_id: str, task_id: str
    ) -> list[AgentOutcomeReport]:
        self._require_matching_task(project_id, task_id)
        return [
            report.model_copy(deep=True)
            for report in self._reports
            if report.project_id == project_id and report.task_id == task_id
        ]

    def _require_matching_task(self, project_id: str, task_id: str) -> ProjectTask:
        task = self._tasks.get(self._task_key(project_id, task_id))
        if task is None:
            raise UnknownTaskError(task_id)
        return task

    @staticmethod
    def _task_key(project_id: str, task_id: str) -> tuple[str, str]:
        return project_id, task_id

    @staticmethod
    def _clarification_key(
        project_id: str, clarification_id: str
    ) -> tuple[str, str]:
        return project_id, clarification_id
