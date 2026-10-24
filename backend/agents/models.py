"""Typed coding-agent task, state, and report contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.memory.models import ContextBundle


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETED = "completed"
    AWAITING_APPROVAL = "awaiting_approval"
    CANCELLED = "cancelled"


class CodingTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    task_id: str
    goal: str
    constraints: list[str] = Field(default_factory=list)
    context: Optional[str] = None
    context_bundle: Optional[ContextBundle] = None

    @field_validator("project_id", "task_id", "goal")
    @classmethod
    def require_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("constraints")
    @classmethod
    def normalize_constraints(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class VerificationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commands_run: list[str] = Field(default_factory=list)
    result: str = "not_run"
    limitations: list[str] = Field(default_factory=list)


class PythonEnvironmentApproval(BaseModel):
    """A bounded, user-visible request to prepare a project Python runtime."""

    model_config = ConfigDict(extra="forbid")

    approval_id: str
    kind: Literal["python_environment"] = "python_environment"
    title: str = "Create a project Python environment?"
    description: str
    environment_path: Literal[".venv"] = ".venv"
    packages: list[str] = Field(min_length=1)
    commands: list[str] = Field(min_length=1)
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("approval_id", "title", "description")
    @classmethod
    def require_approval_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized


class CodingAgentFinal(BaseModel):
    """Schema the model must return after it stops using tools."""

    model_config = ConfigDict(extra="forbid")

    outcome: Literal["completed", "blocked", "failed", "awaiting_approval"]
    summary: str
    work_completed: list[str] = Field(default_factory=list)
    verification_result: str = "not_run"
    limitations: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    recommended_next_action: str


class CodingAgentTurn(BaseModel):
    """One provider-neutral model decision in the coding-agent loop."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tool", "final"]
    tool_name: Optional[str] = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    final: Optional[CodingAgentFinal] = None


class AgentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    project_id: str
    task_id: str
    agent_id: Literal["coding_agent"] = "coding_agent"
    model: str
    outcome: TaskStatus
    summary: str
    work_completed: list[str] = Field(default_factory=list)
    artefacts_created: list[str] = Field(default_factory=list)
    verification: VerificationSummary = Field(default_factory=VerificationSummary)
    unresolved_issues: list[str] = Field(default_factory=list)
    recommended_next_action: str
    approval_request: Optional[PythonEnvironmentApproval] = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: TaskStatus
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CodingTaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task: CodingTask
    status: TaskStatus = TaskStatus.QUEUED
    events: list[AgentEvent] = Field(default_factory=list)
    report: Optional[AgentReport] = None
