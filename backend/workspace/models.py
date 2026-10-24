"""Small, agent-neutral records stored in the Shared Project Workspace."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.memory.models import ContextBundle


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskState(str, Enum):
    PROPOSED = "proposed"
    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETED = "completed"
    AWAITING_APPROVAL = "awaiting_approval"
    CANCELLED = "cancelled"


class ClarificationState(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"


class ClarificationRecord(BaseModel):
    """Project-scoped context retained between voice clarification turns."""

    model_config = ConfigDict(extra="forbid")

    clarification_id: str
    project_id: str
    original_text: str
    current_intent: dict[str, Any]
    question: str
    context_task_id: Optional[str] = None
    replies: list[str] = Field(default_factory=list)
    state: ClarificationState = ClarificationState.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("clarification_id", "project_id", "original_text", "question")
    @classmethod
    def require_clarification_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("context_task_id")
    @classmethod
    def normalize_context_task_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("replies")
    @classmethod
    def normalize_replies(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class ApprovalRequest(BaseModel):
    """Structured consequential action awaiting a click from the user."""

    model_config = ConfigDict(extra="forbid")

    approval_id: str
    kind: Literal["python_environment"]
    title: str
    description: str
    environment_path: str
    packages: list[str] = Field(min_length=1)
    commands: list[str] = Field(min_length=1)
    requested_at: datetime = Field(default_factory=utc_now)

    @field_validator("approval_id", "title", "description", "environment_path")
    @classmethod
    def require_approval_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized


class TaskAttention(BaseModel):
    """Small orchestrator-owned hint for glanceable UI arbitration.

    This is advisory presentation metadata. It never grants permission and it
    cannot override the task lifecycle or approval boundary.
    """

    model_config = ConfigDict(extra="forbid")

    priority: Literal["ambient", "inform", "review", "blocked", "needs_decision"]
    score: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=240)
    headline: Optional[str] = Field(default=None, min_length=1, max_length=72)
    detail: Optional[str] = Field(default=None, min_length=1, max_length=120)
    interrupt: bool = False

    @field_validator("reason", "headline", "detail")
    @classmethod
    def normalize_presentation_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("presentation text must not be empty")
        return normalized


class ProjectTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    task_id: str
    goal: str
    assigned_agent: str
    constraints: list[str] = Field(default_factory=list)
    source_paths: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    context: Optional[str] = None
    context_bundle: Optional[ContextBundle] = None
    target_path: Optional[str] = None
    target_mode: Literal["create", "replace", "append"] = "replace"
    finish_after_first_section: bool = False
    pending_approval: Optional[ApprovalRequest] = None
    attention: Optional[TaskAttention] = None
    state: TaskState = TaskState.PROPOSED
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("project_id", "task_id", "goal", "assigned_agent")
    @classmethod
    def require_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("context", "target_path")
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("constraints", "source_paths", "depends_on")
    @classmethod
    def normalize_lists(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class WorkspaceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    project_id: str
    task_id: str
    actor: str
    state: TaskState
    message: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("event_id", "project_id", "task_id", "actor", "message")
    @classmethod
    def require_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized


class AgentOutcomeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    project_id: str
    task_id: str
    agent_id: str
    outcome: TaskState
    summary: str
    artefact_refs: list[str] = Field(default_factory=list)
    verification: str = "not_run"
    unresolved_issues: list[str] = Field(default_factory=list)
    recommended_next_action: str
    approval_request: Optional[ApprovalRequest] = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "report_id",
        "project_id",
        "task_id",
        "agent_id",
        "summary",
        "verification",
        "recommended_next_action",
    )
    @classmethod
    def require_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("artefact_refs", "unresolved_issues")
    @classmethod
    def normalize_lists(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]
