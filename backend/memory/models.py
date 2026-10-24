"""Provider-neutral contracts for project orientation and contextual recall."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must not be empty")
    return normalized


def _normalize_list(values: list[str]) -> list[str]:
    return [value.strip() for value in values if value.strip()]


class MemoryKind(str, Enum):
    FACT = "fact"
    EPISODE = "episode"
    PREFERENCE = "preference"
    LESSON = "lesson"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"
    DELETED = "deleted"


class SourceAuthority(str, Enum):
    USER_CONFIRMED = "user_confirmed"
    WORKSPACE_RECORD = "workspace_record"
    AGENT_REPORT = "agent_report"
    EXTERNAL_SOURCE = "external_source"
    GENERATED = "generated"


class Sensitivity(str, Enum):
    PROJECT = "project"
    PERSONAL = "personal"
    RESTRICTED = "restricted"


class MemoryProposalState(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class MemoryConflictState(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"


class MemorySourceRef(BaseModel):
    """Stable provenance for a memory statement."""

    model_config = ConfigDict(extra="forbid")

    uri: str
    source_type: str
    authority: SourceAuthority
    version: Optional[str] = None

    @field_validator("uri", "source_type")
    @classmethod
    def require_text(cls, value: str) -> str:
        return _normalize_required(value)

    @field_validator("version")
    @classmethod
    def normalize_version(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class MemoryRecord(BaseModel):
    """One source-backed, project-scoped long-term memory."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    project_id: str
    kind: MemoryKind
    content: str
    source_refs: list[MemorySourceRef] = Field(min_length=1)
    created_by: str
    captured_at: datetime = Field(default_factory=utc_now)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    supersedes: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    sensitivity: Sensitivity = Sensitivity.PROJECT
    allowed_agents: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("memory_id", "project_id", "content", "created_by")
    @classmethod
    def require_text(cls, value: str) -> str:
        return _normalize_required(value)

    @field_validator("supersedes", "allowed_agents", "tags")
    @classmethod
    def normalize_lists(cls, values: list[str]) -> list[str]:
        return _normalize_list(values)

    @model_validator(mode="after")
    def validate_validity_window(self) -> "MemoryRecord":
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_to < self.valid_from
        ):
            raise ValueError("valid_to must not be earlier than valid_from")
        if self.memory_id in self.supersedes:
            raise ValueError("a memory cannot supersede itself")
        return self


class ProjectBrief(BaseModel):
    """Small regenerable orientation snapshot shown before deeper retrieval."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    title: str
    purpose: str
    current_focus: str
    recent_progress: list[str] = Field(default_factory=list)
    key_constraints: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    memory_refs: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)
    as_of_event: Optional[str] = None

    @field_validator("project_id", "title", "purpose", "current_focus")
    @classmethod
    def require_text(cls, value: str) -> str:
        return _normalize_required(value)

    @field_validator(
        "recent_progress",
        "key_constraints",
        "open_questions",
        "next_actions",
        "memory_refs",
    )
    @classmethod
    def normalize_lists(cls, values: list[str]) -> list[str]:
        return _normalize_list(values)

    @field_validator("as_of_event")
    @classmethod
    def normalize_as_of_event(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class MemoryQuery(BaseModel):
    """A bounded retrieval request made by the user, orchestrator, or an agent."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    actor_id: str
    query: str = ""
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    kinds: list[MemoryKind] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)
    max_characters: int = Field(default=4_000, ge=256, le=20_000)

    @field_validator("project_id", "actor_id")
    @classmethod
    def require_text(cls, value: str) -> str:
        return _normalize_required(value)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        return value.strip()

    @field_validator("task_id", "agent_id")
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None


class MemoryCandidate(BaseModel):
    """Untrusted candidate that must be reviewed before becoming a record."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    kind: MemoryKind
    content: str
    source_refs: list[MemorySourceRef] = Field(min_length=1)
    proposed_by: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    sensitivity: Sensitivity = Sensitivity.PROJECT
    allowed_agents: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("project_id", "content", "proposed_by")
    @classmethod
    def require_text(cls, value: str) -> str:
        return _normalize_required(value)

    @field_validator("allowed_agents", "tags")
    @classmethod
    def normalize_lists(cls, values: list[str]) -> list[str]:
        return _normalize_list(values)


class MemoryProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    candidate: MemoryCandidate
    state: MemoryProposalState = MemoryProposalState.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    reviewed_by: Optional[str] = None
    review_note: Optional[str] = None

    @field_validator("proposal_id")
    @classmethod
    def require_text(cls, value: str) -> str:
        return _normalize_required(value)

    @field_validator("reviewed_by", "review_note")
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def validate_review(self) -> "MemoryProposal":
        if self.state != MemoryProposalState.PENDING and self.reviewed_by is None:
            raise ValueError("reviewed_by is required for a decided proposal")
        return self


class DecisionConfirmationRequest(BaseModel):
    """One project decision explicitly confirmed by the user."""

    model_config = ConfigDict(extra="forbid")

    decision: str = Field(min_length=1, max_length=2_000)
    resolves_questions: list[str] = Field(default_factory=list, max_length=20)
    source_paths: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("decision")
    @classmethod
    def require_decision(cls, value: str) -> str:
        return _normalize_required(value)

    @field_validator("resolves_questions", "source_paths")
    @classmethod
    def normalize_confirmation_lists(cls, values: list[str]) -> list[str]:
        return _normalize_list(values)


class DecisionConfirmationResult(BaseModel):
    """The durable memory and updated orientation produced by confirmation."""

    model_config = ConfigDict(extra="forbid")

    memory: MemoryRecord
    brief: ProjectBrief


class MemoryConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conflict_id: str
    project_id: str
    memory_ids: list[str] = Field(min_length=2)
    summary: str
    state: MemoryConflictState = MemoryConflictState.OPEN
    resolved_by: Optional[str] = None
    resolution: Optional[str] = None

    @field_validator("conflict_id", "project_id", "summary")
    @classmethod
    def require_text(cls, value: str) -> str:
        return _normalize_required(value)

    @field_validator("memory_ids")
    @classmethod
    def normalize_memory_ids(cls, values: list[str]) -> list[str]:
        normalized = _normalize_list(values)
        if len(set(normalized)) != len(normalized):
            raise ValueError("memory_ids must be unique")
        return normalized

    @field_validator("resolved_by", "resolution")
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def validate_resolution(self) -> "MemoryConflict":
        if self.state == MemoryConflictState.RESOLVED and (
            self.resolved_by is None or self.resolution is None
        ):
            raise ValueError(
                "resolved conflicts require resolved_by and a resolution"
            )
        return self


class ContextBundle(BaseModel):
    """Bounded context supplied to one user interaction or agent run."""

    model_config = ConfigDict(extra="forbid")

    query: MemoryQuery
    brief: ProjectBrief
    memories: list[MemoryRecord] = Field(default_factory=list)
    conflicts: list[MemoryConflict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    truncated: bool = False
    generated_at: datetime = Field(default_factory=utc_now)

    @field_validator("warnings")
    @classmethod
    def normalize_warnings(cls, values: list[str]) -> list[str]:
        return _normalize_list(values)

    @model_validator(mode="after")
    def validate_project_scope(self) -> "ContextBundle":
        project_ids = {
            self.query.project_id,
            self.brief.project_id,
            *(memory.project_id for memory in self.memories),
            *(conflict.project_id for conflict in self.conflicts),
        }
        if len(project_ids) != 1:
            raise ValueError("all context bundle records must share one project_id")
        if len(self.memories) > self.query.limit:
            raise ValueError("context bundle exceeds the query memory limit")
        return self
