"""Direct Coding and Writing task lifecycle using ANISHA's unchanged agents."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Literal
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from backend.agents.base import AgentExecutor
from backend.agents.coding_adapter import CODING_AGENT, CodingAgentAdapter
from backend.agents.service import CodingAgentService
from backend.agents.writing_adapter import WRITING_AGENT, WritingAgentAdapter
from backend.agents.writing_service import WritingAgentService
from backend.llm.factory import LLMSettings, create_llm_client
from backend.memory.models import MemoryQuery
from backend.memory.service import ContextMemoryService
from backend.memory.store import InMemoryContextStore, UnknownProjectContextError
from backend.workspace.models import AgentOutcomeReport, ProjectTask, TaskState, WorkspaceEvent
from backend.workspace.store import InMemoryWorkspace, UnknownTaskError


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path(__file__).resolve().parent / ".env"
BUILT_IN_PROJECT_WORKSPACES = {"red-wine-quality": "ml_project"}
BUILT_IN_PROJECT_READ_ONLY_PATHS = {
    "red-wine-quality": ("README.md", "winequality-red.csv", "project_context.json"),
}
DESCRIPTORS = {"coding_agent": CODING_AGENT, "writing_agent": WRITING_AGENT}


class AgentTaskProposal(BaseModel):
    """The same task fields the ANISHA orchestrator forwards to an agent."""

    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    goal: str = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    source_paths: list[str] = Field(default_factory=list)
    context: str | None = None
    target_path: str | None = None
    target_mode: Literal["create", "replace", "append"] = "replace"
    finish_after_first_section: bool = False


class TaskDetails(BaseModel):
    task: ProjectTask
    events: list[WorkspaceEvent]
    reports: list[AgentOutcomeReport]


class DirectAgentRuntime:
    def __init__(
        self,
        *,
        workspace: InMemoryWorkspace | None = None,
        context_memory: ContextMemoryService | None = None,
        agent_factory: Callable[[str], AgentExecutor] | None = None,
    ) -> None:
        self.workspace = workspace or InMemoryWorkspace()
        self.context_memory = context_memory
        self.agent_factory = agent_factory or build_agent
        self._agents: dict[str, AgentExecutor] = {}

    def _agent(self, agent_id: str) -> AgentExecutor:
        if agent_id not in DESCRIPTORS:
            raise ValueError("Unknown agent")
        if agent_id not in self._agents:
            self._agents[agent_id] = self.agent_factory(agent_id)
        return self._agents[agent_id]

    def propose(self, agent_id: str, project_id: str, proposal: AgentTaskProposal) -> TaskDetails:
        if agent_id not in DESCRIPTORS:
            raise ValueError("Unknown agent")
        if project_id.startswith("study-"):
            raise ValueError("Measured study trials cannot execute live agents")
        task = ProjectTask(
            project_id=project_id,
            task_id=proposal.task_id or f"task-{uuid4()}",
            goal=proposal.goal,
            assigned_agent=agent_id,
            constraints=proposal.constraints,
            source_paths=proposal.source_paths,
            context=proposal.context,
            target_path=proposal.target_path,
            target_mode=proposal.target_mode,
            finish_after_first_section=proposal.finish_after_first_section,
        )
        if self.context_memory is not None:
            query_parts = [task.goal, *task.constraints, *task.source_paths]
            if task.context is not None:
                query_parts.append(task.context)
            try:
                bundle = self.context_memory.retrieve(MemoryQuery(
                    project_id=project_id,
                    actor_id="orchestrator",
                    task_id=task.task_id,
                    agent_id=agent_id,
                    query="\n".join(query_parts),
                ))
                task = task.model_copy(update={"context_bundle": bundle}, deep=True)
            except UnknownProjectContextError:
                pass
        self.workspace.add_task(task)
        self._event(task, TaskState.PROPOSED, "Task proposed for direct agent review")
        return self.details(agent_id, project_id, task.task_id)

    async def confirm(self, agent_id: str, project_id: str, task_id: str) -> TaskDetails:
        task = self._task(agent_id, project_id, task_id)
        if task.state != TaskState.PROPOSED:
            raise ValueError(f"task {task_id} cannot be confirmed from {task.state.value}")
        agent = self._agent(agent_id)
        self._event(task, TaskState.QUEUED, "Task confirmed and queued")

        async def on_event(event: WorkspaceEvent) -> None:
            self.workspace.append_event(event)

        async def on_report(report: AgentOutcomeReport) -> None:
            self.workspace.append_report(report)

        try:
            await agent.submit(self.workspace.get_task(project_id, task_id),
                               on_event=on_event, on_report=on_report)
        except Exception:
            self._event(task, TaskState.FAILED, "Agent dispatch failed")
            raise
        return self.details(agent_id, project_id, task_id)

    def details(self, agent_id: str, project_id: str, task_id: str) -> TaskDetails:
        task = self._task(agent_id, project_id, task_id)
        return TaskDetails(
            task=task,
            events=self.workspace.list_events(project_id, task_id),
            reports=self.workspace.list_reports(project_id, task_id),
        )

    def list_tasks(self, agent_id: str, project_id: str) -> list[ProjectTask]:
        if agent_id not in DESCRIPTORS:
            raise ValueError("Unknown agent")
        return [task for task in self.workspace.list_tasks(project_id)
                if task.assigned_agent == agent_id]

    async def cancel(self, agent_id: str, project_id: str, task_id: str) -> TaskDetails:
        task = self._task(agent_id, project_id, task_id)
        if task.state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}:
            raise ValueError(f"task {task_id} cannot be cancelled from {task.state.value}")
        if task.state != TaskState.PROPOSED:
            await self._agent(agent_id).cancel(task_id)
        current = self.workspace.get_task(project_id, task_id)
        if current.state != TaskState.CANCELLED:
            self._event(task, TaskState.CANCELLED, "Task cancelled by user")
        return self.details(agent_id, project_id, task_id)

    async def approve(self, agent_id: str, project_id: str, task_id: str, approval_id: str) -> TaskDetails:
        task = self._task(agent_id, project_id, task_id)
        request = task.pending_approval
        if task.state != TaskState.AWAITING_APPROVAL or request is None:
            raise ValueError(f"task {task_id} has no action awaiting approval")
        if request.approval_id != approval_id:
            raise ValueError("approval identifier does not match")
        await self._agent(agent_id).approve(task_id, approval_id)
        return self.details(agent_id, project_id, task_id)

    def _task(self, agent_id: str, project_id: str, task_id: str) -> ProjectTask:
        if agent_id not in DESCRIPTORS:
            raise ValueError("Unknown agent")
        task = self.workspace.get_task(project_id, task_id)
        if task.assigned_agent != agent_id:
            raise UnknownTaskError(task_id)
        return task

    def _event(self, task: ProjectTask, state: TaskState, message: str) -> None:
        self.workspace.append_event(WorkspaceEvent(
            event_id=f"event-{uuid4()}",
            project_id=task.project_id,
            task_id=task.task_id,
            actor="direct_agent_api",
            state=state,
            message=message,
        ))


def _resolve_workspace_path(value: str, base: Path) -> Path:
    if not value.strip():
        raise RuntimeError("ANISHA_WORKSPACE_ROOT must not be empty")
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else base / path).resolve(strict=True)
    if not resolved.is_dir():
        raise RuntimeError(f"Workspace root is not a directory: {resolved}")
    return resolved


def workspace_roots() -> dict[str, Path]:
    roots = {
        project_id: _resolve_workspace_path(path, ROOT)
        for project_id, path in BUILT_IN_PROJECT_WORKSPACES.items()
    }
    raw = os.getenv("ANISHA_PROJECT_WORKSPACES", "").strip()
    if raw:
        configured = json.loads(raw)
        if not isinstance(configured, dict) or not all(
            isinstance(project_id, str) and isinstance(path, str)
            for project_id, path in configured.items()
        ):
            raise RuntimeError("ANISHA_PROJECT_WORKSPACES must map project identifiers to paths")
        for project_id, path in configured.items():
            resolved = _resolve_workspace_path(path, ROOT)
            if project_id in roots and resolved != roots[project_id]:
                raise RuntimeError(f"The built-in project {project_id!r} cannot be remapped")
            roots[project_id] = resolved
    return roots


def build_agent(agent_id: str) -> AgentExecutor:
    load_dotenv(ENV_FILE, override=False)
    default_root = _resolve_workspace_path(os.getenv("ANISHA_WORKSPACE_ROOT", "."), ROOT)
    roots = workspace_roots()
    timeout = int(os.getenv("ANISHA_LLM_TIMEOUT_SECONDS", "120"))
    if agent_id == "coding_agent":
        return CodingAgentAdapter(CodingAgentService(
            llm=create_llm_client(LLMSettings.for_agent("coding_agent")),
            workspace_root=default_root,
            workspace_roots=roots,
            workspace_read_only_paths=BUILT_IN_PROJECT_READ_ONLY_PATHS,
            max_steps=int(os.getenv("ANISHA_CODING_MAX_STEPS", "20")),
            llm_timeout_seconds=timeout,
        ))
    if agent_id == "writing_agent":
        return WritingAgentAdapter(WritingAgentService(
            llm=create_llm_client(LLMSettings.for_agent(
                "writing_agent",
                default_model="gpt-5.4-mini-2026-03-17",
                default_context_length=16_384,
                default_max_output_tokens=2_048,
                default_temperature=0.35,
            )),
            workspace_root=default_root,
            workspace_roots=roots,
            workspace_read_only_paths=BUILT_IN_PROJECT_READ_ONLY_PATHS,
            llm_timeout_seconds=timeout,
            max_calls=int(os.getenv("ANISHA_WRITING_MAX_CALLS", "15")),
        ))
    raise ValueError("Unknown agent")


def build_context_memory() -> ContextMemoryService:
    load_dotenv(ENV_FILE, override=False)
    service = ContextMemoryService(InMemoryContextStore())
    raw = os.getenv("ANISHA_CONTEXT_MEMORY_SEEDS", "ml_project/project_context.json")
    for configured in (value.strip() for value in raw.split(",")):
        if not configured:
            continue
        path = Path(configured).expanduser()
        resolved = (path if path.is_absolute() else ROOT / path).resolve(strict=True)
        if not resolved.is_file():
            raise RuntimeError(f"Context-memory seed is not a file: {resolved}")
        service.load_seed_file(resolved)
    return service


_runtime: DirectAgentRuntime | None = None


def get_live_runtime() -> DirectAgentRuntime:
    global _runtime
    if _runtime is None:
        _runtime = DirectAgentRuntime(context_memory=build_context_memory())
    return _runtime
