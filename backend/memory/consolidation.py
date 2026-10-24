"""Deterministic post-task projection into project context and README files."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from backend.workspace.models import AgentOutcomeReport, ProjectTask, TaskState

from .models import (
    DecisionConfirmationRequest,
    DecisionConfirmationResult,
    MemoryConflict,
    MemoryKind,
    MemoryRecord,
    MemorySourceRef,
    ProjectBrief,
    Sensitivity,
    SourceAuthority,
    utc_now,
)
from .service import ContextMemoryService
from .store import UnknownMemoryRecordError, UnknownProjectContextError


STATUS_START = "<!-- ANISHA:PROJECT-STATUS:START -->"
STATUS_END = "<!-- ANISHA:PROJECT-STATUS:END -->"
MAX_RECENT_PROGRESS = 8


class ProjectMemoryConsolidator:
    """Record completed work without treating generated proposals as decisions."""

    def __init__(
        self,
        *,
        context_memory: ContextMemoryService,
        project_roots: Mapping[str, Path | str],
    ) -> None:
        self.context_memory = context_memory
        self.project_roots = {
            project_id: Path(root).resolve(strict=True)
            for project_id, root in project_roots.items()
        }
        self._write_lock = threading.Lock()

    async def record_completion(
        self, task: ProjectTask, report: AgentOutcomeReport
    ) -> None:
        """Persist one completed task and refresh the live in-memory snapshot."""

        await asyncio.to_thread(self._record_completion, task, report)

    async def confirm_decision(
        self,
        project_id: str,
        confirmation: DecisionConfirmationRequest,
    ) -> DecisionConfirmationResult:
        """Persist exactly one decision explicitly confirmed by the user."""

        return await asyncio.to_thread(
            self._confirm_decision, project_id, confirmation
        )

    def _record_completion(
        self, task: ProjectTask, report: AgentOutcomeReport
    ) -> None:
        if report.outcome != TaskState.COMPLETED:
            return
        if (
            report.project_id != task.project_id
            or report.task_id != task.task_id
            or report.agent_id != task.assigned_agent
        ):
            raise ValueError("task and report identity do not match")

        root = self.project_roots.get(task.project_id)
        if root is None:
            raise UnknownProjectContextError(task.project_id)
        context_path = root / "project_context.json"
        readme_path = root / "README.md"
        if not context_path.is_file() or not readme_path.is_file():
            raise ValueError(
                "project context consolidation requires project_context.json "
                "and README.md"
            )

        with self._write_lock:
            memory_id = _completion_memory_id(task, report)
            try:
                self.context_memory.store.get_memory(task.project_id, memory_id)
            except UnknownMemoryRecordError:
                pass
            else:
                return

            brief = self.context_memory.store.get_brief(task.project_id)
            memories = self.context_memory.store.list_memories(task.project_id)
            conflicts = self.context_memory.store.list_conflicts(task.project_id)
            artefacts = _verified_artefacts(root, report.artefact_refs)
            memory = _completion_memory(task, report, memory_id, artefacts, memories)
            updated_brief = _updated_brief(brief, task, report, memory, artefacts)

            context_content = _context_content(
                updated_brief, [*memories, memory], conflicts
            )
            readme_content = _updated_readme(
                readme_path.read_text(encoding="utf-8"),
                updated_brief,
                task,
                report,
                artefacts,
            )

            _replace_text_atomically(context_path, context_content)
            _replace_text_atomically(readme_path, readme_content)
            self.context_memory.store.add_memory_and_update_brief(
                memory, updated_brief
            )

    def _confirm_decision(
        self,
        project_id: str,
        confirmation: DecisionConfirmationRequest,
    ) -> DecisionConfirmationResult:
        root = self.project_roots.get(project_id)
        if root is None:
            raise UnknownProjectContextError(project_id)
        context_path = root / "project_context.json"
        readme_path = root / "README.md"
        if not context_path.is_file() or not readme_path.is_file():
            raise ValueError(
                "project decision confirmation requires project_context.json "
                "and README.md"
            )

        with self._write_lock:
            memory_id = _decision_memory_id(project_id, confirmation)
            try:
                existing = self.context_memory.store.get_memory(
                    project_id, memory_id
                )
            except UnknownMemoryRecordError:
                pass
            else:
                return DecisionConfirmationResult(
                    memory=existing,
                    brief=self.context_memory.store.get_brief(project_id),
                )

            brief = self.context_memory.store.get_brief(project_id)
            memories = self.context_memory.store.list_memories(project_id)
            conflicts = self.context_memory.store.list_conflicts(project_id)
            unknown_questions = sorted(
                set(confirmation.resolves_questions) - set(brief.open_questions)
            )
            if unknown_questions:
                raise ValueError(
                    "resolved questions must exactly match current open questions: "
                    + "; ".join(unknown_questions)
                )

            sources = _verified_artefacts(root, confirmation.source_paths)
            confirmed_at = utc_now()
            memory = _confirmed_decision_memory(
                project_id,
                memory_id,
                confirmation,
                confirmed_at,
                sources,
                memories,
            )
            updated_brief = _brief_after_decision(
                brief, confirmation, memory, confirmed_at
            )
            context_content = _context_content(
                updated_brief, [*memories, memory], conflicts
            )
            readme_content = _readme_after_decision(
                readme_path.read_text(encoding="utf-8"),
                updated_brief,
                confirmation,
                confirmed_at,
                sources,
            )

            _replace_text_atomically(context_path, context_content)
            _replace_text_atomically(readme_path, readme_content)
            self.context_memory.store.add_memory_and_update_brief(
                memory, updated_brief
            )
            return DecisionConfirmationResult(memory=memory, brief=updated_brief)


def _context_content(
    brief: ProjectBrief,
    memories: list[MemoryRecord],
    conflicts: list[MemoryConflict],
) -> str:
    payload = {
        "brief": brief.model_dump(mode="json"),
        "memories": [memory.model_dump(mode="json") for memory in memories],
    }
    if conflicts:
        payload["conflicts"] = [
            conflict.model_dump(mode="json") for conflict in conflicts
        ]
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _decision_memory_id(
    project_id: str, confirmation: DecisionConfirmationRequest
) -> str:
    identity = json.dumps(
        {
            "project_id": project_id,
            "decision": confirmation.decision.casefold(),
            "resolves_questions": sorted(confirmation.resolves_questions),
            "source_paths": sorted(confirmation.source_paths),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"memory-user-decision-{digest}"


def _confirmed_decision_memory(
    project_id: str,
    memory_id: str,
    confirmation: DecisionConfirmationRequest,
    confirmed_at: datetime,
    sources: list[tuple[str, str]],
    existing_memories: list[MemoryRecord],
) -> MemoryRecord:
    decision_uri = f"user://{project_id}/decisions/{memory_id}"
    source_refs = [
        MemorySourceRef(
            uri=decision_uri,
            source_type="explicit_user_confirmation",
            authority=SourceAuthority.USER_CONFIRMED,
            version=confirmed_at.isoformat(),
        )
    ]
    source_refs.extend(
        MemorySourceRef(
            uri=f"file://{path}",
            source_type="reviewed_project_file",
            authority=SourceAuthority.WORKSPACE_RECORD,
            version=f"sha256:{digest}",
        )
        for path, digest in sources
    )
    allowed_agents = sorted(
        {
            agent
            for memory in existing_memories
            for agent in memory.allowed_agents
        }
    )
    return MemoryRecord(
        memory_id=memory_id,
        project_id=project_id,
        kind=MemoryKind.FACT,
        content=confirmation.decision,
        source_refs=source_refs,
        created_by="user",
        captured_at=confirmed_at,
        importance=1.0,
        confidence=1.0,
        sensitivity=Sensitivity.PROJECT,
        allowed_agents=allowed_agents,
        tags=["decision", "user-confirmed"],
    )


def _brief_after_decision(
    brief: ProjectBrief,
    confirmation: DecisionConfirmationRequest,
    memory: MemoryRecord,
    confirmed_at: datetime,
) -> ProjectBrief:
    progress = f"User confirmed: {_compact(confirmation.decision)}"
    recent_progress = [
        *[item for item in brief.recent_progress if item != progress],
        progress,
    ][-MAX_RECENT_PROGRESS:]
    resolved = set(confirmation.resolves_questions)
    open_questions = [
        question for question in brief.open_questions if question not in resolved
    ]
    if open_questions:
        focus = "Review and resolve the remaining open project questions."
        next_actions = [focus]
    else:
        focus = "Use the confirmed project decisions for the next supervised task."
        next_actions = [
            "Propose the next supervised task using the confirmed project decisions."
        ]
    return brief.model_copy(
        update={
            "current_focus": focus,
            "recent_progress": recent_progress,
            "open_questions": open_questions,
            "next_actions": next_actions,
            "memory_refs": [*brief.memory_refs, memory.memory_id],
            "generated_at": confirmed_at,
            "as_of_event": memory.source_refs[0].uri,
        }
    )


def _completion_memory_id(
    task: ProjectTask, report: AgentOutcomeReport
) -> str:
    identity = f"{task.project_id}\0{task.task_id}\0{report.report_id}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"memory-task-completed-{digest}"


def _verified_artefacts(
    root: Path, artefact_refs: list[str]
) -> list[tuple[str, str]]:
    verified: list[tuple[str, str]] = []
    seen: set[str] = set()
    for reference in artefact_refs:
        relative = Path(reference)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"artefact path escapes the project workspace: {reference}")
        candidate = (root / relative).resolve(strict=True)
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"artefact is not a project file: {reference}")
        normalized = candidate.relative_to(root).as_posix()
        if normalized in seen:
            continue
        seen.add(normalized)
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        verified.append((normalized, digest))
    return verified


def _completion_memory(
    task: ProjectTask,
    report: AgentOutcomeReport,
    memory_id: str,
    artefacts: list[tuple[str, str]],
    existing_memories: list[MemoryRecord],
) -> MemoryRecord:
    agent_name = _agent_name(task.assigned_agent)
    artefact_text = (
        ", ".join(path for path, _ in artefacts)
        if artefacts
        else "no file artefacts"
    )
    source_refs = [
        MemorySourceRef(
            uri=(
                f"workspace://{task.project_id}/tasks/{task.task_id}/"
                f"reports/{report.report_id}"
            ),
            source_type="agent_outcome_report",
            authority=SourceAuthority.AGENT_REPORT,
            version=report.created_at.isoformat(),
        )
    ]
    source_refs.extend(
        MemorySourceRef(
            uri=f"file://{path}",
            source_type="agent_artefact",
            authority=SourceAuthority.AGENT_REPORT,
            version=f"sha256:{digest}",
        )
        for path, digest in artefacts
    )
    allowed_agents = sorted(
        {
            agent
            for memory in existing_memories
            for agent in memory.allowed_agents
        }
        | {task.assigned_agent}
    )
    return MemoryRecord(
        memory_id=memory_id,
        project_id=task.project_id,
        kind=MemoryKind.EPISODE,
        content=(
            f'{agent_name} completed the task "{_compact(task.goal)}". '
            f"Verified artefacts: {artefact_text}. Generated recommendations "
            "and decisions in those artefacts remain unconfirmed until the user "
            "explicitly reviews them."
        ),
        source_refs=source_refs,
        created_by="orchestrator",
        captured_at=report.created_at,
        importance=0.85,
        confidence=1.0,
        sensitivity=Sensitivity.PROJECT,
        allowed_agents=allowed_agents,
        tags=["task-completion", task.assigned_agent, "unconfirmed-output"],
    )


def _updated_brief(
    brief: ProjectBrief,
    task: ProjectTask,
    report: AgentOutcomeReport,
    memory: MemoryRecord,
    artefacts: list[tuple[str, str]],
) -> ProjectBrief:
    agent_name = _agent_name(task.assigned_agent)
    progress = (
        f'{agent_name} completed "{_compact(task.goal)}"'
        + (
            " and produced: " + ", ".join(path for path, _ in artefacts)
            if artefacts
            else ""
        )
        + "."
    )
    recent_progress = [
        *[item for item in brief.recent_progress if item != progress],
        progress,
    ][-MAX_RECENT_PROGRESS:]
    review_action = (
        "Review the completed task output and explicitly confirm or correct any "
        "proposed decisions before authorizing follow-on work."
    )
    next_actions = [review_action]
    recommended = _compact(report.recommended_next_action)
    if recommended and recommended.casefold() != review_action.casefold():
        next_actions.append(f"Agent recommendation (unconfirmed): {recommended}")

    return brief.model_copy(
        update={
            "current_focus": review_action,
            "recent_progress": recent_progress,
            "next_actions": next_actions,
            "memory_refs": [*brief.memory_refs, memory.memory_id],
            "generated_at": report.created_at,
            "as_of_event": (
                f"workspace://{task.project_id}/tasks/{task.task_id}/"
                f"reports/{report.report_id}"
            ),
        }
    )


def _updated_readme(
    content: str,
    brief: ProjectBrief,
    task: ProjectTask,
    report: AgentOutcomeReport,
    artefacts: list[tuple[str, str]],
) -> str:
    artefact_lines = (
        "\n".join(f"- `{path}`" for path, _ in artefacts)
        if artefacts
        else "- No file artefacts were reported."
    )
    progress_lines = "\n".join(f"- {item}" for item in brief.recent_progress)
    question_lines = (
        "\n".join(f"- {item}" for item in brief.open_questions)
        if brief.open_questions
        else "- None recorded."
    )
    action_lines = "\n".join(f"- {item}" for item in brief.next_actions)
    body = (
        f"{STATUS_START}\n"
        f"ANISHA refreshed this section after task `{task.task_id}` completed.\n\n"
        f"**Latest completed task:** {_markdown_inline(task.goal)}\n\n"
        f"**Agent:** {_agent_name(task.assigned_agent)}\n\n"
        f"**Recorded:** {report.created_at.isoformat()}\n\n"
        "### Verified artefacts\n\n"
        f"{artefact_lines}\n\n"
        "### Recent progress\n\n"
        f"{progress_lines}\n\n"
        "### Current focus\n\n"
        f"{brief.current_focus}\n\n"
        "### Open questions\n\n"
        f"{question_lines}\n\n"
        "### Next actions\n\n"
        f"{action_lines}\n\n"
        "Generated task outputs are recorded as evidence, not as user-confirmed "
        "decisions. Any proposed label, preprocessing, evaluation, or modeling "
        "choice remains pending until the user explicitly approves it.\n"
        f"{STATUS_END}"
    )
    return _replace_current_state(content, body)


def _readme_after_decision(
    content: str,
    brief: ProjectBrief,
    confirmation: DecisionConfirmationRequest,
    confirmed_at: datetime,
    sources: list[tuple[str, str]],
) -> str:
    source_lines = (
        "\n".join(f"- `{path}`" for path, _ in sources)
        if sources
        else "- Direct user confirmation; no project file was cited."
    )
    progress_lines = "\n".join(f"- {item}" for item in brief.recent_progress)
    question_lines = (
        "\n".join(f"- {item}" for item in brief.open_questions)
        if brief.open_questions
        else "- None recorded."
    )
    action_lines = "\n".join(f"- {item}" for item in brief.next_actions)
    body = (
        f"{STATUS_START}\n"
        "ANISHA refreshed this section after an explicit user confirmation.\n\n"
        "### Latest confirmed decision\n\n"
        f"{_markdown_inline(confirmation.decision)}\n\n"
        f"**Recorded:** {confirmed_at.isoformat()}\n\n"
        "### Reviewed sources\n\n"
        f"{source_lines}\n\n"
        "### Recent progress\n\n"
        f"{progress_lines}\n\n"
        "### Current focus\n\n"
        f"{brief.current_focus}\n\n"
        "### Open questions\n\n"
        f"{question_lines}\n\n"
        "### Next actions\n\n"
        f"{action_lines}\n\n"
        "Only the decision stated above was promoted to user-confirmed memory. "
        "Other generated recommendations remain unconfirmed.\n"
        f"{STATUS_END}"
    )
    return _replace_current_state(content, body)


def _replace_current_state(content: str, body: str) -> str:
    replacement = f"## Current state\n\n{body}"

    if STATUS_START in content and STATUS_END in content:
        start = content.index(STATUS_START)
        end = content.index(STATUS_END, start) + len(STATUS_END)
        return f"{content[:start]}{body}{content[end:]}".rstrip() + "\n"

    lines = content.splitlines()
    heading_index = next(
        (index for index, line in enumerate(lines) if line.strip() == "## Current state"),
        None,
    )
    if heading_index is None:
        return f"{content.rstrip()}\n\n{replacement}\n"
    next_heading = next(
        (
            index
            for index in range(heading_index + 1, len(lines))
            if lines[index].startswith("## ")
        ),
        len(lines),
    )
    updated_lines = [
        *lines[:heading_index],
        *replacement.splitlines(),
        *lines[next_heading:],
    ]
    return "\n".join(updated_lines).rstrip() + "\n"


def _replace_text_atomically(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.anisha-context-tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _agent_name(agent_id: str) -> str:
    return " ".join(word.capitalize() for word in agent_id.split("_"))


def _compact(value: str) -> str:
    return " ".join(value.split())


def _markdown_inline(value: str) -> str:
    return _compact(value).replace("`", "'")
