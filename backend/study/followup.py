"""Selected-agent voice requests over released, fixed Condition C facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from backend.study.runner import TrialSnapshot
from backend.voice.intent import ExtractedIntent, Intent


class FollowupReview(BaseModel):
    transcription_id: str
    reviewed_transcript: str | None = None


class FollowupResult(BaseModel):
    status: Literal["ready", "needs_clarification"]
    request_kind: Literal["status", "evidence", "action"] | None = None
    agent_id: str
    message: str
    record_ids: list[str] = Field(default_factory=list)
    suggested_choice_id: str | None = None
    revealed_event_id: str | None = None


@dataclass(frozen=True)
class Resolution:
    result: FollowupResult
    revealed_event_id: str | None = None


def _result(agent_id: str, status: str, message: str, **kwargs) -> Resolution:
    return Resolution(FollowupResult(agent_id=agent_id, status=status, message=message, **kwargs))


def resolve_followup(intent: ExtractedIntent, state: TrialSnapshot, agent_id: str) -> Resolution:
    """Use ANISHA's extracted intent, but only the selected agent's released data."""

    name = "Coding" if agent_id == "coding_agent" else "Writing"
    other_name = "Writing" if agent_id == "coding_agent" else "Coding"
    if intent.target_agent is not None and intent.target_agent != agent_id:
        return _result(agent_id, "needs_clarification",
                       f"You are viewing {name}. Switch to {other_name} to ask about that agent.")
    if intent.parameters.clarification_needed:
        return _result(agent_id, "needs_clarification",
                       intent.parameters.clarification_question or f"What would you like to know about {name}?")
    task = next((item for item in state.tasks if item.agent_id == agent_id), None)
    if task is None:
        return _result(agent_id, "needs_clarification", f"{name} has not been briefed yet.")
    latest = next((event for event in reversed(state.events) if event.task_id == task.task_id), None)

    if intent.intent == Intent.QUERY_STATUS:
        detail = f" Latest update: {latest.text}" if latest else " No update has been released yet."
        return Resolution(
            FollowupResult(status="ready", request_kind="status", agent_id=agent_id,
                           message=f"{name} is {task.state.replace('_', ' ')}.{detail}"),
            revealed_event_id=latest.event_id if latest else None,
        )

    if intent.intent == Intent.REQUEST_DETAILS:
        records = [record for record in state.records if (
            record.kind == "instruction"
            or record.kind == "source" and (
                agent_id == "coding_agent" or not record.record_id.endswith("-S3")
            )
            or agent_id == "coding_agent" and record.kind == "code_artifact"
            or agent_id == "writing_agent" and record.kind in {"draft_preview", "report_excerpt", "markdown_artifact"}
        )]
        history = intent.parameters.requested_information and intent.parameters.requested_information.value == "history"
        message = latest.text if history and latest else (
            "No activity has been released yet." if history else "Open a source below to inspect its text."
        )
        return Resolution(
            FollowupResult(status="ready", request_kind="evidence", agent_id=agent_id,
                           message=message, record_ids=[record.record_id for record in records]),
            revealed_event_id=latest.event_id if history and latest else None,
        )

    if intent.intent in {Intent.APPROVE_ACTION, Intent.CORRECT_TASK}:
        options = state.action_options or state.correction_options
        gate = next((event for event in reversed(state.events)
                     if options and event.event_id == options[0].gate_event_id), None)
        if not options or gate is None or gate.task_id != task.task_id:
            return _result(agent_id, "ready", "There is no pending decision for this agent. Nothing was submitted.",
                           request_kind="action")
        suggested = intent.parameters.action_id
        if suggested not in {option.choice_id for option in options}:
            suggested = None
        return Resolution(
            FollowupResult(status="ready", request_kind="action", agent_id=agent_id,
                           message="Review the pending decision and save your choice on screen. Speaking does not submit it.",
                           suggested_choice_id=suggested),
            revealed_event_id=gate.event_id,
        )

    if intent.intent in {Intent.DELEGATE_TASK, Intent.PAUSE_TASK, Intent.CANCEL_TASK}:
        return _result(agent_id, "ready",
                       "This trial follows a fixed task sequence. No task was changed. Ask about its status or review the current decision on screen.",
                       request_kind="action")

    return _result(agent_id, "needs_clarification",
                   f"I can answer questions about {name}, show its released evidence, or help you review its pending decision.")
