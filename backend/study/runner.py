"""Durable, side-effect-free trial state machine shared by both phone modes."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from typing import Callable
from uuid import uuid4

from pydantic import BaseModel, Field
from typing import Literal

from backend.study.materials import EventSpec, StudyMaterials, VariantStimuli
from backend.study.setup import StudySetupStore, TrialRecord, TrialSetupRequest


def server_time_ms() -> int:
    return time.time_ns() // 1_000_000


class BriefingConfirmation(BaseModel):
    agent_id: str
    reviewed_transcript: str = Field(min_length=1)
    goal: str | None = None


class GateAnswer(BaseModel):
    choice_id: str = ""
    reason: str = ""
    confidence: int | None = Field(default=None, ge=0, le=100)
    claim_text: str = ""


class TechnicalFailure(BaseModel):
    reason: str = Field(min_length=1)


class ParticipantInteraction(BaseModel):
    event_type: Literal["event_open", "source_open", "cue_delivered", "agent_switch", "ui_layer_open"]
    target_id: str = Field(min_length=1, max_length=120)
    ui_layer: str = Field(min_length=1, max_length=80)
    client_ts_ms: int | None = None


class InteractionReceipt(BaseModel):
    interaction_id: str
    server_ts_ms: int
    is_new: bool = True


class TaskRecord(BaseModel):
    task_id: str
    agent_id: str
    goal: str
    state: str
    depends_on: str | None
    outline_ready: bool
    verified_results_ready: bool


class ContentRecord(BaseModel):
    record_id: str
    kind: str
    title: str
    text: str


class DeliveredEvent(BaseModel):
    event_id: str
    task_id: str | None
    script_key: str
    text: str
    scheduled_at_ms: int
    occurred_at_ms: int
    cue_id: str | None


class ActionOption(BaseModel):
    gate_event_id: str
    choice_id: str
    label: str


class TrialSnapshot(BaseModel):
    trial_id: str
    interface_label: str
    participant_brief: str
    phase: str
    started_at_ms: int | None
    due_at_ms: int | None
    tasks: list[TaskRecord]
    records: list[ContentRecord]
    events: list[DeliveredEvent]
    action_options: list[ActionOption]
    correction_options: list[ActionOption] = []
    correction_for_event_id: str | None = None
    correction_previous_choice_label: str | None = None
    correction_previous_reason: str | None = None
    masked: bool = False
    confirmed_agents: list[str] = []


class ParticipantEvent(BaseModel):
    event_id: str
    task_id: str | None
    text: str
    cue_id: str | None
    occurred_at_ms: int


class ParticipantTrialView(BaseModel):
    trial_id: str
    interface_label: str
    participant_brief: str
    phase: str
    tasks: list[TaskRecord]
    records: list[ContentRecord]
    events: list[ParticipantEvent]
    action_options: list[ActionOption]
    correction_options: list[ActionOption]
    correction_for_event_id: str | None
    correction_previous_choice_label: str | None
    correction_previous_reason: str | None
    masked: bool
    confirmed_agents: list[str]


def participant_view(state: TrialSnapshot) -> ParticipantTrialView:
    return ParticipantTrialView(
        trial_id=state.trial_id,
        interface_label=state.interface_label,
        participant_brief=state.participant_brief,
        phase=state.phase,
        tasks=state.tasks,
        records=state.records,
        events=[
            ParticipantEvent(
                event_id=event.event_id,
                task_id=event.task_id,
                text=event.text,
                cue_id=event.cue_id,
                occurred_at_ms=event.occurred_at_ms,
            )
            for event in state.events
            if event.script_key != "probe_dependency"
        ],
        action_options=state.action_options,
        correction_options=state.correction_options,
        correction_for_event_id=state.correction_for_event_id,
        correction_previous_choice_label=state.correction_previous_choice_label,
        correction_previous_reason=state.correction_previous_reason,
        masked=state.masked,
        confirmed_agents=state.confirmed_agents,
    )


class StudyRunner:
    """Only changes SQLite study records; it has no agent, file-action, or network hooks."""

    def __init__(self, store: StudySetupStore, clock: Callable[[], int] = server_time_ms):
        self.store = store
        self.clock = clock

    @staticmethod
    def create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS study_trial_runtime (
                trial_id TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                started_at_ms INTEGER,
                anchor_ms INTEGER,
                next_index INTEGER NOT NULL DEFAULT 0,
                pause_phase TEXT,
                FOREIGN KEY (trial_id) REFERENCES study_trials(trial_id)
            );
            CREATE TABLE IF NOT EXISTS study_briefings (
                trial_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                reviewed_transcript TEXT NOT NULL,
                confirmed_at_ms INTEGER NOT NULL,
                PRIMARY KEY (trial_id, agent_id)
            );
            CREATE TABLE IF NOT EXISTS study_condition_c_goals (
                trial_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                PRIMARY KEY (trial_id, agent_id)
            );
            CREATE TABLE IF NOT EXISTS study_tasks (
                trial_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                state TEXT NOT NULL,
                depends_on TEXT,
                outline_ready INTEGER NOT NULL DEFAULT 0,
                verified_results_ready INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (trial_id, task_id)
            );
            CREATE TABLE IF NOT EXISTS study_content (
                trial_id TEXT NOT NULL,
                record_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                release_event_id TEXT,
                PRIMARY KEY (trial_id, record_id)
            );
            CREATE TABLE IF NOT EXISTS study_event_history (
                trial_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                task_id TEXT,
                script_key TEXT NOT NULL,
                text TEXT NOT NULL,
                scheduled_at_ms INTEGER NOT NULL,
                occurred_at_ms INTEGER NOT NULL,
                cue_id TEXT,
                PRIMARY KEY (trial_id, event_id)
            );
            CREATE TABLE IF NOT EXISTS study_action_options (
                trial_id TEXT NOT NULL,
                gate_event_id TEXT NOT NULL,
                choice_id TEXT NOT NULL,
                label TEXT NOT NULL,
                PRIMARY KEY (trial_id, gate_event_id, choice_id)
            );
            CREATE TABLE IF NOT EXISTS study_responses (
                response_id TEXT PRIMARY KEY,
                trial_id TEXT NOT NULL,
                gate_event_id TEXT NOT NULL,
                response_order INTEGER NOT NULL,
                choice_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                confidence INTEGER,
                claim_text TEXT NOT NULL,
                responded_at_ms INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                UNIQUE (trial_id, gate_event_id, response_order)
            );
            CREATE TABLE IF NOT EXISTS study_researcher_operations (
                operation_id TEXT PRIMARY KEY,
                trial_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                reason TEXT NOT NULL,
                occurred_at_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS study_interactions (
                interaction_id TEXT PRIMARY KEY,
                trial_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                ui_layer TEXT NOT NULL,
                client_ts_ms INTEGER,
                server_ts_ms INTEGER NOT NULL
            );
            """
        )

    @staticmethod
    def initialize(
        connection: sqlite3.Connection,
        trial: TrialRecord,
        materials: StudyMaterials,
    ) -> None:
        variant = materials.variants[trial.wine_variant]
        connection.execute(
            "INSERT INTO study_trial_runtime (trial_id, phase) VALUES (?, 'awaiting_briefing')",
            (trial.trial_id,),
        )
        connection.execute(
            "INSERT INTO study_content VALUES (?, 'common-instruction', 'instruction', 'Study instructions', ?, NULL)",
            (trial.trial_id, materials.common_instruction),
        )
        for source in variant.sources:
            connection.execute(
                """INSERT INTO study_content VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    trial.trial_id,
                    source.source_id,
                    "source",
                    source.title,
                    source.text,
                    None if source.release_gate == "briefing" else variant.trace[10].event_id,
                ),
            )
        connection.execute(
            "INSERT INTO study_content VALUES (?, ?, 'draft_preview', 'Writing working draft', ?, ?)",
            (
                trial.trial_id,
                f"{trial.wine_variant}-approval-draft",
                variant.approval_draft_preview,
                variant.trace[8].event_id,
            ),
        )
        connection.execute(
            "INSERT INTO study_content VALUES (?, ?, ?, ?, ?, ?)",
            (
                trial.trial_id,
                f"{trial.wine_variant}-final-excerpt",
                "report_excerpt",
                "Final report excerpt",
                variant.final_report_excerpt,
                variant.trace[10].event_id,
            ),
        )
        for suffix, kind, title, contents in (
            ("final-code", "code_artifact", "Coding analysis.py", variant.final_code_artifact),
            ("final-markdown", "markdown_artifact", "Writing report.md", variant.final_markdown_artifact),
        ):
            connection.execute(
                "INSERT INTO study_content VALUES (?, ?, ?, ?, ?, ?)",
                (
                    trial.trial_id,
                    f"{trial.wine_variant}-{suffix}",
                    kind,
                    title,
                    contents,
                    variant.trace[10].event_id,
                ),
            )
        blocker_id, approval_id, final_id = (
            variant.trace[5].event_id,
            variant.trace[8].event_id,
            variant.trace[10].event_id,
        )
        for source in variant.sources[:2]:
            connection.execute(
                "INSERT INTO study_action_options VALUES (?, ?, ?, ?)",
                (trial.trial_id, blocker_id, source.source_id, f"Use rule from {source.source_id}"),
            )
        for choice in ("approve", "reject", "defer"):
            connection.execute(
                "INSERT INTO study_action_options VALUES (?, ?, ?, ?)",
                (trial.trial_id, approval_id, choice, choice.capitalize()),
            )
        for choice, label in (("ready", "Ready to accept"), ("withhold", "Withhold acceptance")):
            connection.execute(
                "INSERT INTO study_action_options VALUES (?, ?, ?, ?)",
                (trial.trial_id, final_id, choice, label),
            )

    def _open(self) -> sqlite3.Connection:
        connection = self.store._connect()
        self.create_schema(connection)
        return connection

    def _trial_materials(self, trial_id: str) -> tuple[TrialRecord, StudyMaterials, VariantStimuli]:
        trial = self.store.get_trial(trial_id)
        if trial is None:
            raise LookupError("Trial not found")
        materials = self.store.materials_for_trial(trial_id)
        if materials is None:
            raise LookupError("Trial not found")
        return trial, materials, materials.variants[trial.wine_variant]

    def confirm_briefing(self, trial_id: str, confirmation: BriefingConfirmation) -> TrialSnapshot:
        trial, _, variant = self._trial_materials(trial_id)
        now = self.clock()
        with closing(self._open()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._confirm_briefing_in_transaction(
                connection, trial, variant, confirmation, now
            )

    def _confirm_briefing_in_transaction(
        self,
        connection: sqlite3.Connection,
        trial: TrialRecord,
        variant: VariantStimuli,
        confirmation: BriefingConfirmation,
        now: int,
        reviewed_goals: tuple[str, str] | None = None,
    ) -> TrialSnapshot:
        trial_id = trial.trial_id
        allowed = {"both"} if trial.condition == "full_anisha" else {"coding_agent", "writing_agent"}
        if confirmation.agent_id not in allowed:
            raise ValueError("Briefing agent does not match this interface")
        runtime = self._runtime(connection, trial_id)
        if runtime["phase"] != "awaiting_briefing":
            raise ValueError("Briefings are already complete")
        if connection.execute(
            "SELECT 1 FROM study_briefings WHERE trial_id = ? AND agent_id = ?",
            (trial_id, confirmation.agent_id),
        ).fetchone():
            raise ValueError("This briefing was already confirmed")
        connection.execute(
            "INSERT INTO study_briefings VALUES (?, ?, ?, ?)",
            (trial_id, confirmation.agent_id, confirmation.reviewed_transcript.strip(), now),
        )
        if trial.condition == "condition_c" and confirmation.goal and confirmation.goal.strip():
            connection.execute(
                "INSERT INTO study_condition_c_goals VALUES (?, ?, ?)",
                (trial_id, confirmation.agent_id, confirmation.goal.strip()),
            )
        confirmed = {
            row["agent_id"]
            for row in connection.execute(
                "SELECT agent_id FROM study_briefings WHERE trial_id = ?", (trial_id,)
            )
        }
        if confirmed == allowed:
            if trial.condition == "condition_c":
                goals = dict(connection.execute(
                    "SELECT agent_id, goal FROM study_condition_c_goals WHERE trial_id = ?",
                    (trial_id,),
                ).fetchall())
                if len(goals) == 2:
                    reviewed_goals = (goals["coding_agent"], goals["writing_agent"])
            self._create_tasks(connection, trial_id, variant, reviewed_goals)
            connection.execute(
                """UPDATE study_trial_runtime SET phase = 'running',
                   started_at_ms = ?, anchor_ms = ?, next_index = 1 WHERE trial_id = ?""",
                (now, now, trial_id),
            )
            self._deliver(connection, trial_id, variant, variant.trace[0], now, now)
        return self._snapshot(connection, trial, variant)

    def snapshot(self, trial_id: str) -> TrialSnapshot:
        trial, _, variant = self._trial_materials(trial_id)
        with closing(self._open()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            self._advance(connection, trial_id, variant, self.clock())
            return self._snapshot(connection, trial, variant)

    def participant_snapshot(self, trial_id: str) -> TrialSnapshot:
        state = self.snapshot(trial_id)
        if state.phase != "probe":
            return state
        return state.model_copy(
            update={
                "participant_brief": "",
                "tasks": [],
                "records": [],
                "events": [],
                "action_options": [],
                "correction_options": [],
                "correction_for_event_id": None,
                "correction_previous_choice_label": None,
                "correction_previous_reason": None,
                "masked": True,
                "confirmed_agents": [],
            }
        )

    def respond(self, trial_id: str, answer: GateAnswer) -> TrialSnapshot:
        trial, materials, variant = self._trial_materials(trial_id)
        now = self.clock()
        with closing(self._open()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            self._advance(connection, trial_id, variant, now)
            runtime = self._runtime(connection, trial_id)
            phase = runtime["phase"]
            gate_index = {"probe": 4, "blocker": 5, "approval": 8, "final_review": 10}.get(phase)
            if gate_index is None:
                raise ValueError("No decision gate is awaiting a response")
            gate = variant.trace[gate_index]
            if phase != "probe":
                if answer.confidence is None:
                    raise ValueError("Confidence is required before feedback")
                valid = {
                    row["choice_id"]
                    for row in connection.execute(
                        "SELECT choice_id FROM study_action_options WHERE trial_id = ? AND gate_event_id = ?",
                        (trial_id, gate.event_id),
                    )
                }
                if answer.choice_id not in valid:
                    raise ValueError("Invalid choice for this gate")
                if not answer.reason.strip():
                    raise ValueError("A brief reason is required")
                if phase == "final_review" and not answer.claim_text.strip():
                    raise ValueError("The exact claim flagged is required")
            elif not answer.reason.strip():
                raise ValueError("The state-probe answer is required")
            self._save_response(connection, trial_id, gate.event_id, 1, answer, now)
            if phase == "probe":
                connection.execute(
                    "UPDATE study_trial_runtime SET phase = 'awaiting_pair', anchor_ms = ?, next_index = 5 WHERE trial_id = ?",
                    (now, trial_id),
                )
            elif phase == "blocker":
                feedback = (
                    materials.feedback["blocker_correct"]
                    if answer.choice_id == variant.sources[0].source_id
                    else materials.feedback["blocker_wrong"]
                )
                self._deliver(connection, trial_id, variant, variant.trace[7], now, now, feedback)
                connection.execute(
                    "UPDATE study_trial_runtime SET phase = 'awaiting_approval', anchor_ms = ?, next_index = 8 WHERE trial_id = ?",
                    (now, trial_id),
                )
            elif phase == "approval":
                feedback = (
                    materials.feedback["approval_rejected"]
                    if answer.choice_id == "reject"
                    else materials.feedback["approval_approved_or_deferred"]
                )
                self._deliver(connection, trial_id, variant, variant.trace[9], now, now, feedback)
                connection.execute(
                    "UPDATE study_trial_runtime SET phase = 'awaiting_final', anchor_ms = ?, next_index = 10 WHERE trial_id = ?",
                    (now, trial_id),
                )
            else:
                connection.execute(
                    "UPDATE study_trial_runtime SET phase = 'complete', next_index = 11 WHERE trial_id = ?",
                    (trial_id,),
                )
            return self._snapshot(connection, trial, variant)

    def correct(self, trial_id: str, answer: GateAnswer) -> TrialSnapshot:
        trial, _, variant = self._trial_materials(trial_id)
        now = self.clock()
        with closing(self._open()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            phase = self._runtime(connection, trial_id)["phase"]
            gate_index = {"awaiting_approval": 5, "awaiting_final": 8}.get(phase)
            if gate_index is None:
                raise ValueError("No correction window is open")
            gate_id = variant.trace[gate_index].event_id
            first = connection.execute(
                "SELECT choice_id FROM study_responses WHERE trial_id = ? AND gate_event_id = ? AND response_order = 1",
                (trial_id, gate_id),
            ).fetchone()
            if first is None:
                raise ValueError("First response is missing")
            if phase == "awaiting_approval" and first["choice_id"] == variant.sources[0].source_id:
                raise ValueError("No blocker correction is offered after the latest rule")
            if phase == "awaiting_final" and first["choice_id"] == "reject":
                raise ValueError("No approval correction is offered after rejection")
            valid = {
                row["choice_id"]
                for row in connection.execute(
                    "SELECT choice_id FROM study_action_options WHERE trial_id = ? AND gate_event_id = ?",
                    (trial_id, gate_id),
                )
            }
            if answer.choice_id not in valid:
                raise ValueError("Invalid correction choice")
            self._save_response(connection, trial_id, gate_id, 2, answer, now)
            return self._snapshot(connection, trial, variant)

    def pause_after_failure(self, trial_id: str, failure: TechnicalFailure) -> TrialSnapshot:
        trial, _, variant = self._trial_materials(trial_id)
        if not failure.reason.strip():
            raise ValueError("A technical failure reason is required")
        now = self.clock()
        with closing(self._open()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            runtime = self._runtime(connection, trial_id)
            if runtime["phase"] in {"complete", "technical_failure"}:
                raise ValueError("This trial can no longer be paused")
            connection.execute(
                """UPDATE study_trial_runtime SET phase = 'technical_failure',
                   pause_phase = ? WHERE trial_id = ?""",
                (runtime["phase"], trial_id),
            )
            connection.execute(
                "INSERT INTO study_researcher_operations VALUES (?, ?, 'technical_failure', ?, ?)",
                (str(uuid4()), trial_id, failure.reason.strip(), now),
            )
            return self._snapshot(connection, trial, variant)

    def start_next_period(self, trial_id: str) -> TrialRecord:
        trial, _, _ = self._trial_materials(trial_id)
        if trial.period != 1:
            raise ValueError("Only period 1 can be followed by another period")
        with closing(self._open()) as connection:
            phase = self._runtime(connection, trial_id)["phase"]
            existing = connection.execute(
                """SELECT 1 FROM study_trials WHERE participant_id = ?
                   AND slot = ? AND period = 2 LIMIT 1""",
                (trial.participant_id, trial.slot),
            ).fetchone()
        if phase not in {"complete", "technical_failure"}:
            raise ValueError("Finish or stop period 1 before resetting for period 2")
        if existing:
            raise ValueError("Period 2 has already been created for this participant and slot")
        next_trial = self.store.create_trial(
            TrialSetupRequest(
                participant_id=trial.participant_id, slot=trial.slot, period=2
            )
        )
        with closing(self._open()) as connection, connection:
            connection.execute(
                "INSERT INTO study_researcher_operations VALUES (?, ?, 'start_next_period', ?, ?)",
                (str(uuid4()), trial_id, next_trial.trial_id, self.clock()),
            )
        return next_trial

    def export_trial(self, trial_id: str) -> dict:
        """Export stored study facts and history, even if source files later change."""
        trial = self.store.get_trial(trial_id)
        if trial is None:
            raise LookupError("Trial not found")
        tables = {
            "briefings": "study_briefings",
            "tasks": "study_tasks",
            "content": "study_content",
            "events": "study_event_history",
            "action_options": "study_action_options",
            "responses": "study_responses",
            "researcher_operations": "study_researcher_operations",
            "condition_c_goals": "study_condition_c_goals",
            "interactions": "study_interactions",
        }
        with closing(self._open()) as connection:
            runtime_row = connection.execute(
                "SELECT * FROM study_trial_runtime WHERE trial_id = ?", (trial_id,)
            ).fetchone()
            runtime = dict(runtime_row) if runtime_row is not None else None
            rows = {
                key: [dict(row) for row in connection.execute(
                    f"SELECT * FROM {table} WHERE trial_id = ? ORDER BY rowid", (trial_id,)
                )]
                for key, table in tables.items()
            }
        return {"trial": trial.model_dump(), "runtime": runtime, **rows}

    def record_interaction(
        self, trial_id: str, interaction: ParticipantInteraction
    ) -> InteractionReceipt:
        trial, _, variant = self._trial_materials(trial_id)
        now = self.clock()
        with closing(self._open()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            self._advance(connection, trial_id, variant, now)
            if self._runtime(connection, trial_id)["phase"] == "probe":
                raise ValueError("Phone content is hidden during the state probe")
            if interaction.event_type == "event_open":
                event = connection.execute(
                    """SELECT script_key FROM study_event_history
                       WHERE trial_id = ? AND event_id = ?""",
                    (trial_id, interaction.target_id),
                ).fetchone()
                if event is None or event["script_key"] == "probe_dependency":
                    raise ValueError("Event is not available in this trial")
            elif interaction.event_type == "source_open":
                record = connection.execute(
                    "SELECT release_event_id FROM study_content WHERE trial_id = ? AND record_id = ?",
                    (trial_id, interaction.target_id),
                ).fetchone()
                if record is None:
                    raise ValueError("Source is not available in this trial")
                if record["release_event_id"] and not connection.execute(
                    "SELECT 1 FROM study_event_history WHERE trial_id = ? AND event_id = ?",
                    (trial_id, record["release_event_id"]),
                ).fetchone():
                    raise ValueError("Source has not been released")
            elif interaction.event_type == "cue_delivered":
                if not connection.execute(
                    "SELECT 1 FROM study_event_history WHERE trial_id = ? AND cue_id = ?",
                    (trial_id, interaction.target_id),
                ).fetchone():
                    raise ValueError("Cue is not available in this trial")
                previous = connection.execute(
                    """SELECT interaction_id, server_ts_ms FROM study_interactions WHERE trial_id = ?
                       AND event_type = 'cue_delivered' AND target_id = ?""",
                    (trial_id, interaction.target_id),
                ).fetchone()
                if previous:
                    return InteractionReceipt(
                        interaction_id=previous["interaction_id"],
                        server_ts_ms=previous["server_ts_ms"],
                        is_new=False,
                    )
            receipt = InteractionReceipt(interaction_id=str(uuid4()), server_ts_ms=now)
            connection.execute(
                "INSERT INTO study_interactions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    receipt.interaction_id, trial.trial_id, interaction.event_type,
                    interaction.target_id, interaction.ui_layer,
                    interaction.client_ts_ms, receipt.server_ts_ms,
                ),
            )
            return receipt

    @staticmethod
    def _runtime(connection: sqlite3.Connection, trial_id: str) -> sqlite3.Row:
        runtime = connection.execute(
            "SELECT * FROM study_trial_runtime WHERE trial_id = ?", (trial_id,)
        ).fetchone()
        if runtime is None:
            raise ValueError("Trial runtime was not initialized")
        return runtime

    @staticmethod
    def _create_tasks(
        connection: sqlite3.Connection,
        trial_id: str,
        variant: VariantStimuli,
        reviewed_goals: tuple[str, str] | None = None,
    ) -> None:
        accepted = variant.event_copy["tasks_accepted"]
        coding_goal, writing_goal = accepted.split("Writing:", 1)
        coding_goal = reviewed_goals[0] if reviewed_goals else coding_goal.removeprefix("Coding:").strip()
        writing_goal = reviewed_goals[1] if reviewed_goals else writing_goal.strip()
        coding_id, writing_id = f"{trial_id}:coding", f"{trial_id}:writing"
        connection.execute(
            "INSERT INTO study_tasks VALUES (?, ?, ?, ?, ?, ?, 0, 0)",
            (trial_id, coding_id, "coding_agent", coding_goal, "running", None),
        )
        connection.execute(
            "INSERT INTO study_tasks VALUES (?, ?, ?, ?, ?, ?, 0, 0)",
            (trial_id, writing_id, "writing_agent", writing_goal, "running", coding_id),
        )

    @staticmethod
    def _task_id(trial_id: str, key: str) -> str | None:
        if key in {"validation_progress", "analysis_started", "conflicting_definition", "conflicting_outlier_rule", "blocker_feedback"}:
            return f"{trial_id}:coding"
        if key in {"outline_complete", "competing_completion", "overwrite_request", "publish_request", "approval_feedback", "final_review"}:
            return f"{trial_id}:writing"
        return None

    def _deliver(
        self,
        connection: sqlite3.Connection,
        trial_id: str,
        variant: VariantStimuli,
        event: EventSpec,
        scheduled_at_ms: int,
        occurred_at_ms: int,
        feedback_text: str | None = None,
        cue_id: str | None = None,
    ) -> None:
        text = feedback_text if feedback_text is not None else variant.event_copy[event.script_key]
        connection.execute(
            "INSERT INTO study_event_history VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trial_id, event.event_id, self._task_id(trial_id, event.script_key),
                event.script_key, text, scheduled_at_ms, occurred_at_ms, cue_id,
            ),
        )
        if event.script_key == "outline_complete" or event.script_key == "competing_completion":
            connection.execute(
                "UPDATE study_tasks SET state = 'waiting_for_verified_results', outline_ready = 1 WHERE trial_id = ? AND agent_id = 'writing_agent'",
                (trial_id,),
            )
        elif event.script_key in {"conflicting_definition", "conflicting_outlier_rule"}:
            connection.execute(
                "UPDATE study_tasks SET state = 'blocked' WHERE trial_id = ? AND agent_id = 'coding_agent'",
                (trial_id,),
            )
        elif event.script_key == "blocker_feedback":
            connection.execute(
                "UPDATE study_tasks SET state = 'running' WHERE trial_id = ? AND agent_id = 'coding_agent'",
                (trial_id,),
            )
        elif event.script_key in {"overwrite_request", "publish_request"}:
            connection.execute(
                "UPDATE study_tasks SET state = 'awaiting_approval' WHERE trial_id = ? AND agent_id = 'writing_agent'",
                (trial_id,),
            )
        elif event.script_key == "approval_feedback":
            connection.execute(
                "UPDATE study_tasks SET state = 'waiting_for_verified_results' WHERE trial_id = ? AND agent_id = 'writing_agent'",
                (trial_id,),
            )
        elif event.script_key == "final_review":
            connection.execute(
                """UPDATE study_tasks SET state = 'completed', verified_results_ready = 1
                   WHERE trial_id = ? AND agent_id = 'coding_agent'""",
                (trial_id,),
            )
            verified_coding = connection.execute(
                """SELECT 1 FROM study_tasks WHERE trial_id = ? AND agent_id = 'coding_agent'
                   AND state = 'completed' AND verified_results_ready = 1""",
                (trial_id,),
            ).fetchone()
            if verified_coding is None:
                raise ValueError("Writing results cannot complete before verified Coding output")
            connection.execute(
                """UPDATE study_tasks SET state = 'completed', verified_results_ready = 1
                   WHERE trial_id = ? AND agent_id = 'writing_agent'
                   AND depends_on = ?""",
                (trial_id, f"{trial_id}:coding"),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                raise ValueError("Writing dependency is missing or does not match Coding")

    def _advance(
        self, connection: sqlite3.Connection, trial_id: str, variant: VariantStimuli, now: int
    ) -> None:
        runtime = self._runtime(connection, trial_id)
        phase, anchor = runtime["phase"], runtime["anchor_ms"]
        if phase == "running":
            for index in range(runtime["next_index"], 5):
                event = variant.trace[index]
                due = anchor + event.offset_s * 1000
                if now < due:
                    break
                self._deliver(connection, trial_id, variant, event, due, now)
                connection.execute(
                    "UPDATE study_trial_runtime SET next_index = ? WHERE trial_id = ?",
                    (index + 1, trial_id),
                )
                if index == 4:
                    connection.execute(
                        "UPDATE study_trial_runtime SET phase = 'probe' WHERE trial_id = ?",
                        (trial_id,),
                    )
                    break
        elif phase == "awaiting_pair":
            first, second = variant.trace[5:7]
            due = anchor + first.offset_s * 1000
            if now >= due:
                cue = f"{trial_id}:paired-cue"
                self._deliver(connection, trial_id, variant, first, due, now, cue_id=cue)
                self._deliver(connection, trial_id, variant, second, due, now, cue_id=cue)
                connection.execute(
                    "UPDATE study_trial_runtime SET phase = 'blocker', next_index = 7 WHERE trial_id = ?",
                    (trial_id,),
                )
        elif phase in {"awaiting_approval", "awaiting_final"}:
            index = 8 if phase == "awaiting_approval" else 10
            event = variant.trace[index]
            due = anchor + event.offset_s * 1000
            if now >= due:
                self._deliver(connection, trial_id, variant, event, due, now)
                connection.execute(
                    "UPDATE study_trial_runtime SET phase = ?, next_index = ? WHERE trial_id = ?",
                    ("approval" if index == 8 else "final_review", index + 1, trial_id),
                )

    @staticmethod
    def _save_response(
        connection: sqlite3.Connection,
        trial_id: str,
        gate_event_id: str,
        order: int,
        answer: GateAnswer,
        now: int,
    ) -> None:
        event = connection.execute(
            "SELECT occurred_at_ms FROM study_event_history WHERE trial_id = ? AND event_id = ?",
            (trial_id, gate_event_id),
        ).fetchone()
        if event is None:
            raise ValueError("Decision event has not been delivered")
        try:
            connection.execute(
                "INSERT INTO study_responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()), trial_id, gate_event_id, order, answer.choice_id,
                    answer.reason.strip(), answer.confidence, answer.claim_text.strip(),
                    now, max(0, now - event["occurred_at_ms"]),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("This response has already been recorded") from exc

    def _snapshot(
        self, connection: sqlite3.Connection, trial: TrialRecord, variant: VariantStimuli
    ) -> TrialSnapshot:
        runtime = self._runtime(connection, trial.trial_id)
        connection.execute(
            "UPDATE study_trials SET status = ? WHERE trial_id = ?",
            (runtime["phase"], trial.trial_id),
        )
        events = [
            DeliveredEvent(**dict(row))
            for row in connection.execute(
                """SELECT event_id, task_id, script_key, text, scheduled_at_ms,
                          occurred_at_ms, cue_id FROM study_event_history
                   WHERE trial_id = ? ORDER BY rowid""",
                (trial.trial_id,),
            )
        ]
        delivered_ids = {event.event_id for event in events}
        records = [
            ContentRecord(**{key: row[key] for key in ("record_id", "kind", "title", "text")})
            for row in connection.execute(
                "SELECT * FROM study_content WHERE trial_id = ? ORDER BY record_id", (trial.trial_id,)
            )
            if row["release_event_id"] is None or row["release_event_id"] in delivered_ids
        ]
        tasks = [
            TaskRecord(
                **{
                    **dict(row),
                    "outline_ready": bool(row["outline_ready"]),
                    "verified_results_ready": bool(row["verified_results_ready"]),
                },
            )
            for row in connection.execute(
                """SELECT task_id, agent_id, goal, state, depends_on, outline_ready,
                          verified_results_ready FROM study_tasks
                   WHERE trial_id = ? ORDER BY agent_id""",
                (trial.trial_id,),
            )
        ]
        confirmed_agents = [
            row["agent_id"] for row in connection.execute(
                "SELECT agent_id FROM study_briefings WHERE trial_id = ? ORDER BY confirmed_at_ms",
                (trial.trial_id,),
            )
        ]
        active_gate = {
            "blocker": variant.trace[5].event_id,
            "approval": variant.trace[8].event_id,
            "final_review": variant.trace[10].event_id,
        }.get(runtime["phase"])
        actions = (
            [ActionOption(**dict(row)) for row in connection.execute(
                "SELECT gate_event_id, choice_id, label FROM study_action_options WHERE trial_id = ? AND gate_event_id = ? ORDER BY rowid",
                (trial.trial_id, active_gate),
            )]
            if active_gate else []
        )
        correction_gate = {
            "awaiting_approval": variant.trace[5].event_id,
            "awaiting_final": variant.trace[8].event_id,
        }.get(runtime["phase"])
        correction_options = []
        correction_previous_choice_label = None
        correction_previous_reason = None
        if correction_gate:
            first = connection.execute(
                """SELECT choice_id, reason FROM study_responses WHERE trial_id = ?
                   AND gate_event_id = ? AND response_order = 1""",
                (trial.trial_id, correction_gate),
            ).fetchone()
            corrected = connection.execute(
                """SELECT 1 FROM study_responses WHERE trial_id = ?
                   AND gate_event_id = ? AND response_order = 2""",
                (trial.trial_id, correction_gate),
            ).fetchone()
            allowed = (
                first is not None
                and corrected is None
                and (
                    (runtime["phase"] == "awaiting_approval" and first["choice_id"] != variant.sources[0].source_id)
                    or (runtime["phase"] == "awaiting_final" and first["choice_id"] != "reject")
                )
            )
            if allowed:
                correction_options = [ActionOption(**dict(row)) for row in connection.execute(
                    """SELECT gate_event_id, choice_id, label FROM study_action_options
                       WHERE trial_id = ? AND gate_event_id = ? ORDER BY rowid""",
                    (trial.trial_id, correction_gate),
                )]
                previous_option = next(
                    (option for option in correction_options if option.choice_id == first["choice_id"]),
                    None,
                )
                if previous_option is None:
                    raise ValueError("First response has no matching action option")
                correction_previous_choice_label = previous_option.label
                correction_previous_reason = first["reason"]
        due = None
        if runtime["phase"] in {"running", "awaiting_pair", "awaiting_approval", "awaiting_final"}:
            due = runtime["anchor_ms"] + variant.trace[runtime["next_index"]].offset_s * 1000
        return TrialSnapshot(
            trial_id=trial.trial_id,
            interface_label=trial.interface_label,
            participant_brief=variant.participant_brief,
            phase=runtime["phase"],
            started_at_ms=runtime["started_at_ms"],
            due_at_ms=due,
            tasks=tasks,
            confirmed_agents=confirmed_agents,
            records=records,
            events=events,
            action_options=actions,
            correction_options=correction_options,
            correction_for_event_id=correction_gate if correction_options else None,
            correction_previous_choice_label=correction_previous_choice_label,
            correction_previous_reason=correction_previous_reason,
        )
