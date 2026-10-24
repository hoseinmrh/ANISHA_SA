"""Condition C study API: two direct agent screens, one fixed trial runner."""

from __future__ import annotations

import hmac
import os
import secrets
import sqlite3
import json
from contextlib import closing
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from backend.study.runner import (
    BriefingConfirmation, GateAnswer, ParticipantInteraction, StudyRunner,
    TechnicalFailure, participant_view,
)
from backend.study.setup import StudySetupStore, TrialSetupRequest, load_allocations
from backend.study.followup import FollowupReview, FollowupResult, resolve_followup
from backend.voice.stt import SpeechToText
from backend.voice.intent import IntentExtractor, IntentExtractionError
from backend.live_agents import AgentTaskProposal, DESCRIPTORS, TaskDetails, get_live_runtime
from backend.workspace.store import DuplicateRecordError, UnknownTaskError

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ENV_FILE = ROOT / "backend" / ".env"
load_dotenv(BACKEND_ENV_FILE, override=False)
store = StudySetupStore(Path(os.getenv("SEPARATE_AGENTS_DB_PATH", ROOT / "data" / "trials.sqlite3")))
runner = StudyRunner(store)
speech_service = SpeechToText(model_size=os.getenv("ANISHA_WHISPER_MODEL", "small"))
app = FastAPI(title="Separate Agents study interface")


def get_intent_extractor() -> IntentExtractor:
    return IntentExtractor(
        env_file=BACKEND_ENV_FILE,
        agent_catalog=list(DESCRIPTORS.values()),
    )


class Activation(BaseModel):
    transcription_id: str
    activation_index: int


class Transcript(BaseModel):
    transcription_id: str
    activation_index: int
    recognized_transcript: str
    speech_service_version: str
    status: str


class ProposalRequest(BaseModel):
    transcription_id: str
    reviewed_transcript: str = Field(min_length=1)


class Proposal(BaseModel):
    proposal_id: str
    agent_id: str
    reviewed_transcript: str
    goal: str
    constraints: list[str]
    source_refs: list[str]
    dependency: str | None
    approval_boundary: str


class ConfirmRequest(BaseModel):
    goal: str = Field(min_length=1)


def _error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=404 if isinstance(exc, LookupError) else 422, detail=str(exc))


def _researcher_token() -> str:
    token = os.getenv("ANISHA_STUDY_RESEARCHER_TOKEN", "")
    if not token:
        raise HTTPException(503, "Researcher access is not configured")
    return token


def researcher_auth(token: str | None = Header(None, alias="X-ANISHA-Researcher-Token")) -> None:
    if not token or not secrets.compare_digest(token, _researcher_token()):
        raise HTTPException(401, "Researcher token required")


def participant_token(trial_id: str) -> str:
    return hmac.digest(_researcher_token().encode(), trial_id.encode(), "sha256").hex()


def participant_auth(
    trial_id: str,
    token: str | None = Header(None, alias="X-ANISHA-Study-Token"),
) -> None:
    if store.get_trial(trial_id) is None:
        raise HTTPException(404, "Trial not found")
    if not token or not secrets.compare_digest(token, participant_token(trial_id)):
        raise HTTPException(401, "Participant link required")


def condition_c(trial_id: str) -> None:
    trial = store.get_trial(trial_id)
    if trial is None:
        raise LookupError("Trial not found")
    if trial.condition != "condition_c":
        raise ValueError("This participant app is for the assigned separate-agent interface")


def _voice_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS condition_c_voice_attempts (
            transcription_id TEXT PRIMARY KEY,
            trial_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            activation_index INTEGER NOT NULL,
            purpose TEXT NOT NULL DEFAULT 'briefing',
            status TEXT NOT NULL,
            recognized_text TEXT NOT NULL DEFAULT '',
            reviewed_text TEXT NOT NULL DEFAULT '',
            model_version TEXT NOT NULL,
            audio_bytes INTEGER NOT NULL DEFAULT 0,
            started_at_ms INTEGER NOT NULL,
            completed_at_ms INTEGER
        );
        CREATE TABLE IF NOT EXISTS condition_c_proposals (
            proposal_id TEXT PRIMARY KEY,
            trial_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            transcription_id TEXT NOT NULL,
            reviewed_transcript TEXT NOT NULL,
            initial_goal TEXT NOT NULL,
            confirmed_goal TEXT,
            created_at_ms INTEGER NOT NULL,
            confirmed_at_ms INTEGER
        );
        CREATE TABLE IF NOT EXISTS condition_c_followups (
            request_id TEXT PRIMARY KEY,
            trial_id TEXT NOT NULL,
            transcription_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            reviewed_transcript TEXT NOT NULL,
            version_kind TEXT NOT NULL,
            intent_model TEXT NOT NULL,
            intent_json TEXT NOT NULL,
            status TEXT NOT NULL,
            request_kind TEXT,
            message TEXT NOT NULL,
            record_ids_json TEXT NOT NULL,
            suggested_choice_id TEXT,
            created_at_ms INTEGER NOT NULL
        );
    """)
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(condition_c_voice_attempts)")}
    if "purpose" not in columns:
        connection.execute("ALTER TABLE condition_c_voice_attempts ADD COLUMN purpose TEXT NOT NULL DEFAULT 'briefing'")


def _voice_connection() -> sqlite3.Connection:
    connection = store._connect()
    _voice_schema(connection)
    return connection


def _agent(agent_id: str) -> None:
    if agent_id not in {"coding_agent", "writing_agent"}:
        raise HTTPException(404, "Unknown agent")


def _speech_phase(trial_id: str, agent_id: str, purpose: str) -> None:
    state = runner.snapshot(trial_id)
    if purpose == "briefing":
        if state.phase != "awaiting_briefing" or agent_id in state.confirmed_agents:
            raise ValueError("This agent's briefing is unavailable")
    elif purpose == "followup":
        if state.phase in {"awaiting_briefing", "probe", "complete", "technical_failure"}:
            raise ValueError("Follow-up speech is unavailable in this phase")
        if agent_id not in {task.agent_id for task in state.tasks}:
            raise ValueError("The selected agent has no task in this trial")
    else:
        raise ValueError("Unknown speech purpose")


@app.get("/")
def participant_app():
    return FileResponse(ROOT / "frontend" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/app.js")
def participant_script():
    return FileResponse(ROOT / "frontend" / "app.js", media_type="text/javascript",
                        headers={"Cache-Control": "no-store"})


@app.get("/styles.css")
def participant_styles():
    return FileResponse(ROOT / "frontend" / "styles.css", media_type="text/css",
                        headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    return {"status": "ok"}


def _live_error(exc: Exception) -> HTTPException:
    if isinstance(exc, UnknownTaskError):
        return HTTPException(404, "Task not found")
    if isinstance(exc, DuplicateRecordError):
        return HTTPException(409, "Task already exists")
    return HTTPException(422, str(exc))


@app.get("/agents")
def list_agents():
    return [descriptor.__dict__ for descriptor in DESCRIPTORS.values()]


@app.post("/agents/{agent_id}/projects/{project_id}/tasks", response_model=TaskDetails)
def propose_live_task(agent_id: str, project_id: str, proposal: AgentTaskProposal):
    try:
        return get_live_runtime().propose(agent_id, project_id, proposal)
    except (ValueError, DuplicateRecordError, RuntimeError) as exc:
        raise _live_error(exc) from exc


@app.get("/agents/{agent_id}/projects/{project_id}/tasks")
def list_live_tasks(agent_id: str, project_id: str):
    try:
        return get_live_runtime().list_tasks(agent_id, project_id)
    except ValueError as exc:
        raise _live_error(exc) from exc


@app.get("/agents/{agent_id}/projects/{project_id}/tasks/{task_id}", response_model=TaskDetails)
def get_live_task(agent_id: str, project_id: str, task_id: str):
    try:
        return get_live_runtime().details(agent_id, project_id, task_id)
    except (ValueError, UnknownTaskError) as exc:
        raise _live_error(exc) from exc


@app.post("/agents/{agent_id}/projects/{project_id}/tasks/{task_id}/confirm",
          response_model=TaskDetails, status_code=status.HTTP_202_ACCEPTED)
async def confirm_live_task(agent_id: str, project_id: str, task_id: str):
    try:
        return await get_live_runtime().confirm(agent_id, project_id, task_id)
    except (ValueError, UnknownTaskError, RuntimeError) as exc:
        raise _live_error(exc) from exc


@app.post("/agents/{agent_id}/projects/{project_id}/tasks/{task_id}/cancel",
          response_model=TaskDetails)
async def cancel_live_task(agent_id: str, project_id: str, task_id: str):
    try:
        return await get_live_runtime().cancel(agent_id, project_id, task_id)
    except (ValueError, UnknownTaskError) as exc:
        raise _live_error(exc) from exc


@app.post("/agents/{agent_id}/projects/{project_id}/tasks/{task_id}/approvals/{approval_id}/approve",
          response_model=TaskDetails, status_code=status.HTTP_202_ACCEPTED)
async def approve_live_task_action(agent_id: str, project_id: str, task_id: str, approval_id: str):
    try:
        return await get_live_runtime().approve(agent_id, project_id, task_id, approval_id)
    except (ValueError, UnknownTaskError) as exc:
        raise _live_error(exc) from exc


@app.post("/study/researcher/trials", dependencies=[Depends(researcher_auth)])
def create_trial(request: TrialSetupRequest):
    try:
        allocations, _ = load_allocations(store.allocation_path)
        allocation = next((row for row in allocations if row["slot"] == request.slot), None)
        if allocation is None or allocation[f"period_{request.period}_condition"] != "condition_c":
            raise ValueError("Choose a slot and period allocated to the separate-agent interface")
        trial = store.create_trial(request)
        return {"trial": trial, "participant_url": f"/trial/{trial.trial_id}#access={participant_token(trial.trial_id)}"}
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc


@app.get("/study/researcher/trials/{trial_id}", dependencies=[Depends(researcher_auth)])
def researcher_state(trial_id: str):
    try:
        condition_c(trial_id)
        return runner.snapshot(trial_id)
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc


@app.post("/study/researcher/trials/{trial_id}/probe", dependencies=[Depends(researcher_auth)])
def probe_answer(trial_id: str, answer: GateAnswer):
    try:
        condition_c(trial_id)
        if runner.snapshot(trial_id).phase != "probe":
            raise ValueError("State probe is not active")
        return runner.respond(trial_id, answer)
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc


@app.post("/study/researcher/trials/{trial_id}/technical-failure", dependencies=[Depends(researcher_auth)])
def technical_failure(trial_id: str, failure: TechnicalFailure):
    try:
        condition_c(trial_id)
        return runner.pause_after_failure(trial_id, failure)
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc


@app.get("/study/researcher/trials/{trial_id}/export", dependencies=[Depends(researcher_auth)])
def export_trial(trial_id: str):
    try:
        condition_c(trial_id)
        export = runner.export_trial(trial_id)
        with closing(_voice_connection()) as connection:
            export["condition_c_voice_attempts"] = [dict(row) for row in connection.execute(
                "SELECT * FROM condition_c_voice_attempts WHERE trial_id = ? ORDER BY started_at_ms", (trial_id,)
            )]
            export["condition_c_proposals"] = [dict(row) for row in connection.execute(
                "SELECT * FROM condition_c_proposals WHERE trial_id = ? ORDER BY created_at_ms", (trial_id,)
            )]
            export["condition_c_followups"] = [dict(row) for row in connection.execute(
                "SELECT * FROM condition_c_followups WHERE trial_id = ? ORDER BY created_at_ms", (trial_id,)
            )]
        return export
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc


@app.get("/trial/{trial_id}")
def trial_app(trial_id: str):
    try:
        condition_c(trial_id)
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc
    return FileResponse(ROOT / "frontend" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/study/trials/{trial_id}/state", dependencies=[Depends(participant_auth)])
def participant_state(trial_id: str):
    try:
        condition_c(trial_id)
        return participant_view(runner.participant_snapshot(trial_id))
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc


@app.post("/study/trials/{trial_id}/agents/{agent_id}/speech/activations", dependencies=[Depends(participant_auth)])
def begin_activation(
    trial_id: str, agent_id: str,
    purpose: str = Query(default="briefing", pattern="^(briefing|followup)$"),
) -> Activation:
    _agent(agent_id)
    try:
        condition_c(trial_id)
        _speech_phase(trial_id, agent_id, purpose)
        with closing(_voice_connection()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            index = connection.execute(
                "SELECT COALESCE(MAX(activation_index), 0) + 1 FROM condition_c_voice_attempts WHERE trial_id = ?", (trial_id,)
            ).fetchone()[0]
            ticket = Activation(transcription_id=str(uuid4()), activation_index=index)
            connection.execute(
                "INSERT INTO condition_c_voice_attempts (transcription_id, trial_id, agent_id, activation_index, purpose, status, model_version, started_at_ms) VALUES (?, ?, ?, ?, ?, 'recording', ?, ?)",
                (ticket.transcription_id, trial_id, agent_id, index, purpose, speech_service.model_size, runner.clock()),
            )
            return ticket
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc


@app.post("/study/trials/{trial_id}/agents/{agent_id}/speech/activations/{transcription_id}/abort", dependencies=[Depends(participant_auth)])
def abort_activation(trial_id: str, agent_id: str, transcription_id: str):
    _agent(agent_id)
    with closing(_voice_connection()) as connection, connection:
        cursor = connection.execute(
            "UPDATE condition_c_voice_attempts SET status = 'aborted', completed_at_ms = ? WHERE trial_id = ? AND agent_id = ? AND transcription_id = ? AND status = 'recording'",
            (runner.clock(), trial_id, agent_id, transcription_id),
        )
        if cursor.rowcount != 1:
            raise HTTPException(422, "Activation is unavailable")
    return {"status": "aborted"}


@app.post("/study/trials/{trial_id}/agents/{agent_id}/speech", dependencies=[Depends(participant_auth)])
async def transcribe(
    trial_id: str, agent_id: str,
    activation_id: str = Query(...),
    purpose: str = Query(default="briefing", pattern="^(briefing|followup)$"),
    audio: bytes = Body(media_type="application/octet-stream"),
) -> Transcript:
    _agent(agent_id)
    try:
        condition_c(trial_id)
        _speech_phase(trial_id, agent_id, purpose)
        with closing(_voice_connection()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT activation_index FROM condition_c_voice_attempts WHERE trial_id = ? AND agent_id = ? AND transcription_id = ? AND purpose = ? AND status = 'recording'",
                (trial_id, agent_id, activation_id, purpose),
            ).fetchone()
            if row is None:
                raise ValueError("Microphone activation is unavailable")
            connection.execute(
                "UPDATE condition_c_voice_attempts SET status = 'transcribing', audio_bytes = ? WHERE transcription_id = ?",
                (len(audio), activation_id),
            )
        status, recognized = "failed", ""
        if len(audio) >= 32:
            try:
                recognized = (await speech_service.transcribe_async(BytesIO(audio))).text
                status = "recognized" if recognized else "empty"
            except Exception:
                status = "failed"
        with closing(_voice_connection()) as connection, connection:
            connection.execute(
                "UPDATE condition_c_voice_attempts SET status = ?, recognized_text = ?, completed_at_ms = ? WHERE transcription_id = ?",
                (status, recognized, runner.clock(), activation_id),
            )
        return Transcript(
            transcription_id=activation_id, activation_index=row["activation_index"],
            recognized_transcript=recognized, speech_service_version=speech_service.model_size,
            status=status,
        )
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc


@app.post("/study/trials/{trial_id}/agents/{agent_id}/proposals", dependencies=[Depends(participant_auth)])
def prepare_proposal(trial_id: str, agent_id: str, review: ProposalRequest) -> Proposal:
    _agent(agent_id)
    try:
        condition_c(trial_id)
        state = runner.snapshot(trial_id)
        if state.phase != "awaiting_briefing" or agent_id in state.confirmed_agents:
            raise ValueError("This briefing is unavailable")
        with closing(_voice_connection()) as connection, connection:
            row = connection.execute(
                "SELECT status FROM condition_c_voice_attempts WHERE trial_id = ? AND agent_id = ? AND transcription_id = ? AND purpose = 'briefing'",
                (trial_id, agent_id, review.transcription_id),
            ).fetchone()
            if row is None or row["status"] != "recognized":
                raise ValueError("Review requires recognized speech")
            proposal_id = str(uuid4())
            reviewed = review.reviewed_transcript.strip()
            with connection:
                connection.execute(
                    "UPDATE condition_c_voice_attempts SET reviewed_text = ? WHERE transcription_id = ?",
                    (reviewed, review.transcription_id),
                )
                connection.execute(
                    "INSERT INTO condition_c_proposals (proposal_id, trial_id, agent_id, transcription_id, reviewed_transcript, initial_goal, created_at_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (proposal_id, trial_id, agent_id, review.transcription_id, reviewed, reviewed, runner.clock()),
                )
        return Proposal(
            proposal_id=proposal_id, agent_id=agent_id, reviewed_transcript=reviewed,
            goal=reviewed,
            constraints=["Use the supplied wine sources; keep the fixed study trace."],
            source_refs=[record.record_id for record in state.records if record.kind == "source"],
            dependency="Writing results wait for verified Coding output" if agent_id == "writing_agent" else None,
            approval_boundary="No overwrite or publication occurs without a reviewed decision.",
        )
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc


@app.post("/study/trials/{trial_id}/agents/{agent_id}/followups",
          response_model=FollowupResult, dependencies=[Depends(participant_auth)])
async def review_followup(trial_id: str, agent_id: str, review: FollowupReview) -> FollowupResult:
    _agent(agent_id)
    try:
        condition_c(trial_id)
        _speech_phase(trial_id, agent_id, "followup")
        with closing(_voice_connection()) as connection:
            attempt = connection.execute(
                """SELECT recognized_text FROM condition_c_voice_attempts
                   WHERE trial_id = ? AND agent_id = ? AND transcription_id = ?
                   AND purpose = 'followup' AND status = 'recognized'""",
                (trial_id, agent_id, review.transcription_id),
            ).fetchone()
        if attempt is None:
            raise ValueError("Recognized follow-up speech is unavailable")
        reviewed = (review.reviewed_transcript if review.reviewed_transcript is not None
                    else attempt["recognized_text"]).strip()
        if not reviewed:
            raise ValueError("Follow-up transcript must not be blank")
        extractor = get_intent_extractor()
        intent = await extractor.extract(reviewed)
        state = runner.participant_snapshot(trial_id)
        _speech_phase(trial_id, agent_id, "followup")
        resolution = resolve_followup(intent, state, agent_id)
        if resolution.revealed_event_id:
            runner.record_interaction(trial_id, ParticipantInteraction(
                event_type="event_open",
                target_id=resolution.revealed_event_id,
                ui_layer=f"{agent_id}_voice",
            ))
        now = runner.clock()
        with closing(_voice_connection()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            current = runner._runtime(connection, trial_id)
            if current["phase"] in {"probe", "complete", "technical_failure"}:
                raise ValueError("Follow-up speech is unavailable in this phase")
            connection.execute(
                "UPDATE condition_c_voice_attempts SET reviewed_text = ? WHERE transcription_id = ?",
                (reviewed, review.transcription_id),
            )
            result = resolution.result.model_copy(update={"revealed_event_id": resolution.revealed_event_id})
            connection.execute(
                """INSERT INTO condition_c_followups VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid4()), trial_id, review.transcription_id, agent_id, reviewed,
                    "reviewed" if review.reviewed_transcript is not None else "used_as_recognized",
                    extractor.model, intent.model_dump_json(), result.status, result.request_kind,
                    result.message, json.dumps(result.record_ids), result.suggested_choice_id, now,
                ),
            )
        return result
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc
    except (IntentExtractionError, RuntimeError) as exc:
        raise HTTPException(502, "The question could not be interpreted. Please retry or use the on-screen controls.") from exc


@app.post("/study/trials/{trial_id}/agents/{agent_id}/proposals/{proposal_id}/confirm", dependencies=[Depends(participant_auth)])
def confirm_proposal(trial_id: str, agent_id: str, proposal_id: str, request: ConfirmRequest):
    _agent(agent_id)
    try:
        condition_c(trial_id)
        with closing(_voice_connection()) as connection:
            row = connection.execute(
                "SELECT reviewed_transcript, confirmed_at_ms FROM condition_c_proposals WHERE trial_id = ? AND agent_id = ? AND proposal_id = ?",
                (trial_id, agent_id, proposal_id),
            ).fetchone()
        if row is None or row["confirmed_at_ms"] is not None:
            raise ValueError("Proposal is unavailable")
        state = runner.confirm_briefing(trial_id, BriefingConfirmation(
            agent_id=agent_id, reviewed_transcript=row["reviewed_transcript"], goal=request.goal.strip(),
        ))
        with closing(_voice_connection()) as connection, connection:
            connection.execute(
                "UPDATE condition_c_proposals SET confirmed_goal = ?, confirmed_at_ms = ? WHERE proposal_id = ?",
                (request.goal.strip(), runner.clock(), proposal_id),
            )
        return participant_view(state)
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc


@app.post("/study/trials/{trial_id}/interactions", dependencies=[Depends(participant_auth)])
def interaction(trial_id: str, entry: ParticipantInteraction):
    try:
        condition_c(trial_id)
        return runner.record_interaction(trial_id, entry)
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc


@app.post("/study/trials/{trial_id}/responses", dependencies=[Depends(participant_auth)])
def respond(trial_id: str, answer: GateAnswer):
    try:
        condition_c(trial_id)
        if runner.snapshot(trial_id).phase == "probe":
            raise ValueError("The researcher records the masked state probe")
        return participant_view(runner.respond(trial_id, answer))
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc


@app.post("/study/trials/{trial_id}/corrections", dependencies=[Depends(participant_auth)])
def correct(trial_id: str, answer: GateAnswer):
    try:
        condition_c(trial_id)
        return participant_view(runner.correct(trial_id, answer))
    except (LookupError, ValueError) as exc:
        raise _error(exc) from exc
