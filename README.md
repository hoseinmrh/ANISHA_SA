# Separate Agents (Condition C)

This is the independent, two-screen phone interface for the thesis comparison. Coding and Writing each have a tab, a separate microphone briefing, their own chronological event feed, and normal decision controls. The participant sees the allocated neutral **Interface A/B** label. There is no project-level attention ranking or orchestrator route in this app.

## What works in this first implementation

- Researcher creates an allocated Condition C trial at `/`; the participant opens the issued trial link. The link carries a trial-scoped access token. Set `ANISHA_STUDY_RESEARCHER_TOKEN` before starting the server.
- Both agent briefings use the same English `faster-whisper` speech service and model setting (`ANISHA_WHISPER_MODEL`, default `small`) as ANISHA. The participant reviews or edits each transcript and the proposed goal before confirming. The frozen runner starts only after both confirmations, in either order.
- After briefing, each agent tab has a microphone for status, evidence, and pending-decision questions. It uses ANISHA's copied Whisper recognizer and exact `IntentExtractor` prompt/model setting (`OPENAI_MODEL`, default `gpt-5.6-luna`). Speech stays with the selected agent; a request explicitly targeting the other agent asks the participant to switch tabs. The recognized transcript can be edited and resubmitted. Spoken choices can suggest an on-screen option but never save it.
- The copied frozen scenario, trace, final artefacts, and counterbalancing file are local to this future Git repository. An adapted copy of the ANISHA study runner controls task state, event timing, source release, feedback, decisions, corrections, and safety. Every participant action stays in SQLite; no live Coding or Writing executor is called.
- Each agent has a dark conversation screen. Generated documents and decision controls remain in the chat; an openable Context panel holds the project brief and released reference notes, while Activity holds that agent's full chronological trace. The shared blocker/routine pair gets one cue; each item can be opened in either order. Context/Activity access, source and event opens, tab switches, cue delivery, first answers, corrections, and speech attempts are logged. The researcher can export the trial as JSON.
- At the masked state probe, the screen turns black. The researcher records the participant's verbatim response from `/`, then the trial resumes.

## Run locally

From this directory, use the same Python environment and Whisper model as ANISHA:

```sh
export ANISHA_STUDY_RESEARCHER_TOKEN='choose-a-local-researcher-token'
export ANISHA_WHISPER_MODEL=small
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8010
```

Open `http://127.0.0.1:8010/` on the researcher computer. Phone microphone access requires trusted HTTPS. For LAN phone testing, run Uvicorn with a trusted certificate and `--host 0.0.0.0 --ssl-certfile ... --ssl-keyfile ...`; use the same model and phone setup as ANISHA. Never put the researcher token in the participant link.

The SQLite database defaults to `data/trials.sqlite3`; override with `SEPARATE_AGENTS_DB_PATH`. Study material and application hashes are stored with each trial. Editing source or materials during a trial invalidates subsequent state reads; create a new trial after a code change.

## Direct Coding and Writing agents

`backend/agents`, `backend/tools`, `backend/llm`, `backend/memory`, and `backend/workspace` are byte-for-byte copies of the corresponding ANISHA modules. This includes the exact Coding and Writing system prompts, task prompt builders, tool schemas, model settings, safety limits, and report contracts. The built-in red-wine project files are copied too. `backend/live_agents.py` is the new direct task API layer; it does not rank attention or synthesize across agents.

Copy `backend/.env.example` to `backend/.env`, add your `OPENAI_API_KEY`, and use the same `ANISHA_*` and `OPENAI_MODEL` settings as ANISHA. The study's later spoken questions use this key to interpret intent while answers remain fixed to released trial facts. Direct agent tasks remain proposals until explicit confirmation. For example:

```sh
curl -X POST http://127.0.0.1:8010/agents/coding_agent/projects/red-wine-quality/tasks \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"coding-001","goal":"Inspect the wine analysis code","constraints":["Do not publish"]}'
curl -X POST http://127.0.0.1:8010/agents/coding_agent/projects/red-wine-quality/tasks/coding-001/confirm
curl http://127.0.0.1:8010/agents/coding_agent/projects/red-wine-quality/tasks/coding-001
```

Replace `coding_agent` with `writing_agent` and include `source_paths`, `target_path`, and `target_mode` for writing tasks. The same paths provide `GET` task lists, `POST .../{task_id}/cancel`, and Coding's exact `POST .../{task_id}/approvals/{approval_id}/approve` flow. The live task store is currently in memory, matching ANISHA's current agent service. A server restart clears it.

Measured `study-*` trial IDs are refused by these live routes. The thesis comparison continues to use fixed agent events and outputs; live model responses do not change a participant trial.

Run the backend flow check with:

```sh
python -B -m unittest discover -s backend/tests -v
node --test frontend/tests/chat_layout.test.js
python -B scripts/check_material_parity.py
python -B scripts/check_agent_parity.py
```

## Study integration still needed

This is a runnable first vertical slice, not a frozen measurement build. The separate project currently stores its own SQLite trial records. Before paired periods, connect both interfaces to one shared allocation/session export or perform a verified join by participant, slot, and period. The copied study materials and agent/voice modules must be compared byte-for-byte with ANISHA at freeze time. Validate actual iPhone and Android microphone capture and follow-up interpretation with the configured OpenAI key, cue perception, timing, legibility, and both wine variants before a measured run.
