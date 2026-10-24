import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main
from backend.study.runner import StudyRunner
from backend.study.setup import StudySetupStore
from backend.voice.intent import ExtractedIntent, Intent, IntentParameters, RequestedInformation


class FakeSpeech:
    model_size = "small"

    async def transcribe_async(self, _audio):
        return SimpleNamespace(text="Please work on my wine task")


class FakeIntentExtractor:
    model = "gpt-5.6-luna"

    async def extract(self, text):
        if "other agent" in text:
            return ExtractedIntent(intent=Intent.QUERY_STATUS, target_agent="writing_agent")
        if "evidence" in text:
            return ExtractedIntent(intent=Intent.REQUEST_DETAILS, target_agent="coding_agent",
                                   parameters=IntentParameters(requested_information=RequestedInformation.EVIDENCE))
        if text.startswith("choose "):
            return ExtractedIntent(intent=Intent.APPROVE_ACTION, target_agent="coding_agent",
                                   parameters=IntentParameters(action_id=text.removeprefix("choose ")))
        if "approve" in text:
            return ExtractedIntent(intent=Intent.APPROVE_ACTION, target_agent="coding_agent",
                                   parameters=IntentParameters(action_id="approve"))
        return ExtractedIntent(intent=Intent.QUERY_STATUS)


class ConditionCFlowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clock = [1_000_000]
        self.store = StudySetupStore(Path(self.temp.name) / "study.sqlite3")
        self.runner = StudyRunner(self.store, clock=lambda: self.clock[0])
        self.patches = [
            patch.object(main, "store", self.store),
            patch.object(main, "runner", self.runner),
            patch.object(main, "speech_service", FakeSpeech()),
            patch.dict(os.environ, {"ANISHA_STUDY_RESEARCHER_TOKEN": "research-test"}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.client = TestClient(main.app)
        self.researcher = {"X-ANISHA-Researcher-Token": "research-test"}

    def call(self, method, path, *, headers=None, **kwargs):
        response = self.client.request(method, path, headers=headers or self.participant, **kwargs)
        self.assertLess(response.status_code, 400, response.text)
        return response.json()

    def test_two_briefings_share_one_trace_and_gate(self):
        created = self.call("POST", "/study/researcher/trials", headers=self.researcher,
                            json={"participant_id": "p01", "slot": "02", "period": 1})
        trial_id = created["trial"]["trial_id"]
        self.participant = {"X-ANISHA-Study-Token": main.participant_token(trial_id)}
        base = f"/study/trials/{trial_id}"
        self.assertEqual(self.call("GET", base + "/state")["events"], [])

        for agent in ("writing_agent", "coding_agent"):
            activation = self.call("POST", f"{base}/agents/{agent}/speech/activations")
            heard = self.call("POST", f"{base}/agents/{agent}/speech",
                              params={"activation_id": activation["transcription_id"]},
                              content=b"a" * 64)
            self.assertEqual(heard["status"], "recognized")
            proposal = self.call("POST", f"{base}/agents/{agent}/proposals", json={
                "transcription_id": heard["transcription_id"],
                "reviewed_transcript": f"Please brief {agent}",
            })
            state = self.call("POST", f"{base}/agents/{agent}/proposals/{proposal['proposal_id']}/confirm",
                              json={"goal": f"Edited {agent} goal"})
            if agent == "writing_agent":
                self.assertEqual(state["phase"], "awaiting_briefing")
                self.assertEqual(state["events"], [])
            else:
                self.assertEqual(state["phase"], "running")
                self.assertEqual(len(state["tasks"]), 2)
                self.assertEqual({task["goal"] for task in state["tasks"]}, {
                    "Edited coding_agent goal", "Edited writing_agent goal"})

        self.clock[0] += 220_000
        masked = self.call("GET", base + "/state")
        self.assertTrue(masked["masked"])
        self.assertEqual(masked["events"], [])
        self.call("POST", f"/study/researcher/trials/{trial_id}/probe", headers=self.researcher,
                  json={"reason": "participant state answer"})
        self.clock[0] += 45_000
        pair = self.call("GET", base + "/state")
        self.assertEqual(pair["phase"], "blocker")
        paired = [event for event in pair["events"] if event["cue_id"]]
        self.assertEqual(len(paired), 2)
        self.assertEqual(paired[0]["cue_id"], paired[1]["cue_id"])
        self.assertEqual({event["task_id"].split(':')[-1] for event in paired}, {"coding", "writing"})
        self.assertEqual(self.call("POST", base + "/interactions", json={
            "event_type": "cue_delivered", "target_id": paired[0]["cue_id"], "ui_layer": "agent_tabs",
        })["is_new"], True)
        self.assertEqual(self.call("POST", base + "/interactions", json={
            "event_type": "cue_delivered", "target_id": paired[0]["cue_id"], "ui_layer": "agent_tabs",
        })["is_new"], False)
        answer = self.call("POST", base + "/responses", json={
            "choice_id": pair["action_options"][1]["choice_id"], "reason": "My first choice", "confidence": 60,
        })
        self.assertEqual(answer["phase"], "awaiting_approval")
        if answer["correction_options"]:
            corrected = self.call("POST", base + "/corrections", json={
                "choice_id": answer["correction_options"][0]["choice_id"],
                "reason": "I checked the newer instruction",
            })
            self.assertEqual(corrected["phase"], "awaiting_approval")
        self.clock[0] += 90_000
        approval = self.call("GET", base + "/state")
        self.assertEqual(approval["phase"], "approval")
        self.assertTrue(any(record["kind"] == "draft_preview" for record in approval["records"]))
        self.call("POST", base + "/responses", json={
            "choice_id": "reject", "reason": "Do not change the draft", "confidence": 90,
        })
        self.clock[0] += 120_000
        final = self.call("GET", base + "/state")
        self.assertEqual(final["phase"], "final_review")
        self.assertEqual(sum(record["kind"] in {"code_artifact", "markdown_artifact"}
                             for record in final["records"]), 2)
        done = self.call("POST", base + "/responses", json={
            "choice_id": final["action_options"][0]["choice_id"],
            "reason": "Review completed", "confidence": 80, "claim_text": "Flagged claim",
        })
        self.assertEqual(done["phase"], "complete")
        export = self.call("GET", f"/study/researcher/trials/{trial_id}/export", headers=self.researcher)
        self.assertEqual(len(export["condition_c_voice_attempts"]), 2)
        self.assertEqual(len(export["condition_c_proposals"]), 2)
        self.assertEqual(len([row for row in export["responses"] if row["response_order"] == 1]), 4)

    def test_second_wine_variant_starts_after_both_agents(self):
        created = self.call("POST", "/study/researcher/trials", headers=self.researcher,
                            json={"participant_id": "p03", "slot": "04", "period": 1})
        trial_id = created["trial"]["trial_id"]
        self.assertEqual(created["trial"]["wine_variant"], "wine_2")
        first = self.runner.confirm_briefing(trial_id, main.BriefingConfirmation(
            agent_id="coding_agent", reviewed_transcript="Coding first", goal="Coding variant two"))
        self.assertEqual(first.events, [])
        second = self.runner.confirm_briefing(trial_id, main.BriefingConfirmation(
            agent_id="writing_agent", reviewed_transcript="Writing second", goal="Writing variant two"))
        self.assertEqual(second.phase, "running")
        self.assertEqual(len(second.events), 1)
        self.assertEqual({task.goal for task in second.tasks}, {"Coding variant two", "Writing variant two"})

    def test_single_digit_slot_is_normalized_before_allocation(self):
        created = self.call("POST", "/study/researcher/trials", headers=self.researcher,
                            json={"participant_id": "p-slot", "slot": "2", "period": 1})
        self.assertEqual(created["trial"]["slot"], "02")
        self.assertEqual(created["trial"]["condition"], "condition_c")

    def test_selected_agent_followup_uses_same_speech_and_logs_repairs(self):
        created = self.call("POST", "/study/researcher/trials", headers=self.researcher,
                            json={"participant_id": "p04", "slot": "02", "period": 1})
        trial_id = created["trial"]["trial_id"]
        self.participant = {"X-ANISHA-Study-Token": main.participant_token(trial_id)}
        base = f"/study/trials/{trial_id}"
        for agent in ("coding_agent", "writing_agent"):
            self.runner.confirm_briefing(trial_id, main.BriefingConfirmation(
                agent_id=agent, reviewed_transcript=agent, goal=f"{agent} goal"))
        with patch.object(main, "get_intent_extractor", return_value=FakeIntentExtractor()):
            activation = self.call("POST", f"{base}/agents/coding_agent/speech/activations",
                                   params={"purpose": "followup"})
            heard = self.call("POST", f"{base}/agents/coding_agent/speech",
                              params={"purpose": "followup", "activation_id": activation["transcription_id"]},
                              content=b"a" * 64)
            self.assertEqual(heard["speech_service_version"], "small")
            answer = self.call("POST", f"{base}/agents/coding_agent/followups",
                               json={"transcription_id": heard["transcription_id"]})
            self.assertEqual(answer["agent_id"], "coding_agent")
            self.assertEqual(answer["request_kind"], "status")
            self.assertIn("Coding is running", answer["message"])
            mismatch = self.call("POST", f"{base}/agents/coding_agent/followups", json={
                "transcription_id": heard["transcription_id"],
                "reviewed_transcript": "How is the other agent?",
            })
            self.assertEqual(mismatch["status"], "needs_clarification")
            self.assertNotIn("Writing is", mismatch["message"])
            export = self.call("GET", f"/study/researcher/trials/{trial_id}/export", headers=self.researcher)
            self.assertEqual([row["purpose"] for row in export["condition_c_voice_attempts"]], ["followup"])
            self.assertEqual([row["version_kind"] for row in export["condition_c_followups"]],
                             ["used_as_recognized", "reviewed"])
            self.assertEqual(export["condition_c_followups"][0]["intent_model"], "gpt-5.6-luna")
            self.assertEqual(self.call("GET", base + "/state")["phase"], "running")

    def test_followup_microphone_hidden_at_probe(self):
        created = self.call("POST", "/study/researcher/trials", headers=self.researcher,
                            json={"participant_id": "p05", "slot": "02", "period": 1})
        trial_id = created["trial"]["trial_id"]
        self.participant = {"X-ANISHA-Study-Token": main.participant_token(trial_id)}
        for agent in ("coding_agent", "writing_agent"):
            self.runner.confirm_briefing(trial_id, main.BriefingConfirmation(
                agent_id=agent, reviewed_transcript=agent))
        self.clock[0] += 220_000
        response = self.client.post(
            f"/study/trials/{trial_id}/agents/coding_agent/speech/activations?purpose=followup",
            headers=self.participant,
        )
        self.assertEqual(response.status_code, 422)

    def test_selected_agent_voice_reveals_only_its_events_and_never_saves_a_choice(self):
        created = self.call("POST", "/study/researcher/trials", headers=self.researcher,
                            json={"participant_id": "p06", "slot": "02", "period": 1})
        trial_id = created["trial"]["trial_id"]
        self.participant = {"X-ANISHA-Study-Token": main.participant_token(trial_id)}
        base = f"/study/trials/{trial_id}"
        for agent in ("coding_agent", "writing_agent"):
            self.runner.confirm_briefing(trial_id, main.BriefingConfirmation(
                agent_id=agent, reviewed_transcript=agent))
        self.clock[0] += 220_000
        self.call("GET", base + "/state")
        self.call("POST", f"/study/researcher/trials/{trial_id}/probe", headers=self.researcher,
                  json={"reason": "probe response"})
        self.clock[0] += 45_000
        pair = self.call("GET", base + "/state")
        self.assertEqual(pair["phase"], "blocker")
        coding_event = next(event for event in reversed(pair["events"])
                            if (event["task_id"] or "").endswith(":coding"))
        writing_event = next(event for event in reversed(pair["events"])
                             if (event["task_id"] or "").endswith(":writing"))
        choice = pair["action_options"][0]["choice_id"]

        with patch.object(main, "get_intent_extractor", return_value=FakeIntentExtractor()):
            activation = self.call("POST", f"{base}/agents/coding_agent/speech/activations",
                                   params={"purpose": "followup"})
            wrong_purpose = self.client.post(
                f"{base}/agents/coding_agent/speech",
                params={"activation_id": activation["transcription_id"]},
                headers=self.participant, content=b"a" * 64,
            )
            self.assertEqual(wrong_purpose.status_code, 422)
            heard = self.call("POST", f"{base}/agents/coding_agent/speech",
                              params={"purpose": "followup", "activation_id": activation["transcription_id"]},
                              content=b"a" * 64)
            speech_id = heard["transcription_id"]
            status = self.call("POST", f"{base}/agents/coding_agent/followups",
                               json={"transcription_id": speech_id})
            self.assertIn(coding_event["text"], status["message"])
            self.assertNotIn(writing_event["text"], status["message"])
            self.assertEqual(status["revealed_event_id"], coding_event["event_id"])
            evidence = self.call("POST", f"{base}/agents/coding_agent/followups", json={
                "transcription_id": speech_id, "reviewed_transcript": "Show coding evidence",
            })
            visible = {record["record_id"]: record for record in pair["records"]}
            self.assertTrue(evidence["record_ids"])
            self.assertTrue(all(visible[record_id]["kind"] in {"source", "instruction", "code_artifact"}
                                for record_id in evidence["record_ids"]))
            action = self.call("POST", f"{base}/agents/coding_agent/followups", json={
                "transcription_id": speech_id, "reviewed_transcript": f"choose {choice}",
            })
            self.assertEqual(action["suggested_choice_id"], choice)
            self.assertIn("save your choice on screen", action["message"])

        export = self.call("GET", f"/study/researcher/trials/{trial_id}/export", headers=self.researcher)
        opened = {row["target_id"] for row in export["interactions"] if row["event_type"] == "event_open"}
        self.assertIn(coding_event["event_id"], opened)
        self.assertNotIn(writing_event["event_id"], opened)
        self.assertFalse(any(row["gate_event_id"] == coding_event["event_id"]
                             for row in export["responses"]))

    def test_rejects_other_condition_and_missing_token(self):
        response = self.client.post("/study/researcher/trials", headers=self.researcher,
                                    json={"participant_id": "p02", "slot": "01", "period": 1})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(list(Path(self.temp.name).glob('*.sqlite3')), [])


if __name__ == "__main__":
    unittest.main()
