import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main
from backend.live_agents import DirectAgentRuntime, build_context_memory
from backend.workspace.models import AgentOutcomeReport, TaskState, WorkspaceEvent


class CompletingAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.submitted = []

    async def submit(self, task, *, on_event, on_report):
        self.submitted.append(task)
        await on_event(WorkspaceEvent(
            event_id=f"started-{task.task_id}", project_id=task.project_id,
            task_id=task.task_id, actor=self.agent_id, state=TaskState.RUNNING,
            message="Agent started",
        ))
        await on_report(AgentOutcomeReport(
            report_id=f"report-{task.task_id}", project_id=task.project_id,
            task_id=task.task_id, agent_id=self.agent_id,
            outcome=TaskState.COMPLETED, summary="Done",
            recommended_next_action="Review the result.",
        ))

    async def cancel(self, task_id):
        raise AssertionError(f"Completed task {task_id} must not be cancelled")


class LiveAgentApiTest(unittest.TestCase):
    def setUp(self):
        self.agents = {name: CompletingAgent(name) for name in ("coding_agent", "writing_agent")}
        self.runtime = DirectAgentRuntime(
            context_memory=build_context_memory(), agent_factory=lambda name: self.agents[name]
        )
        self.patcher = patch.object(main, "get_live_runtime", return_value=self.runtime)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.client = TestClient(main.app)

    def test_direct_routes_keep_agents_separate_and_preserve_context(self):
        for agent in self.agents:
            base = f"/agents/{agent}/projects/red-wine-quality/tasks"
            response = self.client.post(base, json={
                "task_id": f"{agent}-1", "goal": "Analyze the wine project",
                "constraints": ["Do not publish"],
            })
            self.assertEqual(response.status_code, 200, response.text)
            proposal = response.json()
            self.assertEqual(proposal["task"]["state"], "proposed")
            self.assertEqual(proposal["task"]["assigned_agent"], agent)
            self.assertIsNotNone(proposal["task"]["context_bundle"])
            self.assertEqual(self.client.get(base).json()[0]["task_id"], f"{agent}-1")
            self.assertEqual(self.client.post(base + f"/{agent}-1/confirm").status_code, 202)
            details = self.client.get(base + f"/{agent}-1").json()
            self.assertEqual(details["task"]["state"], "completed")
            self.assertEqual(details["reports"][0]["agent_id"], agent)
            self.assertEqual(self.agents[agent].submitted[0].context_bundle.query.agent_id, agent)
            self.assertEqual(self.client.post(base, json={
                "task_id": f"{agent}-1", "goal": "Duplicate",
            }).status_code, 409)
        self.assertEqual(self.client.get(
            "/agents/writing_agent/projects/red-wine-quality/tasks/coding_agent-1"
        ).status_code, 404)

    def test_study_trial_ids_cannot_run_live_agents(self):
        response = self.client.post(
            "/agents/coding_agent/projects/study-123/tasks",
            json={"goal": "Run code"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.runtime.workspace.list_tasks("study-123"), [])


if __name__ == "__main__":
    unittest.main()
