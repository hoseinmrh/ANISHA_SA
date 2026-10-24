"""Structured intent extraction for transcribed ANISHA voice commands.

The extractor converts free-form speech into the small routing contract used by
the orchestrator: an intent, an optional target agent, and typed parameters.
It deliberately does not execute actions or grant approval; it only reports
what the user explicitly said.
"""

from __future__ import annotations

import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

from dotenv import load_dotenv
from openai import AsyncOpenAI, pydantic_function_tool
from pydantic import BaseModel, ConfigDict, Field, field_validator


BACKEND_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
DEFAULT_MODEL = "gpt-5.6-luna"


class Intent(str, Enum):
    """Voice operations understood by the ANISHA orchestrator and UI."""

    DELEGATE_TASK = "delegate_task"
    QUERY_STATUS = "query_status"
    REQUEST_DETAILS = "request_details"
    APPROVE_ACTION = "approve_action"
    CORRECT_TASK = "correct_task"
    PAUSE_TASK = "pause_task"
    CANCEL_TASK = "cancel_task"
    DISMISS_OVERLAY = "dismiss_overlay"
    UNKNOWN = "unknown"


class TargetAgent(str, Enum):
    """Convenience constants for planned and built-in agent identifiers.

    ``ExtractedIntent.target_agent`` is intentionally a validated string so
    runtime extensions are not limited to this enumeration.
    """

    CODING = "coding_agent"
    COMMUNICATION = "communication_agent"
    RESEARCH = "research_agent"
    WRITING = "writing_agent"


class RequestedInformation(str, Enum):
    """Progressive-disclosure information a user may ask to see."""

    SUMMARY = "summary"
    ATTENTION = "attention"
    EVIDENCE = "evidence"
    HISTORY = "history"


class IntentParameters(BaseModel):
    """Information needed to dispatch or handle an extracted intent."""

    model_config = ConfigDict(extra="forbid")

    goal: Optional[str] = Field(
        default=None,
        description=(
            "The primary delegated action, stated as a concise command without its "
            "acceptance criteria or operating constraints."
        ),
    )
    constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit requirements, acceptance criteria, limits, and prohibitions, "
            "such as 'the tests pass', 'use Python', or 'do not merge'."
        ),
    )
    source_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit workspace-relative text files the writing agent should read "
            "as source material. Files are read-only and must be inside the workspace."
        ),
    )
    context: Optional[str] = Field(
        default=None,
        description="Referenced project, meeting, repository, document, or deadline context.",
    )
    target_path: Optional[str] = Field(
        default=None,
        description=(
            "Exact workspace-relative Markdown destination when the user explicitly "
            "names where a writing task should create or edit a document; otherwise null."
        ),
    )
    target_mode: Literal["create", "replace", "append"] = Field(
        default="replace",
        description=(
            "For a target_path, create a new document, replace an existing document "
            "with a full revision, or append new material while preserving it."
        ),
    )
    finish_after_first_section: bool = Field(
        default=False,
        description=(
            "Deprecated compatibility flag. The writing agent now completes every "
            "usable draft response without a section-by-section loop."
        ),
    )
    task_id: Optional[str] = Field(
        default=None,
        description="An explicitly referenced task identifier, when present.",
    )
    action_id: Optional[str] = Field(
        default=None,
        description="The pending action or presented option the user selected.",
    )
    correction: Optional[str] = Field(
        default=None,
        description="The user's replacement or corrective instruction.",
    )
    requested_information: Optional[RequestedInformation] = Field(
        default=None,
        description="The state layer requested by a status or detail query.",
    )
    clarification_needed: bool = Field(
        default=False,
        description="Whether safe routing requires more information from the user.",
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="A single short question to resolve an ambiguous command.",
    )

    @field_validator(
        "goal",
        "context",
        "target_path",
        "task_id",
        "action_id",
        "correction",
        "clarification_question",
    )
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("constraints", "source_paths")
    @classmethod
    def normalize_text_list(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class ExtractedIntent(BaseModel):
    """Validated output returned by the intent extractor."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    target_agent: Optional[str] = Field(
        default=None,
        description="The agent addressed by the user, or null for all-agent/UI commands.",
    )
    parameters: IntentParameters = Field(default_factory=IntentParameters)

    @field_validator("target_agent")
    @classmethod
    def normalize_target_agent(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class IntentToolArguments(BaseModel):
    """Arguments shared by the router's native intent tools."""

    model_config = ConfigDict(extra="forbid")

    target_agent: Optional[str] = Field(
        default=None,
        description="The selected runtime agent, or null for UI-wide operations.",
    )
    parameters: IntentParameters = Field(default_factory=IntentParameters)


class ProjectBundleToolArguments(BaseModel):
    """One spoken briefing for both existing Coding and Writing agents."""

    model_config = ConfigDict(extra="forbid")

    coding_goal: str = Field(description="What Coding should do, in the participant's words.")
    writing_goal: str = Field(description="What Writing should do, in the participant's words.")
    clarification_question: Optional[str] = Field(
        default=None,
        description="One short question when either task is missing or unclear.",
    )

    @field_validator("coding_goal", "writing_goal")
    @classmethod
    def normalize_goal(cls, value: str) -> str:
        return value.strip()


ROUTING_TOOL_DESCRIPTIONS = {
    Intent.DELEGATE_TASK: "Create new work for one selected specialist agent.",
    Intent.QUERY_STATUS: "Query current project or task status.",
    Intent.REQUEST_DETAILS: "Request evidence, provenance, history, or an explanation.",
    Intent.APPROVE_ACTION: "Identify an explicit approval request for UI confirmation.",
    Intent.CORRECT_TASK: "Apply a corrective instruction to an existing task.",
    Intent.PAUSE_TASK: "Pause an existing task.",
    Intent.CANCEL_TASK: "Cancel an existing task.",
    Intent.DISMISS_OVERLAY: "Dismiss the visible overlay without approving an action.",
    Intent.UNKNOWN: "Report that the command cannot be routed safely.",
}


SYSTEM_PROMPT = """You are ANISHA's intent router for a supervised, multi-agent
university-work assistant. Select exactly one supplied routing tool and populate
its arguments from the user's transcribed voice command.

Route delegated work to exactly one agent from the runtime catalogue appended to
these instructions. Never invent an agent identifier that is absent from it.

Intent rules:
- delegate_task creates new work.
- query_status asks what is happening or what needs attention.
- request_details asks for evidence, provenance, history, or an explanation.
- approve_action identifies an explicit approval request, but voice routing never
  executes that consequential action. The user must confirm with the visible UI
  control. Never infer approval from vague agreement.
- correct_task changes an existing instruction.
- pause_task and cancel_task must remain distinct.
- dismiss_overlay covers acknowledgements such as "OK" or "got it" only when
  they dismiss the interface, not when they approve an action.
- unknown is required when the command cannot be routed safely.

Clarification policy:
- Ask only when execution is genuinely blocked by a missing user-owned decision:
  the requested outcome or target agent is unclear, explicit requirements
  conflict, an existing external target cannot be identified, or proceeding
  could cause a materially different scope, safety, or user-visible result.
- The project route selects the configured workspace. Never ask a coding-task
  user which repository, workspace, or absolute path to use. Relative locations
  such as "the examples directory" are resolved inside that workspace.
- Coding agents own ordinary implementation decisions when the user did not
  constrain them: language/runtime versions, libraries, frameworks, packages,
  dependency and package managers, build/test tools, file names, directory
  layout, project structure, and coding style. Leave these choices to the
  assigned coding agent instead of asking or inventing a user constraint.
- Phrases such as "choose the best", "you decide", "use an appropriate one",
  "as needed", or "whatever is suitable" explicitly delegate the choice. Never
  ask the user to make that choice again.
- A preference that could improve the result is not a blocker. Clarification is
  for information without which the task cannot be routed or safely attempted.

Separate the primary action from every explicit requirement. Put only the action
in parameters.goal. Preserve requirements verbatim in parameters.constraints,
including:
- acceptance or success criteria ("so the tests pass", "ensure it compiles")
- required tools, formats, scope, quality bars, and deadlines
- negative constraints ("do not merge", "without changing the API", "draft only")

For example, "Fix the code inside calculator.py so it passes the test" has goal
"Fix the code inside calculator.py" and constraint "it passes the test". Never
leave an explicit acceptance criterion only inside the goal.

Put referenced background in parameters.context, explicit task identifiers in
parameters.task_id, and corrections in parameters.correction. For ambiguous
agent, action, or task references, set
clarification_needed to true and provide one concise clarification question.
Do not invent missing context, action identifiers, permission, or approval.
For a writing task with an explicitly named output or edit path, put the exact
workspace-relative Markdown path in parameters.target_path. Use target_mode=create
when the user asks to create, write, or save a new named file; use replace or append
only when the user asks to modify an existing document. Leave target_path null when
the user requests a draft but does not name its destination.
"""


CLARIFICATION_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You are resolving a pending clarification, not routing the reply as a new,
standalone command. The input contains the original transcript, the current
structured intent, the question ANISHA asked, and the user's spoken reply.
Return one complete revised ExtractedIntent.

- Preserve the original action, requirements, context, and explicitly supplied
  values unless the reply directly corrects them.
- Use the reply only to fill in or correct the information requested by the
  clarification question.
- A short answer such as "Python" or "the first one" inherits the original
  request; it is not an unknown standalone intent.
- Never interpret "yes", "okay", or similar acknowledgement as approval for a
  consequential action.
- If the reply is still insufficient or introduces a conflict, keep
  clarification_needed true and ask one updated, concise question.
- Otherwise set clarification_needed false and clarification_question null.
"""


TRAILING_SUCCESS_CRITERION = re.compile(
    r"^(?P<goal>.+?)(?:,\s*|\s+)so(?:\s+that)?\s+(?P<constraint>.+?)\s*[.!?]*$",
    re.IGNORECASE,
)
CONSTRAINT_STOP_WORDS = {
    "a",
    "all",
    "an",
    "it",
    "must",
    "should",
    "so",
    "that",
    "the",
}
CONSTRAINT_TOKEN_NORMALIZATION = {
    "passed": "pass",
    "passes": "pass",
    "passing": "pass",
    "tests": "test",
}
DELEGATED_CHOICE_PATTERN = re.compile(
    r"\b(?:you\s+(?:can\s+)?(?:choose|decide|think)|choose\s+(?:the\s+)?best|"
    r"use\s+(?:an?\s+)?appropriate|as\s+needed|up\s+to\s+you|whatever\s+is\s+"
    r"suitable)\b",
    re.IGNORECASE,
)
WORKSPACE_CLARIFICATION_PATTERN = re.compile(
    r"\b(?:repository|repo|workspace|working\s+directory|absolute\s+path|"
    r"project\s+path)\b",
    re.IGNORECASE,
)
IMPLEMENTATION_CLARIFICATION_PATTERN = re.compile(
    r"\b(?:language|runtime|version|library|framework|package|toolkit|"
    r"dependency\s+manager|package\s+manager|packaging\s+tool|build\s+tool|"
    r"test(?:ing)?\s+(?:tool|framework)|file\s+name|directory\s+name|folder\s+"
    r"name|directory\s+layout|folder\s+layout|project\s+structure|coding\s+"
    r"style)\b",
    re.IGNORECASE,
)
BLOCKING_CLARIFICATION_PATTERN = re.compile(
    r"\b(?:conflict|contradict|which\s+existing|recipient|account|credential|"
    r"production|deployment\s+target|external\s+service|delete|overwrite|"
    r"irreversible)\b",
    re.IGNORECASE,
)
NAMED_DOCUMENT_CREATE_PATTERN = re.compile(
    r"\b(?:create|draft|prepare|produce|save|write)\b",
    re.IGNORECASE,
)
NAMED_DOCUMENT_EDIT_PATTERN = re.compile(
    r"\b(?:append|edit|existing|modify|replace|revis(?:e|ed|ion)|rewrite|"
    r"update|add\s+to)\b",
    re.IGNORECASE,
)


class IntentExtractionError(RuntimeError):
    """Raised when the model cannot produce a usable structured intent."""


class IntentExtractor:
    """Async OpenAI-backed intent extraction service.

    Configuration is loaded from ``backend/.env`` without overriding variables
    already set by the process. A client may be injected for tests.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        env_file: Path = BACKEND_ENV_FILE,
        agent_catalog: Optional[Sequence[Mapping[str, Any] | Any]] = None,
    ) -> None:
        load_dotenv(dotenv_path=env_file, override=False)

        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        if not self.model.strip():
            raise ValueError("OPENAI_MODEL must not be empty")

        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    f"OPENAI_API_KEY is missing. Add it to {env_file} or export it "
                    "in the process environment."
                )
            client = AsyncOpenAI(api_key=api_key)

        self._client = client
        self._agent_catalog = _normalize_agent_catalog(agent_catalog)

    async def extract(self, text: str) -> ExtractedIntent:
        """Extract and validate routing information from transcribed speech."""

        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("text must not be empty")

        try:
            response = await self._client.responses.parse(
                model=self.model,
                instructions=_routing_instructions(
                    SYSTEM_PROMPT, self._agent_catalog
                ),
                input=normalized_text,
                tools=_routing_tools(),
                tool_choice="required",
                parallel_tool_calls=False,
                max_tool_calls=1,
            )
        except Exception as exc:
            raise IntentExtractionError("OpenAI intent extraction failed") from exc

        result = _intent_from_response(response)
        if result is None:
            raise IntentExtractionError(
                "OpenAI returned no routing tool call "
                "(the response may have been refused or incomplete)"
            )
        result = _preserve_explicit_success_criteria(normalized_text, result)
        result = _normalize_named_document_creation(normalized_text, result)
        return _suppress_nonblocking_coding_clarification(normalized_text, result)

    async def extract_project_bundle(self, text: str) -> ProjectBundleToolArguments:
        """Use the same intent model to understand one Coding and Writing brief."""

        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("text must not be empty")
        try:
            response = await self._client.responses.parse(
                model=self.model,
                instructions=(
                    "Extract the two tasks from a natural spoken ANISHA project brief. "
                    "Coding checks or analyses data; Writing drafts or revises a report. "
                    "Accept paraphrases and implicit role names when the two jobs are clear. "
                    "Return empty text for a missing job and ask one short clarification. "
                    "Do not claim any work has started or finished."
                ),
                input=normalized_text,
                tools=[pydantic_function_tool(
                    ProjectBundleToolArguments,
                    name="delegate_project_bundle",
                    description="Understand a combined Coding and Writing request.",
                )],
                tool_choice="required",
                parallel_tool_calls=False,
                max_tool_calls=1,
            )
        except Exception as exc:
            raise IntentExtractionError("OpenAI project briefing extraction failed") from exc
        for item in getattr(response, "output", ()):
            if getattr(item, "type", None) != "function_call" or getattr(item, "name", None) != "delegate_project_bundle":
                continue
            try:
                parsed = getattr(item, "parsed_arguments", None)
                return ProjectBundleToolArguments.model_validate(
                    parsed if parsed is not None else json.loads(item.arguments)
                )
            except (TypeError, ValueError, AttributeError) as exc:
                raise IntentExtractionError("Invalid project briefing extraction") from exc
        raise IntentExtractionError("OpenAI returned no project briefing tool call")

    async def resolve_clarification(
        self,
        *,
        original_text: str,
        current_intent: ExtractedIntent,
        question: str,
        reply: str,
    ) -> ExtractedIntent:
        """Resolve a short reply against its retained clarification context."""

        normalized_reply = reply.strip()
        if not normalized_reply:
            raise ValueError("reply must not be empty")
        payload = json.dumps(
            {
                "original_transcript": original_text,
                "current_intent": current_intent.model_dump(mode="json"),
                "clarification_question": question,
                "user_reply": normalized_reply,
            },
            ensure_ascii=False,
        )
        try:
            response = await self._client.responses.parse(
                model=self.model,
                instructions=_routing_instructions(
                    CLARIFICATION_SYSTEM_PROMPT, self._agent_catalog
                ),
                input=payload,
                tools=_routing_tools(),
                tool_choice="required",
                parallel_tool_calls=False,
                max_tool_calls=1,
            )
        except Exception as exc:
            raise IntentExtractionError(
                "OpenAI clarification resolution failed"
            ) from exc

        result = _intent_from_response(response)
        if result is None:
            raise IntentExtractionError(
                "OpenAI returned no clarification routing tool call"
            )
        result = _preserve_explicit_success_criteria(original_text, result)
        result = _normalize_named_document_creation(
            f"{original_text}\n{normalized_reply}", result
        )
        return _suppress_nonblocking_coding_clarification(
            f"{original_text}\n{normalized_reply}", result
        )


def _routing_tools() -> list[dict[str, Any]]:
    """Expose each safe orchestrator intent as one native function tool."""

    return [
        pydantic_function_tool(
            IntentToolArguments,
            name=intent.value,
            description=ROUTING_TOOL_DESCRIPTIONS[intent],
        )
        for intent in Intent
    ]


def _intent_from_response(response: Any) -> Optional[ExtractedIntent]:
    """Normalize a selected routing tool into the public intent contract."""

    for item in getattr(response, "output", ()):
        if getattr(item, "type", None) != "function_call":
            continue
        try:
            intent = Intent(getattr(item, "name"))
            parsed = getattr(item, "parsed_arguments", None)
            if isinstance(parsed, IntentToolArguments):
                arguments = parsed
            elif parsed is not None:
                arguments = IntentToolArguments.model_validate(parsed)
            else:
                arguments = IntentToolArguments.model_validate_json(item.arguments)
        except (AttributeError, TypeError, ValueError):
            continue
        return ExtractedIntent(
            intent=intent,
            target_agent=arguments.target_agent,
            parameters=arguments.parameters,
        )

    # Retain compatibility with injected clients and replay fixtures created
    # before routing moved from text structured output to native function tools.
    parsed_output = getattr(response, "output_parsed", None)
    if isinstance(parsed_output, ExtractedIntent):
        return parsed_output
    return None


def _normalize_named_document_creation(
    source_text: str, result: ExtractedIntent
) -> ExtractedIntent:
    """Recover a high-confidence named-create request from the transcript."""

    parameters = result.parameters
    if (
        result.intent != Intent.DELEGATE_TASK
        or result.target_agent != TargetAgent.WRITING
        or parameters.target_path is None
        or not NAMED_DOCUMENT_CREATE_PATTERN.search(source_text)
        or NAMED_DOCUMENT_EDIT_PATTERN.search(source_text)
    ):
        return result
    return result.model_copy(
        update={
            "parameters": parameters.model_copy(update={"target_mode": "create"})
        }
    )


def _preserve_explicit_success_criteria(
    source_text: str, result: ExtractedIntent
) -> ExtractedIntent:
    """Recover an explicit trailing success clause omitted by the model.

    The model remains responsible for general language understanding. This
    deterministic guard only handles a high-confidence ``... so [that] ...``
    construction and never adds a condition that was not present in the source.
    """

    if result.intent != Intent.DELEGATE_TASK:
        return result
    source_parts = _split_success_criterion(source_text)
    if source_parts is None:
        return result

    _source_goal, source_constraint = source_parts
    constraints = list(result.parameters.constraints)
    if not any(
        _constraints_equivalent(source_constraint, existing)
        for existing in constraints
    ):
        constraints.append(source_constraint)

    goal = result.parameters.goal
    if goal is not None:
        goal_parts = _split_success_criterion(goal)
        if goal_parts is not None and _constraints_equivalent(
            source_constraint, goal_parts[1]
        ):
            goal = goal_parts[0]

    parameters = result.parameters.model_copy(
        update={"goal": goal, "constraints": constraints}
    )
    return result.model_copy(update={"parameters": parameters})


def _split_success_criterion(text: str) -> Optional[tuple[str, str]]:
    match = TRAILING_SUCCESS_CRITERION.match(text.strip())
    if match is None:
        return None
    goal = match.group("goal").rstrip(" ,;:")
    constraint = match.group("constraint").rstrip(" .!?")
    if not goal or not constraint:
        return None
    return goal, constraint


def _constraints_equivalent(first: str, second: str) -> bool:
    first_tokens = _constraint_tokens(first)
    second_tokens = _constraint_tokens(second)
    if not first_tokens or not second_tokens:
        return first.strip().casefold() == second.strip().casefold()
    overlap = len(first_tokens & second_tokens)
    return overlap / min(len(first_tokens), len(second_tokens)) >= 0.8


def _constraint_tokens(value: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    return {
        CONSTRAINT_TOKEN_NORMALIZATION.get(token, token)
        for token in tokens
        if token not in CONSTRAINT_STOP_WORDS
    }


def _suppress_nonblocking_coding_clarification(
    source_text: str, result: ExtractedIntent
) -> ExtractedIntent:
    """Keep routine implementation choices with the configured coding agent."""

    parameters = result.parameters
    if (
        result.intent != Intent.DELEGATE_TASK
        or result.target_agent != TargetAgent.CODING.value
        or parameters.goal is None
        or not parameters.clarification_needed
    ):
        return result

    question = parameters.clarification_question or ""
    if BLOCKING_CLARIFICATION_PATTERN.search(question):
        return result

    choice_was_delegated = DELEGATED_CHOICE_PATTERN.search(source_text) is not None
    asks_for_runtime_owned_workspace = (
        WORKSPACE_CLARIFICATION_PATTERN.search(question) is not None
    )
    asks_for_implementation_preference = (
        IMPLEMENTATION_CLARIFICATION_PATTERN.search(question) is not None
    )
    if not (
        choice_was_delegated
        or asks_for_runtime_owned_workspace
        or asks_for_implementation_preference
    ):
        return result

    updated = parameters.model_copy(
        update={
            "clarification_needed": False,
            "clarification_question": None,
        }
    )
    return result.model_copy(update={"parameters": updated})


def _normalize_agent_catalog(
    catalog: Optional[Sequence[Mapping[str, Any] | Any]],
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    if catalog is None:
        return (
            (
                TargetAgent.CODING.value,
                "Source code, tests, repositories, and debugging.",
                (),
            ),
            (
                TargetAgent.WRITING.value,
                "Documents, reports, summaries, notes, and rewriting.",
                (),
            ),
        )

    normalized = []
    for item in catalog:
        if isinstance(item, Mapping):
            agent_id = str(item.get("agent_id", "")).strip()
            description = str(item.get("description", "")).strip()
            capabilities = item.get("capabilities", ())
        else:
            agent_id = str(getattr(item, "agent_id", "")).strip()
            description = str(getattr(item, "description", "")).strip()
            capabilities = getattr(item, "capabilities", ())
        if not agent_id or not description:
            raise ValueError("agent catalogue entries need agent_id and description")
        normalized.append(
            (agent_id, description, tuple(str(value) for value in capabilities))
        )
    return tuple(normalized)


def _routing_instructions(
    base_prompt: str,
    catalog: tuple[tuple[str, str, tuple[str, ...]], ...],
) -> str:
    lines = ["\nRuntime agent catalogue:"]
    for agent_id, description, capabilities in catalog:
        suffix = f" Capabilities: {', '.join(capabilities)}." if capabilities else ""
        lines.append(f"- {agent_id}: {description}{suffix}")
    if not catalog:
        lines.append("- No specialist agents are currently registered.")
    return base_prompt + "\n".join(lines)


async def extract_intent(
    text: str,
    *,
    extractor: Optional[IntentExtractor] = None,
) -> ExtractedIntent:
    """Convenience entry point for callers that do not manage a service object."""

    service = extractor or IntentExtractor()
    return await service.extract(text)
