"""Parse and validate the versioned, shared study materials."""

from __future__ import annotations

import csv
import hashlib
import re
from io import StringIO
from pathlib import Path

from pydantic import BaseModel

from backend.study.setup import MATERIALS_DIR


SCENARIO_PATH = MATERIALS_DIR / "scenario_stimuli.md"
TRACE_PATH = MATERIALS_DIR / "event_trace.csv"
ARTIFACT_DIR = MATERIALS_DIR / "final_artifacts"
TRACE_COLUMNS = ("variant", "event_id", "trigger", "offset_s", "gate", "script_key", "notes")
VARIANTS = {1: "wine_1", 2: "wine_2"}
FEEDBACK_KEYS = {"blocker_feedback", "approval_feedback"}


class SourceCard(BaseModel):
    source_id: str
    title: str
    text: str
    release_gate: str


class EventSpec(BaseModel):
    variant: str
    event_id: str
    trigger: str
    offset_s: int
    gate: bool
    script_key: str
    notes: str


class VariantStimuli(BaseModel):
    variant: str
    participant_brief: str
    sources: list[SourceCard]
    approval_draft_preview: str
    final_report_excerpt: str
    final_code_artifact: str
    final_markdown_artifact: str
    event_copy: dict[str, str]
    trace: list[EventSpec]


class StudyMaterials(BaseModel):
    version: str
    common_instruction: str
    variants: dict[str, VariantStimuli]
    feedback: dict[str, str]
    scenario_checksum: str
    trace_checksum: str


def _section(text: str, start: str, end: str | None = None) -> str:
    if text.count(start) != 1:
        raise ValueError(f"Expected one study section: {start}")
    after = text.split(start, 1)[1]
    if end is None:
        return after
    if end not in after:
        raise ValueError(f"Missing study section boundary: {end}")
    return after.split(end, 1)[0]


def _quote_after(text: str, marker: str) -> str:
    if text.count(marker) != 1:
        raise ValueError(f"Expected one study marker: {marker}")
    after = text.split(marker, 1)[1].lstrip()
    quoted = []
    for line in after.splitlines():
        if not line.startswith("> "):
            break
        quoted.append(line[2:])
    if not quoted:
        raise ValueError(f"Missing quoted study copy after: {marker}")
    return "\n".join(quoted).strip()


def _source_cards(section: str, variant_number: int) -> list[SourceCard]:
    pattern = re.compile(r"^\*\*Source (V[12]-S[123]) — (.+?):\*\*\s*$", re.MULTILINE)
    matches = list(pattern.finditer(section))
    expected_ids = [f"V{variant_number}-S{number}" for number in (1, 2, 3)]
    if [match.group(1) for match in matches] != expected_ids:
        raise ValueError(f"Variant {variant_number} must have source cards {expected_ids}")
    cards = []
    for match in matches:
        cards.append(
            SourceCard(
                source_id=match.group(1),
                title=match.group(2),
                text=_quote_after(section, match.group(0)),
                release_gate="final_review" if match.group(1).endswith("S3") else "briefing",
            )
        )
    return cards


def _event_copy(section: str) -> dict[str, dict[str, str]]:
    rows = []
    for line in section.splitlines():
        if not line.startswith("| ") or line.startswith("| ---") or line.startswith("| Script key"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            raise ValueError(f"Invalid fixed event copy row: {line}")
        rows.append(cells)
    if not rows:
        raise ValueError("Missing fixed event copy table")
    copies = {variant: {} for variant in VARIANTS.values()}
    for key_cell, first, second in rows:
        keys = re.findall(r"`([^`]+)`", key_cell)
        if not keys:
            raise ValueError(f"Missing script key in row: {key_cell}")
        if second in {"Same text.", "Same procedure and question.", "Same instruction."}:
            second = first
        for key in keys:
            if key in copies["wine_1"]:
                raise ValueError(f"Duplicate script key: {key}")
            copies["wine_1"][key] = first.strip("“”")
            copies["wine_2"][key] = second.strip("“”")
    return copies


def _feedback(text: str) -> dict[str, str]:
    section = _section(text, "## Fixed branch feedback and safety", "## Interface parity acceptance test")
    prefixes = {
        "blocker_correct": "- If blocker answer is correct:",
        "blocker_wrong": "- If blocker answer is wrong or unclear:",
        "approval_rejected": "- If approval is rejected:",
        "approval_approved_or_deferred": "- If approval is approved or deferred:",
    }
    feedback = {}
    for key, prefix in prefixes.items():
        line = next((item for item in section.splitlines() if item.startswith(prefix)), None)
        if line is None:
            raise ValueError(f"Missing feedback rule: {key}")
        quoted = re.findall(r"“([^”]+)”", line)
        if len(quoted) != 1:
            raise ValueError(f"Expected one fixed feedback message: {key}")
        feedback[key] = quoted[0]
    return feedback


def load_study_materials(
    scenario_path: Path = SCENARIO_PATH,
    trace_path: Path = TRACE_PATH,
    artifact_dir: Path = ARTIFACT_DIR,
) -> StudyMaterials:
    scenario_raw = scenario_path.read_bytes()
    trace_raw = trace_path.read_bytes()
    scenario = scenario_raw.decode("utf-8-sig")
    version_match = re.search(r"^# Frozen wine scenario stimuli — (.+)$", scenario, re.MULTILINE)
    if version_match is None:
        raise ValueError("Scenario materials need a versioned heading")
    common = _section(
        scenario,
        "## Common participant instruction before either variant",
        "### Neutral training project",
    )
    event_section = _section(scenario, "## Fixed event copy", "## Fixed branch feedback and safety")
    event_copies = _event_copy(event_section)

    reader = csv.DictReader(StringIO(trace_raw.decode("utf-8-sig")))
    if tuple(reader.fieldnames or ()) != TRACE_COLUMNS:
        raise ValueError("event_trace.csv has unexpected columns")
    traces = {variant: [] for variant in VARIANTS.values()}
    seen_ids = set()
    for row in reader:
        if row["variant"] not in traces:
            raise ValueError(f"Unknown trace variant: {row['variant']}")
        if row["event_id"] in seen_ids:
            raise ValueError(f"Duplicate trace event ID: {row['event_id']}")
        seen_ids.add(row["event_id"])
        if row["gate"] not in {"yes", "no"}:
            raise ValueError(f"Invalid gate value for {row['event_id']}")
        if row["script_key"] not in set(event_copies[row["variant"]]) | FEEDBACK_KEYS:
            raise ValueError(f"Missing scenario copy for {row['event_id']}: {row['script_key']}")
        try:
            offset = int(row["offset_s"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid offset for {row['event_id']}") from exc
        if offset < 0:
            raise ValueError(f"Negative offset for {row['event_id']}")
        traces[row["variant"]].append(
            EventSpec(
                variant=row["variant"],
                event_id=row["event_id"],
                trigger=row["trigger"],
                offset_s=offset,
                gate=row["gate"] == "yes",
                script_key=row["script_key"],
                notes=row["notes"],
            )
        )
    for number, variant in VARIANTS.items():
        expected_ids = [f"W{number}_{index:02d}" for index in range(11)]
        if [event.event_id for event in traces[variant]] != expected_ids:
            raise ValueError(f"Trace for {variant} must contain ordered events {expected_ids}")

    variants = {}
    scenario_digest = hashlib.sha256()
    for name, contents in ((scenario_path.name, scenario_raw),):
        scenario_digest.update(name.encode("utf-8"))
        scenario_digest.update(len(contents).to_bytes(8, "big"))
        scenario_digest.update(contents)
    for number, variant in VARIANTS.items():
        next_heading = f"### Variant {number + 1}:" if number == 1 else "## Fixed event copy"
        start_heading = re.search(rf"^### Variant {number}:.*$", scenario, re.MULTILINE)
        if start_heading is None:
            raise ValueError(f"Missing participant brief for {variant}")
        variant_section = _section(scenario, start_heading.group(0), next_heading)
        code_name = f"{variant}_analysis.py"
        markdown_name = f"{variant}_report.md"
        code_raw = (artifact_dir / code_name).read_bytes()
        markdown_raw = (artifact_dir / markdown_name).read_bytes()
        for name, contents in ((code_name, code_raw), (markdown_name, markdown_raw)):
            if not contents.strip():
                raise ValueError(f"Final study artifact is empty: {name}")
            scenario_digest.update(name.encode("utf-8"))
            scenario_digest.update(len(contents).to_bytes(8, "big"))
            scenario_digest.update(contents)
        final_excerpt = _quote_after(event_section, f"**Variant {number} final report excerpt:**")
        markdown = markdown_raw.decode("utf-8-sig")
        if final_excerpt not in markdown:
            raise ValueError(f"Final Markdown artifact does not contain the fixed report excerpt: {markdown_name}")
        variants[variant] = VariantStimuli(
            variant=variant,
            participant_brief=_quote_after(variant_section, "**Participant brief (show before delegation):**"),
            sources=_source_cards(variant_section, number),
            approval_draft_preview=_quote_after(
                variant_section,
                f"**Variant {number} Writing draft preview (available with approval request):**",
            ),
            final_report_excerpt=final_excerpt,
            final_code_artifact=code_raw.decode("utf-8-sig"),
            final_markdown_artifact=markdown,
            event_copy=event_copies[variant],
            trace=traces[variant],
        )

    return StudyMaterials(
        version=version_match.group(1),
        common_instruction=_quote_after(scenario, "## Common participant instruction before either variant"),
        variants=variants,
        feedback=_feedback(scenario),
        scenario_checksum=scenario_digest.hexdigest(),
        trace_checksum=hashlib.sha256(trace_raw).hexdigest(),
    )
