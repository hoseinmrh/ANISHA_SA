"""Check shared facts while allowing the versioned Condition C event difference."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

local = Path(__file__).resolve().parents[1] / "materials"
upstream = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else
    Path(__file__).resolve().parents[2] / "Deliverables" / "ANISHA_Study_Package" / "materials"
)
paths = ["counterbalancing.csv"]
paths += [f"final_artifacts/{path.name}" for path in (local / "final_artifacts").iterdir() if path.is_file()]
different = []
for relative in sorted(paths):
    own = local / relative
    other = upstream / relative
    own_hash = hashlib.sha256(own.read_bytes()).hexdigest()
    other_hash = hashlib.sha256(other.read_bytes()).hexdigest() if other.exists() else "missing"
    print(f"{relative}: {'match' if own_hash == other_hash else 'DIFF'} ({own_hash[:12]})")
    if own_hash != other_hash:
        different.append(relative)
if different:
    raise SystemExit(f"Study materials differ: {', '.join(different)}")

def source_text(scenario: str, source_id: str) -> str:
    pattern = rf"\*\*Source {source_id} — [^\n]+\*\*\s*\n\s*> ([^\n]+)"
    match = re.search(pattern, scenario)
    if not match:
        raise SystemExit(f"Missing source {source_id}")
    return match.group(1)


condition_c = (local / "scenario_stimuli.md").read_text(encoding="utf-8")
anisha = (upstream / "scenario_stimuli.md").read_text(encoding="utf-8")
for variant in (1, 2):
    for source in (1, 2, 3):
        source_id = f"V{variant}-S{source}"
        if source_text(condition_c, source_id) != source_text(anisha, source_id):
            raise SystemExit(f"Shared source text differs: {source_id}")
        print(f"{source_id}: match")

condition_c_trace = (local / "event_trace.csv").read_text(encoding="utf-8")
anisha_trace = (upstream / "event_trace.csv").read_text(encoding="utf-8")
if condition_c_trace.count(",coding_complete,") != 2 or ",coding_complete," in anisha_trace:
    raise SystemExit("Expected two Condition C-only Coding-completion events")
if condition_c_trace.count(",writing_requests_result,") != 2 or ",writing_requests_result," in anisha_trace:
    raise SystemExit("Expected two Condition C-only participant-handoff requests")
print("Condition C trace: two completion and handoff events; ANISHA trace unchanged")
