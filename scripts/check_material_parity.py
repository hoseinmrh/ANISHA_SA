"""Compare this repository's frozen study inputs with the ANISHA source tree."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

local = Path(__file__).resolve().parents[1] / "materials"
upstream = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else
    Path(__file__).resolve().parents[2] / "Deliverables" / "ANISHA_Study_Package" / "materials"
)
paths = ["scenario_stimuli.md", "event_trace.csv", "counterbalancing.csv"]
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
