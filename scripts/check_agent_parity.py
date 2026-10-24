"""Verify that the copied agent runtime still matches the ANISHA source files."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

local = Path(__file__).resolve().parents[1]
upstream = Path(sys.argv[1]) if len(sys.argv) > 1 else local.parent / "Anisha"
relative_paths = [
    path.relative_to(local) for directory in ("agents", "tools", "llm", "memory", "workspace")
    for path in (local / "backend" / directory).glob("*.py")
]
relative_paths += [Path("backend/requirements.txt")]
relative_paths += [Path("backend/voice/stt.py"), Path("backend/voice/intent.py")]
relative_paths += [path.relative_to(local) for path in (local / "ml_project").iterdir() if path.is_file()]
different = []
for relative in sorted(relative_paths):
    copied = (local / relative).read_bytes()
    source_path = upstream / relative
    source = source_path.read_bytes() if source_path.exists() else b""
    copied_hash = hashlib.sha256(copied).hexdigest()
    matches = copied == source
    print(f"{relative}: {'match' if matches else 'DIFF'} ({copied_hash[:12]})")
    if not matches:
        different.append(str(relative))
if different:
    raise SystemExit(f"Agent source differs: {', '.join(different)}")
