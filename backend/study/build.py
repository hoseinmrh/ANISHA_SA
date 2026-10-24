"""Stable fingerprint of the local study application's source build."""

from __future__ import annotations

import hashlib
from pathlib import Path


ANISHA_ROOT = Path(__file__).resolve().parents[2]


def current_build_hash(root: Path = ANISHA_ROOT) -> str:
    patterns = (
        "backend/**/*.py",
        "frontend/**/*.html",
        "frontend/**/*.css",
        "frontend/**/*.js",
        "backend/requirements.txt",
    )
    files = sorted(
        {
            path
            for pattern in patterns
            for path in root.glob(pattern)
            if path.is_file() and "tests" not in path.relative_to(root).parts
        }
    )
    if not files:
        raise ValueError("No study application source files found for build hash")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()
