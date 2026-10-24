"""Task-scoped Markdown document writing for the ANISHA writing agent."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MAX_DOCUMENT_CHARACTERS = 100_000
IMMUTABLE_PATH_PARTS = {"meetings", "meeting-notes", "meeting_notes", "sources"}
IGNORED_SOURCE_PARTS = {".git", ".venv", "node_modules", "__pycache__"}
SENSITIVE_SOURCE_NAMES = {".env", "credentials.json", "id_rsa", "id_ed25519"}
SENSITIVE_SOURCE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


@dataclass(frozen=True)
class ToolExecutionResult:
    ok: bool
    content: str
    artefact_refs: tuple[str, ...] = ()


class WritingToolbox:
    """Create drafts and edit named Markdown files within one workspace root."""

    def __init__(
        self,
        root: Path | str,
        *,
        read_only_paths: Sequence[str] = (),
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("writing workspace root must be a directory")
        self.read_only_paths = frozenset(
            self._normalize_read_only_path(path) for path in read_only_paths
        )

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_document",
                    "description": (
                        "Save a Markdown draft as a new version in the project "
                        "drafts directory. The runtime chooses the output path."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["task_id", "title", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Read a UTF-8 text file explicitly named in the confirmed task "
                        "from inside the project workspace. Read-only."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_document",
                    "description": (
                        "Create a new Markdown document at the exact path confirmed "
                        "by the user. Never overwrite an existing path."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "target_path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["task_id", "target_path", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_document",
                    "description": (
                        "Replace an explicitly named Markdown document inside the "
                        "project workspace. Preserve its original under this task's drafts."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "target_path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["task_id", "target_path", "content"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def execute(
        self, name: str, arguments: Mapping[str, Any]
    ) -> ToolExecutionResult:
        if name not in {
            "read_file",
            "write_document",
            "create_document",
            "edit_document",
        }:
            return ToolExecutionResult(False, f"Unknown writing tool: {name}")
        try:
            if name == "read_file":
                return self.read_file(**dict(arguments))
            if name == "write_document":
                return self.write_document(**dict(arguments))
            if name == "create_document":
                return self.create_document(**dict(arguments))
            return self.edit_document(**dict(arguments))
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            return ToolExecutionResult(False, str(exc))

    def read_file(self, path: str) -> ToolExecutionResult:
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source path must stay inside the writing workspace")
        if not path.strip() or any(
            part.casefold() in IGNORED_SOURCE_PARTS for part in relative.parts
        ):
            raise ValueError("source path is empty or restricted")
        if (
            relative.name.casefold() in SENSITIVE_SOURCE_NAMES
            or relative.suffix.casefold() in SENSITIVE_SOURCE_SUFFIXES
        ):
            raise ValueError("sensitive files cannot be read by the writing agent")

        candidate = self.root / relative
        cursor = self.root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("writing agent cannot read symlinked files")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(self.root):
            raise ValueError("source path escapes the writing workspace")
        if not resolved.is_file():
            raise ValueError(f"source is not a file: {path}")
        if resolved.stat().st_size > MAX_DOCUMENT_CHARACTERS:
            raise ValueError(
                f"source file exceeds {MAX_DOCUMENT_CHARACTERS} byte limit"
            )
        content = resolved.read_text(encoding="utf-8")
        return ToolExecutionResult(
            True,
            content,
            (resolved.relative_to(self.root).as_posix(),),
        )

    def write_document(
        self, task_id: str, title: str, content: str
    ) -> ToolExecutionResult:
        normalized_task_id = task_id.strip()
        normalized_title = " ".join(title.split()).lstrip("# ")
        normalized_content = content.strip()
        if not normalized_task_id:
            raise ValueError("task_id must not be empty")
        if not normalized_title:
            raise ValueError("document title must not be empty")
        if not normalized_content:
            raise ValueError("document content must not be empty")
        if len(normalized_content) > MAX_DOCUMENT_CHARACTERS:
            raise ValueError(
                f"document exceeds {MAX_DOCUMENT_CHARACTERS} character limit"
            )

        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", normalized_task_id).strip(".-")
        slug = slug or "task"
        task_directory = self._task_directory(normalized_task_id, slug)

        destination = task_directory / f"v{self._next_version(task_directory, 'v')}.md"
        document = f"# {normalized_title}\n\n{normalized_content}\n"
        self._write_atomically(destination, document, normalized_task_id)
        relative_path = destination.relative_to(self.root).as_posix()
        return ToolExecutionResult(True, relative_path, (relative_path,))

    def edit_document(
        self, task_id: str, target_path: str, content: str
    ) -> ToolExecutionResult:
        normalized_task_id = task_id.strip()
        normalized_content = content.strip()
        if not normalized_task_id:
            raise ValueError("task_id must not be empty")
        if not normalized_content:
            raise ValueError("document content must not be empty")
        if len(normalized_content) > MAX_DOCUMENT_CHARACTERS:
            raise ValueError(
                f"document exceeds {MAX_DOCUMENT_CHARACTERS} character limit"
            )

        target = self._existing_document_path(target_path)
        original = target.read_text(encoding="utf-8")
        if len(original) > MAX_DOCUMENT_CHARACTERS:
            raise ValueError(
                f"existing document exceeds {MAX_DOCUMENT_CHARACTERS} character limit"
            )

        task_slug = re.sub(
            r"[^A-Za-z0-9._-]+", "-", normalized_task_id
        ).strip(".-") or "task"
        task_directory = self._task_directory(normalized_task_id, task_slug)
        backup = task_directory / (
            f"original-v{self._next_version(task_directory, 'original-v')}.md"
        )
        self._write_atomically(backup, original, normalized_task_id)

        current_target = self._existing_document_path(target_path)
        self._write_atomically(
            current_target,
            f"{normalized_content}\n",
            normalized_task_id,
        )
        edited_ref = current_target.relative_to(self.root).as_posix()
        backup_ref = backup.relative_to(self.root).as_posix()
        return ToolExecutionResult(
            True,
            edited_ref,
            (edited_ref, backup_ref),
        )

    def create_document(
        self, task_id: str, target_path: str, content: str
    ) -> ToolExecutionResult:
        normalized_task_id = task_id.strip()
        normalized_content = content.strip()
        if not normalized_task_id:
            raise ValueError("task_id must not be empty")
        if not normalized_content:
            raise ValueError("document content must not be empty")
        if len(normalized_content) > MAX_DOCUMENT_CHARACTERS:
            raise ValueError(
                f"document exceeds {MAX_DOCUMENT_CHARACTERS} character limit"
            )

        target = self._new_document_path(target_path)
        try:
            with target.open("x", encoding="utf-8") as document:
                document.write(f"{normalized_content}\n")
        except FileExistsError as exc:
            raise ValueError(f"target already exists: {target_path}") from exc

        relative_path = target.relative_to(self.root).as_posix()
        return ToolExecutionResult(True, relative_path, (relative_path,))

    def _new_document_path(self, path: str) -> Path:
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("target path must stay inside the writing workspace")
        if relative.suffix.casefold() != ".md":
            raise ValueError("writing agent can create Markdown files only")
        if any(part.casefold() in IMMUTABLE_PATH_PARTS for part in relative.parts):
            raise ValueError("original meeting notes and source files are immutable")
        self._require_mutable_target(relative)

        candidate = self.root / relative
        cursor = self.root
        for part in relative.parts[:-1]:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(
                    "writing agent cannot create through symlinked directories"
                )
            if cursor.exists() and not cursor.is_dir():
                raise ValueError("target parent is not a directory")
        if candidate.is_symlink():
            raise ValueError("writing agent cannot replace symlinked files")
        if candidate.exists():
            raise ValueError(f"target already exists: {path}")

        candidate.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = candidate.parent.resolve(strict=True)
        if not resolved_parent.is_relative_to(self.root):
            raise ValueError("target path escapes the writing workspace")
        return candidate

    def _existing_document_path(self, path: str) -> Path:
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("target path must stay inside the writing workspace")
        if relative.suffix.casefold() != ".md":
            raise ValueError("writing agent can edit Markdown files only")
        if any(part.casefold() in IMMUTABLE_PATH_PARTS for part in relative.parts):
            raise ValueError("original meeting notes and source files are immutable")
        self._require_mutable_target(relative)

        candidate = self.root / relative
        if candidate.is_symlink():
            raise ValueError("writing agent cannot edit symlinked files")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(self.root):
            raise ValueError("target path escapes the writing workspace")
        if not resolved.is_file():
            raise ValueError(f"target is not a file: {path}")
        return resolved

    @staticmethod
    def _normalize_read_only_path(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("read-only paths must be non-empty relative paths")
        path = Path(value.strip())
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("read-only paths must stay inside the writing workspace")
        normalized = path.as_posix().removeprefix("./")
        if normalized in {"", "."}:
            raise ValueError("the entire writing workspace cannot be read-only")
        return normalized

    def _require_mutable_target(self, relative: Path) -> None:
        normalized = relative.as_posix().removeprefix("./")
        if normalized in self.read_only_paths:
            raise ValueError(f"Path is orchestrator-managed and read-only: {normalized}")

    def read_document(self, target_path: str) -> ToolExecutionResult:
        try:
            target = self._existing_document_path(target_path)
            content = target.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            return ToolExecutionResult(False, str(exc))
        if len(content) > MAX_DOCUMENT_CHARACTERS:
            return ToolExecutionResult(
                False,
                f"existing document exceeds {MAX_DOCUMENT_CHARACTERS} character limit",
            )
        return ToolExecutionResult(
            True,
            content,
            (target.relative_to(self.root).as_posix(),),
        )

    def _task_directory(self, task_id: str, slug: str) -> Path:
        task_hash = hashlib.sha256(task_id.encode()).hexdigest()[:8]
        task_directory = self.root / "drafts" / f"{slug}-{task_hash}"
        if not task_directory.resolve().is_relative_to(self.root):
            raise ValueError("draft path escapes the writing workspace")
        task_directory.mkdir(parents=True, exist_ok=True)
        if not task_directory.resolve(strict=True).is_relative_to(self.root):
            raise ValueError("draft path escapes the writing workspace")
        return task_directory

    @staticmethod
    def _next_version(directory: Path, prefix: str) -> int:
        version = 1
        while (directory / f"{prefix}{version}.md").exists():
            version += 1
        return version

    @staticmethod
    def _write_atomically(path: Path, content: str, task_id: str) -> None:
        task_hash = hashlib.sha256(task_id.encode()).hexdigest()[:8]
        temporary = path.with_name(f".{path.name}.{task_hash}.anisha-tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
