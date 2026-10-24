"""Repository-scoped tools for the ANISHA coding agent.

Version 1.0.0 intentionally exposes no commit, push, merge, pull-request, delete,
or unrestricted shell operation. Consequential actions remain orchestrator-owned.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


IGNORED_PARTS = {".git", ".venv", "node_modules", "__pycache__"}
SENSITIVE_NAMES = {".env", "id_rsa", "id_ed25519", "credentials.json"}
SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
MAX_READ_BYTES = 256_000
MAX_TOOL_OUTPUT = 40_000
TEST_COMMAND_TIMEOUT = 30
RETRY_COMMAND_TIMEOUT = 10
MAX_COMMAND_TIMEOUTS = 3
PYTHON_CHECK_HARNESS = r"""
import importlib.util
import json
import pathlib
import sys

payload = json.load(sys.stdin)
module_path = pathlib.Path(payload["module_path"])
sys.path.insert(0, str(module_path.parent))
spec = importlib.util.spec_from_file_location("anisha_checked_module", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {module_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for index, check in enumerate(payload["checks"], start=1):
    function = getattr(module, check["function"])
    actual = function(*check["args"])
    expected = check["expected"]
    if actual != expected:
        raise AssertionError(
            f"check {index} failed: {check['function']} returned {actual!r}; "
            f"expected {expected!r}"
        )
print(f"{len(payload['checks'])} Python behavioral checks passed")
"""


@dataclass(frozen=True)
class ToolExecutionResult:
    ok: bool
    content: str

    def as_json(self) -> str:
        return json.dumps({"ok": self.ok, "content": self.content})


class CodingToolbox:
    """Safe-by-default file and verification tools within one repository root."""

    def __init__(
        self,
        root: Path | str,
        *,
        command_timeout: int = 120,
        read_only_paths: Sequence[str] = (),
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("coding workspace root must be a directory")
        if command_timeout < 1:
            raise ValueError("command_timeout must be positive")
        self.read_only_paths = frozenset(
            self._normalize_read_only_path(path) for path in read_only_paths
        )
        self.command_timeout = command_timeout
        self.changed_files: set[str] = set()
        self.executed_commands: list[list[str]] = []
        self.verification_results: list[bool] = []
        self._command_timeouts: dict[tuple[str, ...], int] = {}

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [
            _tool(
                "list_files",
                "List repository files under a relative directory.",
                {
                    "path": {"type": "string", "default": "."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                },
            ),
            _tool(
                "read_file",
                "Read a UTF-8 repository file, optionally by inclusive line range.",
                {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                required=["path"],
            ),
            _tool(
                "search_files",
                "Search UTF-8 repository files for a text or regular-expression pattern.",
                {
                    "query": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "regex": {"type": "boolean", "default": False},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                required=["query"],
            ),
            _tool(
                "write_file",
                "Create or replace a UTF-8 file inside the repository.",
                {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                required=["path", "content"],
            ),
            _tool(
                "replace_in_file",
                "Replace one exact text occurrence in a repository file.",
                {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                required=["path", "old", "new"],
            ),
            _tool(
                "run_verification",
                "Run an allowlisted test, lint, build, type-check, or read-only git command.",
                {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "cwd": {"type": "string", "default": "."},
                },
                required=["argv"],
            ),
            _tool(
                "run_python_checks",
                (
                    "Run focused function checks against a Python source file without "
                    "creating a test file. Use this when the task prohibits test files."
                ),
                {
                    "module_path": {"type": "string"},
                    "checks": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 50,
                        "items": {
                            "type": "object",
                            "properties": {
                                "function": {"type": "string"},
                                "args": {"type": "array", "default": []},
                                "expected": {},
                            },
                            "required": ["function", "expected"],
                            "additionalProperties": False,
                        },
                    },
                },
                required=["module_path", "checks"],
            ),
            _tool(
                "run_python_file",
                (
                    "Run one Python source file inside the project with the "
                    "project-local environment when it exists. No command-line "
                    "arguments or shell are permitted."
                ),
                {"path": {"type": "string"}},
                required=["path"],
            ),
        ]

    def execute(self, name: str, arguments: Mapping[str, Any]) -> ToolExecutionResult:
        handlers: dict[str, Callable[..., ToolExecutionResult]] = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search_files": self.search_files,
            "write_file": self.write_file,
            "replace_in_file": self.replace_in_file,
            "run_verification": self.run_verification,
            "run_python_checks": self.run_python_checks,
            "run_python_file": self.run_python_file,
        }
        handler = handlers.get(name)
        if handler is None:
            return ToolExecutionResult(False, f"Unknown or prohibited tool: {name}")
        try:
            return handler(**dict(arguments))
        except (OSError, TypeError, ValueError, re.error) as exc:
            return ToolExecutionResult(False, str(exc))

    def list_files(self, path: str = ".", limit: int = 200) -> ToolExecutionResult:
        limit = _bounded_limit(limit, maximum=500)
        directory = self._resolve(path, must_exist=True)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {path}")
        files: list[str] = []
        for candidate in sorted(directory.rglob("*")):
            relative = candidate.relative_to(self.root)
            if (
                any(part in IGNORED_PARTS for part in relative.parts)
                or _is_sensitive(relative)
                or not self._is_confined(candidate)
            ):
                continue
            if candidate.is_file():
                files.append(relative.as_posix())
                if len(files) >= limit:
                    break
        return ToolExecutionResult(True, "\n".join(files))

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
    ) -> ToolExecutionResult:
        file_path = self._resolve(path, must_exist=True)
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")
        if file_path.stat().st_size > MAX_READ_BYTES:
            raise ValueError(f"File exceeds {MAX_READ_BYTES} byte read limit: {path}")
        if start_line < 1 or (end_line is not None and end_line < start_line):
            raise ValueError("invalid line range")
        lines = file_path.read_text(encoding="utf-8").splitlines()
        selected = lines[start_line - 1 : end_line]
        numbered = [
            f"{number}: {line}"
            for number, line in enumerate(selected, start=start_line)
        ]
        return ToolExecutionResult(True, _truncate("\n".join(numbered)))

    def search_files(
        self,
        query: str,
        path: str = ".",
        regex: bool = False,
        limit: int = 100,
    ) -> ToolExecutionResult:
        if not query:
            raise ValueError("query must not be empty")
        limit = _bounded_limit(limit, maximum=200)
        start = self._resolve(path, must_exist=True)
        candidates = [start] if start.is_file() else sorted(start.rglob("*"))
        pattern = re.compile(query if regex else re.escape(query))
        matches: list[str] = []
        for candidate in candidates:
            relative = candidate.relative_to(self.root)
            if (
                not candidate.is_file()
                or any(part in IGNORED_PARTS for part in relative.parts)
                or _is_sensitive(relative)
                or not self._is_confined(candidate)
                or candidate.stat().st_size > MAX_READ_BYTES
            ):
                continue
            try:
                lines = candidate.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if pattern.search(line):
                    matches.append(f"{relative.as_posix()}:{line_number}:{line}")
                    if len(matches) >= limit:
                        return ToolExecutionResult(True, _truncate("\n".join(matches)))
        return ToolExecutionResult(True, _truncate("\n".join(matches)))

    def write_file(self, path: str, content: str) -> ToolExecutionResult:
        file_path = self._resolve(path, must_exist=False)
        self._require_writable(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = file_path.with_name(f".{file_path.name}.anisha-tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(file_path)
        relative = file_path.relative_to(self.root).as_posix()
        self.changed_files.add(relative)
        return ToolExecutionResult(True, f"Wrote {relative} ({len(content)} characters)")

    def replace_in_file(self, path: str, old: str, new: str) -> ToolExecutionResult:
        if not old:
            raise ValueError("old text must not be empty")
        file_path = self._resolve(path, must_exist=True)
        self._require_writable(file_path)
        if file_path.stat().st_size > MAX_READ_BYTES:
            raise ValueError(f"File exceeds {MAX_READ_BYTES} byte edit limit: {path}")
        content = file_path.read_text(encoding="utf-8")
        count = content.count(old)
        if count != 1:
            raise ValueError(f"Expected exactly one match in {path}; found {count}")
        return self.write_file(path, content.replace(old, new, 1))

    def run_verification(
        self, argv: Sequence[str], cwd: str = "."
    ) -> ToolExecutionResult:
        command = [str(part) for part in argv]
        _validate_command(command)
        command = self._project_python_command(command)
        working_directory = self._resolve(cwd, must_exist=True)
        if not working_directory.is_dir():
            raise ValueError(f"Not a directory: {cwd}")
        self.executed_commands.append(command)
        command_key = tuple(command)
        timeout_count = self._command_timeouts.get(command_key, 0)
        if timeout_count >= MAX_COMMAND_TIMEOUTS:
            self.verification_results.append(False)
            return ToolExecutionResult(
                False,
                "Verification circuit breaker is open: this command already "
                f"failed to terminate {timeout_count} times. Use a different "
                "deterministic, non-interactive verification strategy.",
            )
        timeout_seconds = _verification_timeout(
            command, self.command_timeout, timeout_count
        )
        try:
            # Each verification gets a fresh bytecode cache. Otherwise rapid,
            # same-size Python edits can leave a timestamp-based .pyc looking
            # valid and cause tests to execute the previous source contents.
            with tempfile.TemporaryDirectory(
                prefix="anisha-verification-pycache-"
            ) as pycache_directory:
                environment = _safe_environment()
                environment["PYTHONPYCACHEPREFIX"] = pycache_directory
                completed = subprocess.run(
                    command,
                    cwd=working_directory,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    shell=False,
                    env=environment,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            timeout_count += 1
            self._command_timeouts[command_key] = timeout_count
            self.verification_results.append(False)
            partial_output = _timeout_output(exc)
            output_suffix = (
                f"\nPartial output:\n{partial_output}" if partial_output else ""
            )
            return ToolExecutionResult(
                False,
                f"Verification timed out after {timeout_seconds} seconds "
                f"(non-terminating attempt {timeout_count}/{MAX_COMMAND_TIMEOUTS}). "
                "The process was stopped. Verification must terminate without "
                "human interaction. Do not rerun it unchanged: inspect the "
                "implementation for interactive input, event loops, servers, "
                "watchers, deadlocks, or unbounded work; isolate the behavior and "
                f"choose a deterministic verification strategy.{output_suffix}",
            )
        self._command_timeouts.pop(command_key, None)
        output = "\n".join(
            part for part in (completed.stdout.strip(), completed.stderr.strip()) if part
        )
        succeeded = completed.returncode == 0
        self.verification_results.append(succeeded)
        return ToolExecutionResult(
            succeeded,
            _truncate(f"exit_code={completed.returncode}\n{output}".rstrip()),
        )

    def run_python_checks(
        self,
        module_path: str,
        checks: Sequence[Mapping[str, Any]],
    ) -> ToolExecutionResult:
        """Run data-only function checks with a runtime-owned Python harness."""

        module_file = self._resolve(module_path, must_exist=True)
        if not module_file.is_file() or module_file.suffix.casefold() != ".py":
            raise ValueError("module_path must reference a Python source file")
        normalized_checks = _validate_python_checks(checks)
        relative_module = module_file.relative_to(self.root).as_posix()
        command = ["python-checks", relative_module]
        self.executed_commands.append(command)

        payload = json.dumps(
            {
                "module_path": str(module_file),
                "checks": normalized_checks,
            }
        )
        try:
            completed = subprocess.run(
                [str(self._python_executable()), "-I", "-B", "-c", PYTHON_CHECK_HARNESS],
                input=payload,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=min(self.command_timeout, TEST_COMMAND_TIMEOUT),
                shell=False,
                env=_safe_environment(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self.verification_results.append(False)
            partial_output = _timeout_output(exc)
            return ToolExecutionResult(
                False,
                "Python behavioral checks timed out."
                + (f"\nPartial output:\n{partial_output}" if partial_output else ""),
            )

        output = "\n".join(
            part
            for part in (completed.stdout.strip(), completed.stderr.strip())
            if part
        )
        succeeded = completed.returncode == 0
        self.verification_results.append(succeeded)
        return ToolExecutionResult(
            succeeded,
            _truncate(f"exit_code={completed.returncode}\n{output}".rstrip()),
        )

    def run_python_file(self, path: str) -> ToolExecutionResult:
        """Execute a single confined Python file without a shell or arguments."""

        source = self._resolve(path, must_exist=True)
        if not source.is_file() or source.suffix.casefold() != ".py":
            raise ValueError("path must reference a Python source file")
        relative = source.relative_to(self.root).as_posix()
        command = [str(self._python_executable()), relative]
        self.executed_commands.append(command)
        try:
            completed = subprocess.run(
                command,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
                shell=False,
                env=_safe_environment(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self.verification_results.append(False)
            partial_output = _timeout_output(exc)
            return ToolExecutionResult(
                False,
                "Python file execution timed out."
                + (f"\nPartial output:\n{partial_output}" if partial_output else ""),
            )
        output = "\n".join(
            part for part in (completed.stdout.strip(), completed.stderr.strip()) if part
        )
        succeeded = completed.returncode == 0
        self.verification_results.append(succeeded)
        return ToolExecutionResult(
            succeeded,
            _truncate(f"exit_code={completed.returncode}\n{output}".rstrip()),
        )

    def _python_executable(self) -> Path:
        candidate = (
            self.root / ".venv" / ("Scripts" if os.name == "nt" else "bin") /
            ("python.exe" if os.name == "nt" else "python")
        )
        return candidate if candidate.is_file() else Path(sys.executable)

    def _project_python_command(self, command: list[str]) -> list[str]:
        executable = Path(command[0]).name.casefold()
        if re.fullmatch(r"python(?:3(?:\.\d+)?)?(?:\.exe)?", executable):
            return [str(self._python_executable()), *command[1:]]
        if executable == "pytest":
            return [str(self._python_executable()), "-m", "pytest", *command[1:]]
        return command

    def _resolve(self, relative_path: str, *, must_exist: bool) -> Path:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError("path must be a non-empty relative path")
        requested = Path(relative_path)
        if requested.is_absolute():
            raise ValueError("absolute paths are prohibited")
        candidate = (self.root / requested).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("path escapes the coding workspace") from exc
        if any(part in IGNORED_PARTS for part in candidate.relative_to(self.root).parts):
            raise ValueError("access to internal or dependency directories is prohibited")
        if _is_sensitive(candidate.relative_to(self.root)):
            raise ValueError("access to environment or credential files is prohibited")
        if must_exist and not candidate.exists():
            raise ValueError(f"Path does not exist: {relative_path}")
        return candidate

    def _is_confined(self, candidate: Path) -> bool:
        try:
            candidate.resolve(strict=False).relative_to(self.root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _normalize_read_only_path(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("read-only paths must be non-empty relative paths")
        path = Path(value.strip())
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("read-only paths must stay inside the coding workspace")
        normalized = path.as_posix().removeprefix("./")
        if normalized in {"", "."}:
            raise ValueError("the entire coding workspace cannot be read-only")
        return normalized

    def _require_writable(self, candidate: Path) -> None:
        relative = candidate.relative_to(self.root).as_posix()
        if relative in self.read_only_paths:
            raise ValueError(f"Path is read-only project input: {relative}")


def _tool(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    *,
    required: Optional[list[str]] = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(properties),
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


def _bounded_limit(value: int, *, maximum: int) -> int:
    value = int(value)
    if not 1 <= value <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
    return value


def _validate_python_checks(
    checks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(checks, (str, bytes)) or not 1 <= len(checks) <= 50:
        raise ValueError("checks must contain between 1 and 50 items")

    normalized: list[dict[str, Any]] = []
    for check in checks:
        if not isinstance(check, Mapping):
            raise ValueError("each Python check must be an object")
        if set(check) - {"function", "args", "expected"}:
            raise ValueError("Python checks contain unsupported fields")
        function = check.get("function")
        args = check.get("args", [])
        if not isinstance(function, str) or not function.isidentifier():
            raise ValueError("check function must be a simple Python identifier")
        if not isinstance(args, list):
            raise ValueError("check args must be an array")
        if "expected" not in check:
            raise ValueError("each Python check requires an expected value")
        candidate = {
            "function": function,
            "args": args,
            "expected": check["expected"],
        }
        try:
            json.dumps(candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError("Python check values must be JSON-compatible") from exc
        normalized.append(candidate)
    return normalized


def _truncate(value: str) -> str:
    if len(value) <= MAX_TOOL_OUTPUT:
        return value
    return value[:MAX_TOOL_OUTPUT] + "\n... output truncated ..."


def _is_sensitive(path: Path) -> bool:
    name = path.name.lower()
    return (
        name in SENSITIVE_NAMES
        or (name.startswith(".env.") and name != ".env.example")
        or path.suffix.lower() in SENSITIVE_SUFFIXES
    )


def _safe_environment() -> dict[str, str]:
    allowed_names = {
        "CI",
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONPATH",
        "SYSTEMROOT",
        "TMPDIR",
        "VIRTUAL_ENV",
    }
    environment = {
        name: value for name, value in os.environ.items() if name in allowed_names
    }
    environment["CI"] = "true"
    return environment


def _verification_timeout(
    command: Sequence[str], configured_timeout: int, timeout_count: int
) -> int:
    if timeout_count:
        return min(configured_timeout, RETRY_COMMAND_TIMEOUT)
    if _is_test_command(command):
        return min(configured_timeout, TEST_COMMAND_TIMEOUT)
    return configured_timeout


def _is_test_command(command: Sequence[str]) -> bool:
    if not command:
        return False
    executable = Path(command[0]).name.casefold()
    args = list(command[1:])
    python_test = bool(
        re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable)
        and len(args) >= 2
        and args[0] == "-m"
        and args[1] in {"pytest", "unittest"}
    )
    if python_test or executable in {"pytest", "phpunit"}:
        return True
    if executable in {"npm", "pnpm", "yarn"}:
        return bool(args) and (
            args[0] == "test"
            or (len(args) >= 2 and args[0] == "run" and args[1] == "test")
        )
    if executable in {"cargo", "go", "dotnet", "mix", "swift"}:
        return bool(args) and args[0] == "test"
    return executable in {"gradle", "gradlew", "mvn", "mvnw"} and "test" in args


def _timeout_output(exc: subprocess.TimeoutExpired) -> str:
    parts: list[str] = []
    for value in (exc.stdout, exc.stderr):
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return _truncate("\n".join(parts))


def _validate_command(argv: Sequence[str]) -> None:
    if not argv or any(not part for part in argv):
        raise ValueError("argv must contain non-empty strings")
    if any("\x00" in part or "\n" in part for part in argv):
        raise ValueError("invalid command argument")

    executable = Path(argv[0]).name
    args = list(argv[1:])
    for argument in args:
        candidate = argument.split("=", 1)[-1] if "=" in argument else argument
        candidate_path = Path(candidate)
        if candidate_path.is_absolute() or ".." in candidate_path.parts:
            raise ValueError("command arguments may not access paths outside the workspace")
        if argument.startswith(("--git-dir", "--work-tree", "--config-env")):
            raise ValueError("Git repository overrides are prohibited")

    allowed = False
    if re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable):
        allowed = len(args) >= 2 and args[0] == "-m" and args[1] in {
            "pytest",
            "unittest",
            "compileall",
        }
    elif executable in {"pytest", "ruff", "mypy", "tsc"}:
        allowed = True
    elif executable == "git":
        if args and args[0] == "status":
            allowed = all(
                argument in {"--short", "--porcelain", "--branch"}
                or argument.startswith("--untracked-files=")
                for argument in args[1:]
            )
        elif args and args[0] == "diff":
            prohibited_diff_flags = {"--no-index", "--ext-diff", "--output"}
            allowed = not any(
                argument in prohibited_diff_flags
                or argument.startswith("--output=")
                for argument in args[1:]
            )
    elif executable == "npm":
        allowed = bool(args) and (
            args[0] == "test"
            or (len(args) >= 2 and args[0] == "run" and args[1] in {"test", "lint", "build", "typecheck"})
        )
    elif executable in {"pnpm", "yarn"}:
        allowed = bool(args) and args[0] in {"test", "lint", "build", "typecheck"}
    elif executable == "cargo":
        allowed = bool(args) and args[0] in {"test", "check", "clippy"}
    elif executable == "go":
        allowed = bool(args) and args[0] == "test"

    if not allowed:
        raise ValueError(
            "command is not allowlisted; commits, pushes, merges, network commands, "
            "and unrestricted shell execution are prohibited"
        )
