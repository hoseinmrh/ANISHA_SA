"""Runtime-owned, approval-gated Python environment preparation."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


ENVIRONMENT_PATH = ".venv"
PACKAGE_ALLOWLIST = {
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
}
MISSING_MODULE_PATTERN = re.compile(
    r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]"
)


def missing_module_from_output(output: str) -> str | None:
    match = MISSING_MODULE_PATTERN.search(output)
    return match.group(1).split(".", 1)[0] if match else None


def discover_allowlisted_packages(root: Path, missing_module: str) -> list[str]:
    """Return exact PyPI names for allowlisted imports used by project Python files."""

    modules = {missing_module.split(".", 1)[0]}
    for path in root.rglob("*.py"):
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".", 1)[0])

    packages = {
        PACKAGE_ALLOWLIST[module]
        for module in modules
        if module in PACKAGE_ALLOWLIST
    }
    return sorted(packages)


def environment_python(root: Path) -> Path:
    if sys.platform == "win32":
        return root / ENVIRONMENT_PATH / "Scripts" / "python.exe"
    return root / ENVIRONMENT_PATH / "bin" / "python"


class PythonEnvironmentSetupError(RuntimeError):
    pass


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class PythonEnvironmentManager:
    """Create one fixed local environment and install only allowlisted packages."""

    def __init__(
        self,
        root: Path | str,
        *,
        command_runner: CommandRunner = subprocess.run,
        timeout_seconds: int = 600,
    ) -> None:
        self.root = Path(root).resolve(strict=True)
        self.command_runner = command_runner
        self.timeout_seconds = timeout_seconds

    def prepare(self, packages: Sequence[str]) -> None:
        normalized = sorted(set(packages))
        allowed = set(PACKAGE_ALLOWLIST.values())
        if not normalized or any(package not in allowed for package in normalized):
            raise ValueError("Python environment packages must come from the allowlist")

        environment_directory = self.root / ENVIRONMENT_PATH
        if environment_directory.exists():
            if environment_directory.is_symlink():
                raise PythonEnvironmentSetupError(
                    "The project .venv directory must not be a symbolic link"
                )
            try:
                environment_directory.resolve(strict=True).relative_to(self.root)
            except ValueError as exc:
                raise PythonEnvironmentSetupError(
                    "The project .venv resolves outside the project workspace"
                ) from exc
            if not environment_directory.is_dir():
                raise PythonEnvironmentSetupError(
                    "The project .venv is not a directory"
                )

        python = environment_python(self.root)
        if not python.is_file():
            self._run([sys.executable, "-m", "venv", ENVIRONMENT_PATH])
        try:
            python.parent.resolve(strict=True).relative_to(self.root)
        except (FileNotFoundError, ValueError) as exc:
            raise PythonEnvironmentSetupError(
                "The project environment bin directory is missing or escapes "
                "the workspace"
            ) from exc
        if not python.is_file():
            raise PythonEnvironmentSetupError(
                "The project environment Python executable is missing"
            )
        self._run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *normalized,
            ]
        )

    def _run(self, command: list[str]) -> None:
        completed = self.command_runner(
            command,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            shell=False,
            check=False,
            env=_safe_setup_environment(),
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown error").strip()
            raise PythonEnvironmentSetupError(detail[-2_000:])


def _safe_setup_environment() -> dict[str, str]:
    allowed_names = {
        "LANG",
        "LC_ALL",
        "PATH",
        "PIP_CERT",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TMPDIR",
    }
    environment = {
        name: value for name, value in os.environ.items() if name in allowed_names
    }
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment
