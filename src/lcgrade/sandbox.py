from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(slots=True)
class SandboxRun:
    """Result from running a Python script in a lightweight subprocess sandbox."""

    command: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: float


def run_python_script(
    script_path: Path,
    *,
    args: Sequence[str] = (),
    timeout_seconds: float | None = 5.0,
    stdin_text: str | None = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    python_executable: str | None = None,
) -> SandboxRun:
    """Run a Python script in a subprocess and capture its output.

    This is intentionally small beta groundwork for later integration into the
    execution pipeline. It provides process isolation, stdout/stderr capture,
    and timeout handling without pulling in extra dependencies.
    """

    resolved_script = Path(script_path)
    if not resolved_script.exists():
        raise FileNotFoundError(resolved_script)

    command = (
        python_executable or sys.executable,
        "-I",
        str(resolved_script),
        *tuple(args),
    )

    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=None if cwd is None else str(cwd),
        env=None if env is None else dict(env),
        stdin=subprocess.PIPE if stdin_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    timed_out = False
    returncode: int | None
    try:
        stdout, stderr = process.communicate(input=stdin_text, timeout=timeout_seconds)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate()
        returncode = None

    duration_ms = (time.perf_counter() - started) * 1000.0
    return SandboxRun(
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        duration_ms=duration_ms,
    )
