from __future__ import annotations

from pathlib import Path

from lcgrade.sandbox import run_python_script


def test_run_python_script_captures_stdout_and_stderr(tmp_path: Path) -> None:
    script_path = tmp_path / "hello.py"
    script_path.write_text(
        "import sys\n"
        "print('hello stdout')\n"
        "print('hello stderr', file=sys.stderr)\n",
        encoding="utf-8",
    )

    result = run_python_script(script_path, timeout_seconds=2.0)

    assert result.timed_out is False
    assert result.returncode == 0
    assert "hello stdout" in result.stdout
    assert "hello stderr" in result.stderr
    assert result.command[-1] == str(script_path)


def test_run_python_script_reports_timeout_and_partial_output(tmp_path: Path) -> None:
    script_path = tmp_path / "sleepy.py"
    script_path.write_text(
        "import sys\n"
        "import time\n"
        "print('start', flush=True)\n"
        "print('waiting', file=sys.stderr, flush=True)\n"
        "time.sleep(5)\n"
        "print('done', flush=True)\n",
        encoding="utf-8",
    )

    result = run_python_script(script_path, timeout_seconds=0.2)

    assert result.timed_out is True
    assert result.returncode is None
    assert "start" in result.stdout
    assert "waiting" in result.stderr
    assert "done" not in result.stdout
