from __future__ import annotations

import json
from pathlib import Path

from lcgrade.execution import execute_solution
from lcgrade.problems import parse_problem_directory


def _build_problem_dir(tmp_path: Path, solution_body: str, tests_payload: list[dict[str, object]]) -> Path:
    problem_dir = tmp_path / "echo-problem"
    problem_dir.mkdir()
    (problem_dir / "statement.md").write_text(
        "---\n"
        "schema_version: 1\n"
        "title: Echo Problem\n"
        "slug: echo-problem\n"
        "difficulty: easy\n"
        "function_name: echo_value\n"
        "params:\n"
        "  - name: value\n"
        "    type: int\n"
        "return_type: int\n"
        "validator: exact_match\n"
        "---\n"
        "Return the input value.\n",
        encoding="utf-8",
    )
    (problem_dir / "tests.json").write_text(json.dumps(tests_payload), encoding="utf-8")
    (problem_dir / "starter.py").write_text(solution_body, encoding="utf-8")
    return problem_dir


def test_execute_solution_passes_in_subprocess_sandbox(tmp_path: Path) -> None:
    problem_dir = _build_problem_dir(
        tmp_path,
        "def echo_value(value):\n    return value\n",
        [{"input": {"value": 7}, "expected": 7, "validator": "exact_match", "source": "verified"}],
    )
    problem = parse_problem_directory(problem_dir)

    result = execute_solution(problem)

    assert result.status == "pass"
    assert result.bundled_passed == 1
    assert result.bundled_total == 1
    assert result.llm_passed == 0
    assert result.verdicts[0].passed is True


def test_execute_solution_reports_runtime_errors_from_subprocess(tmp_path: Path) -> None:
    problem_dir = _build_problem_dir(
        tmp_path,
        "def echo_value(value):\n    raise ValueError('boom')\n",
        [{"input": {"value": 7}, "expected": 7, "validator": "exact_match", "source": "verified"}],
    )
    problem = parse_problem_directory(problem_dir)

    result = execute_solution(problem)

    assert result.status == "fail"
    assert result.verdicts[0].passed is False
    assert "boom" in (result.verdicts[0].error or "")


def test_execute_solution_reports_timeout_status(tmp_path: Path) -> None:
    problem_dir = _build_problem_dir(
        tmp_path,
        "import time\n\ndef echo_value(value):\n    time.sleep(5)\n    return value\n",
        [{"input": {"value": 7}, "expected": 7, "validator": "exact_match", "source": "verified"}],
    )
    problem = parse_problem_directory(problem_dir)

    result = execute_solution(problem, timeout_seconds=0.2)

    assert result.status == "TLE"
    assert result.verdicts[0].passed is False
    assert "Timed out" in (result.verdicts[0].error or "")
