from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import ModuleType
from typing import Any, Callable, Sequence

from .problems import ProblemDocument
from .preflight import run_preflight
from .sandbox import run_python_script
from .types import TestCase, TestVerdict
from .validators import get_validator


class ExecutionError(RuntimeError):
    """Raised when a solution file cannot be loaded or executed."""


@dataclass(slots=True)
class ExecutionSummary:
    verdicts: list[TestVerdict]
    bundled_passed: int
    bundled_total: int
    llm_passed: int
    llm_total: int
    runtime_ms: float
    status: str
    code_snapshot: str
    code_hash: str
    tests_hash: str


_SANDBOX_RUNNER_SOURCE = """\
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    solution_path = Path(sys.argv[1])
    function_name = sys.argv[2]
    payload = json.loads(sys.stdin.read() or "{}")

    spec = importlib.util.spec_from_file_location(solution_path.stem, solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {solution_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise RuntimeError(f"Expected callable {function_name!r} in {solution_path}")

    actual = function(**payload)
    print(json.dumps({"ok": True, "actual": actual}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        raise
"""


def hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def load_test_cases(problem: ProblemDocument) -> list[TestCase]:
    if problem.tests_path is None or not problem.tests_path.exists():
        return []

    payload = json.loads(problem.tests_path.read_text(encoding="utf-8"))
    test_cases: list[TestCase] = []
    for index, item in enumerate(payload, start=1):
        test_cases.append(
            TestCase(
                name=f"bundled-{index}",
                input=dict(item["input"]),
                expected=item["expected"],
                validator=str(item.get("validator", problem.metadata.validator)),
                source=str(item.get("source", "verified")),
            )
        )
    return test_cases


def execute_solution(
    problem: ProblemDocument,
    solution_path: Path | None = None,
    test_cases: Sequence[TestCase] | None = None,
    timeout_seconds: float = 2.0,
) -> ExecutionSummary:
    resolved_solution = solution_path or problem.starter_path
    if resolved_solution is None or not resolved_solution.exists():
        raise ExecutionError(f"No solution file found for {problem.slug}")

    run_preflight(problem, resolved_solution)
    code_snapshot = resolved_solution.read_text(encoding="utf-8")
    code_hash = hash_text(code_snapshot)
    tests_hash = problem.tests_hash or hash_text("[]")
    active_test_cases = list(test_cases) if test_cases is not None else load_test_cases(problem)

    verdicts: list[TestVerdict] = []
    started = time.perf_counter()
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="lcgrade-sandbox-") as tmp_dir:
        runner_path = Path(tmp_dir) / "runner.py"
        runner_path.write_text(_SANDBOX_RUNNER_SOURCE, encoding="utf-8")
        for test_case in active_test_cases:
            verdict = run_test_case_in_sandbox(
                runner_path=runner_path,
                solution_path=resolved_solution,
                function_name=problem.metadata.function_name,
                test_case=test_case,
                timeout_seconds=timeout_seconds,
            )
            if verdict.error and verdict.error.startswith("Timed out"):
                timed_out = True
            verdicts.append(verdict)
    runtime_ms = (time.perf_counter() - started) * 1000.0

    bundled_verdicts = [verdict for verdict, test_case in zip(verdicts, active_test_cases) if test_case.source == "verified"]
    llm_verdicts = [verdict for verdict, test_case in zip(verdicts, active_test_cases) if test_case.source != "verified"]
    bundled_passed = sum(1 for verdict in bundled_verdicts if verdict.passed)
    bundled_total = len(bundled_verdicts)
    llm_passed = sum(1 for verdict in llm_verdicts if verdict.passed)
    llm_total = len(llm_verdicts)
    total_tests = len(active_test_cases)
    status = "pass" if sum(1 for verdict in verdicts if verdict.passed) == total_tests else "fail"
    if total_tests == 0:
        status = "error"
    elif timed_out:
        status = "TLE"

    return ExecutionSummary(
        verdicts=verdicts,
        bundled_passed=bundled_passed,
        bundled_total=bundled_total,
        llm_passed=llm_passed,
        llm_total=llm_total,
        runtime_ms=runtime_ms,
        status=status,
        code_snapshot=code_snapshot,
        code_hash=code_hash,
        tests_hash=tests_hash,
    )


def load_solution_function(solution_path: Path, function_name: str) -> Callable[..., Any]:
    module = load_python_module(solution_path)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ExecutionError(f"Expected callable {function_name!r} in {solution_path}")
    return function


def load_python_module(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ExecutionError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - exercised indirectly
        raise ExecutionError(f"Failed to import {module_path}: {exc}") from exc
    return module


def run_test_case(function: Callable[..., Any], test_case: TestCase) -> TestVerdict:
    validator = get_validator(test_case.validator)
    try:
        actual = function(**test_case.input)
        passed = validator.check(test_case.expected, actual, test_case.input)
        return TestVerdict(
            name=test_case.name,
            passed=passed,
            input=test_case.input,
            expected=test_case.expected,
            actual=actual,
            source=test_case.source,
        )
    except Exception as exc:
        return TestVerdict(
            name=test_case.name,
            passed=False,
            input=test_case.input,
            expected=test_case.expected,
            error=str(exc),
            source=test_case.source,
        )


def run_test_case_in_sandbox(
    *,
    runner_path: Path,
    solution_path: Path,
    function_name: str,
    test_case: TestCase,
    timeout_seconds: float,
) -> TestVerdict:
    validator = get_validator(test_case.validator)
    payload = json.dumps(test_case.input, ensure_ascii=False)
    sandbox_run = run_python_script(
        runner_path,
        args=(str(solution_path), function_name),
        stdin_text=payload,
        timeout_seconds=timeout_seconds,
    )
    if sandbox_run.timed_out:
        return TestVerdict(
            name=test_case.name,
            passed=False,
            input=test_case.input,
            expected=test_case.expected,
            error=f"Timed out after {timeout_seconds:.2f}s",
            source=test_case.source,
        )

    stdout = sandbox_run.stdout.strip()
    stderr = sandbox_run.stderr.strip()
    if not stdout:
        return TestVerdict(
            name=test_case.name,
            passed=False,
            input=test_case.input,
            expected=test_case.expected,
            error=stderr or "Sandbox produced no output.",
            source=test_case.source,
        )

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return TestVerdict(
            name=test_case.name,
            passed=False,
            input=test_case.input,
            expected=test_case.expected,
            error=stderr or stdout,
            source=test_case.source,
        )

    if not payload.get("ok", False):
        return TestVerdict(
            name=test_case.name,
            passed=False,
            input=test_case.input,
            expected=test_case.expected,
            error=str(payload.get("error") or stderr or "Runtime error"),
            source=test_case.source,
        )

    actual = payload.get("actual")
    passed = validator.check(test_case.expected, actual, test_case.input)
    return TestVerdict(
        name=test_case.name,
        passed=passed,
        input=test_case.input,
        expected=test_case.expected,
        actual=actual,
        source=test_case.source,
    )
