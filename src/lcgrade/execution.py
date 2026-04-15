from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import time
from types import ModuleType
from typing import Any, Callable

from .problems import ProblemDocument
from .types import TestCase, TestVerdict
from .validators import get_validator


class ExecutionError(RuntimeError):
    """Raised when a solution file cannot be loaded or executed."""


@dataclass(slots=True)
class ExecutionSummary:
    verdicts: list[TestVerdict]
    bundled_passed: int
    bundled_total: int
    runtime_ms: float
    status: str
    code_snapshot: str
    code_hash: str
    tests_hash: str


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
) -> ExecutionSummary:
    resolved_solution = solution_path or problem.starter_path
    if resolved_solution is None or not resolved_solution.exists():
        raise ExecutionError(f"No solution file found for {problem.slug}")

    code_snapshot = resolved_solution.read_text(encoding="utf-8")
    code_hash = hash_text(code_snapshot)
    tests_hash = problem.tests_hash or hash_text("[]")
    function = load_solution_function(
        resolved_solution,
        problem.metadata.function_name,
    )
    test_cases = load_test_cases(problem)

    verdicts: list[TestVerdict] = []
    started = time.perf_counter()
    for test_case in test_cases:
        verdicts.append(run_test_case(function, test_case))
    runtime_ms = (time.perf_counter() - started) * 1000.0

    bundled_passed = sum(1 for verdict in verdicts if verdict.passed)
    bundled_total = len(verdicts)
    status = "pass" if bundled_passed == bundled_total else "fail"
    if bundled_total == 0:
        status = "error"

    return ExecutionSummary(
        verdicts=verdicts,
        bundled_passed=bundled_passed,
        bundled_total=bundled_total,
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
