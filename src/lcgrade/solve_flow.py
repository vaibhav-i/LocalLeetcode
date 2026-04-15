from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Callable, Mapping, Sequence

from .db import CacheKey, find_cached_attempt, insert_attempt, mark_problem_auto_solved
from .execution import ExecutionSummary, execute_solution, hash_text
from .llm import LLMBackend
from .problems import ProblemDocument
from .test_generation import TestGenerationResult, generate_test_generation
from .types import TestCase
from .types import AttemptSummary


class SolveFlowError(RuntimeError):
    """Raised when the solve flow cannot prepare or execute a run."""


@dataclass(slots=True, frozen=True)
class SolutionSnapshot:
    solution_path: Path
    code_snapshot: str
    code_hash: str
    tests_hash: str


@dataclass(slots=True, frozen=True)
class CachedAttemptRecord:
    id: int
    slug: str
    timestamp: str
    test_mode: str
    code_hash: str
    tests_hash: str
    bundled_passed: int
    bundled_total: int
    llm_passed: int
    llm_total: int
    runtime_ms: float
    status: str
    code_snapshot: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CachedAttemptRecord":
        return cls(
            id=int(mapping["id"]),
            slug=str(mapping["slug"]),
            timestamp=str(mapping["timestamp"]),
            test_mode=str(mapping["test_mode"]),
            code_hash=str(mapping["code_hash"]),
            tests_hash=str(mapping["tests_hash"]),
            bundled_passed=int(mapping["bundled_passed"]),
            bundled_total=int(mapping["bundled_total"]),
            llm_passed=int(mapping["llm_passed"]),
            llm_total=int(mapping["llm_total"]),
            runtime_ms=float(mapping["runtime_ms"]),
            status=str(mapping["status"]),
            code_snapshot=str(mapping["code_snapshot"]),
        )

    def to_attempt_summary(self) -> AttemptSummary:
        return AttemptSummary(
            slug=self.slug,
            test_mode=self.test_mode,
            bundled_passed=self.bundled_passed,
            bundled_total=self.bundled_total,
            llm_passed=self.llm_passed,
            llm_total=self.llm_total,
            status=self.status,
            runtime_ms=self.runtime_ms,
        )


@dataclass(slots=True)
class SolveFlowResult:
    problem: ProblemDocument
    requested_test_mode: str
    effective_test_mode: str
    force: bool
    used_cache: bool
    cache_key: CacheKey
    solution_snapshot: SolutionSnapshot
    attempt: AttemptSummary
    attempt_id: int | None
    cached_attempt: CachedAttemptRecord | None = None
    execution: ExecutionSummary | None = None
    test_generation_result: TestGenerationResult | None = None


def prepare_solution_snapshot(
    problem: ProblemDocument,
    solution_path: Path | None = None,
) -> SolutionSnapshot:
    resolved_solution = solution_path or problem.starter_path
    if resolved_solution is None or not resolved_solution.exists():
        raise SolveFlowError(f"No solution file found for {problem.slug}")

    code_snapshot = resolved_solution.read_text(encoding="utf-8")
    return SolutionSnapshot(
        solution_path=resolved_solution,
        code_snapshot=code_snapshot,
        code_hash=hash_text(code_snapshot),
        tests_hash=problem.tests_hash or hash_text("[]"),
    )


def _attempt_from_execution(
    problem: ProblemDocument,
    test_mode: str,
    execution: ExecutionSummary,
) -> AttemptSummary:
    return AttemptSummary(
        slug=problem.slug,
        test_mode=test_mode,
        bundled_passed=execution.bundled_passed,
        bundled_total=execution.bundled_total,
        llm_passed=execution.llm_passed,
        llm_total=execution.llm_total,
        status=execution.status,
        runtime_ms=execution.runtime_ms,
    )


def solve_problem(
    conn: sqlite3.Connection,
    problem: ProblemDocument,
    *,
    solution_path: Path | None = None,
    requested_test_mode: str = "both",
    force: bool = False,
    llm: LLMBackend | None = None,
    executor: Callable[[ProblemDocument, Path | None, Sequence[TestCase] | None], ExecutionSummary] = execute_solution,
    cache_lookup: Callable[[sqlite3.Connection, str, CacheKey], dict[str, Any] | None] = find_cached_attempt,
    save_attempt: Callable[..., int] = insert_attempt,
) -> SolveFlowResult:
    test_generation_result = generate_test_generation(
        problem,
        requested_test_mode=requested_test_mode,
        llm=llm,
    )
    normalized_test_mode = test_generation_result.requested_test_mode
    current_effective_test_mode = test_generation_result.effective_test_mode
    solution_snapshot = prepare_solution_snapshot(problem, solution_path)
    cache_key = CacheKey(
        code_hash=solution_snapshot.code_hash,
        test_mode=normalized_test_mode,
        tests_hash=solution_snapshot.tests_hash,
    )

    cached_attempt: CachedAttemptRecord | None = None
    if not force:
        cached_row = cache_lookup(conn, problem.slug, cache_key)
        if cached_row is not None:
            cached_attempt = CachedAttemptRecord.from_mapping(cached_row)
            return SolveFlowResult(
                problem=problem,
                requested_test_mode=normalized_test_mode,
                effective_test_mode=current_effective_test_mode,
                force=force,
                used_cache=True,
                cache_key=cache_key,
                solution_snapshot=solution_snapshot,
                attempt=cached_attempt.to_attempt_summary(),
                attempt_id=cached_attempt.id,
                cached_attempt=cached_attempt,
                execution=None,
                test_generation_result=test_generation_result,
            )

    execution = executor(
        problem,
        solution_snapshot.solution_path,
        test_generation_result.combined_test_cases(),
    )
    attempt = _attempt_from_execution(problem, normalized_test_mode, execution)
    attempt_id = save_attempt(
        conn,
        slug=problem.slug,
        test_mode=normalized_test_mode,
        code_hash=execution.code_hash,
        tests_hash=execution.tests_hash,
        bundled_passed=execution.bundled_passed,
        bundled_total=execution.bundled_total,
        llm_passed=execution.llm_passed,
        llm_total=execution.llm_total,
        runtime_ms=execution.runtime_ms,
        status=execution.status,
        code_snapshot=execution.code_snapshot,
    )
    if execution.bundled_total > 0 and execution.bundled_passed == execution.bundled_total:
        mark_problem_auto_solved(conn, problem.slug)
    return SolveFlowResult(
        problem=problem,
        requested_test_mode=normalized_test_mode,
        effective_test_mode=current_effective_test_mode,
        force=force,
        used_cache=False,
        cache_key=cache_key,
        solution_snapshot=solution_snapshot,
        attempt=attempt,
        attempt_id=attempt_id,
        cached_attempt=None,
        execution=execution,
        test_generation_result=test_generation_result,
    )
