from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from lcgrade.db import bootstrap_database, fetch_problem_record, upsert_problem_record
from lcgrade.execution import ExecutionSummary, hash_text
from lcgrade.llm import MockBackend
from lcgrade.problems import ProblemDocument, ProblemMetadata, ProblemParam
from lcgrade.solve_flow import solve_problem


def build_problem(tmp_path: Path) -> ProblemDocument:
    problem_dir = tmp_path / "two-sum"
    problem_dir.mkdir()
    statement_path = problem_dir / "statement.md"
    statement_path.write_text("---\nslug: two-sum\ntitle: Two Sum\ndifficulty: easy\nfunction_name: two_sum\nparams:\n  - name: nums\n    type: list[int]\n  - name: target\n    type: int\n---\n", encoding="utf-8")
    tests_path = problem_dir / "tests.json"
    tests_path.write_text("[]", encoding="utf-8")
    starter_path = problem_dir / "starter.py"
    starter_text = "def two_sum(nums, target):\n    return []\n"
    starter_path.write_text(starter_text, encoding="utf-8")

    metadata = ProblemMetadata(
        schema_version=1,
        title="Two Sum",
        slug="two-sum",
        difficulty="easy",
        tags=("array", "hash-map"),
        category="Arrays",
        function_name="two_sum",
        params=(ProblemParam(name="nums", type="list[int]"), ProblemParam(name="target", type="int")),
        return_type="list[int]",
        validator="exact_match",
    )
    return ProblemDocument(
        metadata=metadata,
        body="",
        problem_dir=problem_dir,
        statement_path=statement_path,
        tests_path=tests_path,
        starter_path=starter_path,
        validator_path=None,
        reference_solution_path=None,
        statement_hash="statement-hash",
        statement_mtime=1.0,
        tests_hash="tests-hash",
        tests_mtime=1.0,
    )


def build_execution(
    problem: ProblemDocument,
    code_snapshot: str,
    *,
    bundled_passed: int = 2,
    bundled_total: int = 2,
    llm_passed: int = 0,
    llm_total: int = 0,
) -> ExecutionSummary:
    return ExecutionSummary(
        verdicts=[],
        bundled_passed=bundled_passed,
        bundled_total=bundled_total,
        llm_passed=llm_passed,
        llm_total=llm_total,
        runtime_ms=1.5,
        status="pass" if bundled_passed == bundled_total and llm_passed == llm_total else "fail",
        code_snapshot=code_snapshot,
        code_hash=hash_text(code_snapshot),
        tests_hash=problem.tests_hash or hash_text("[]"),
    )


@pytest.fixture()
def problem_and_db(tmp_path: Path) -> tuple[sqlite3.Connection, ProblemDocument]:
    conn = bootstrap_database(tmp_path / "lcgrade.db")
    problem = build_problem(tmp_path)
    upsert_problem_record(conn, problem.to_db_record())
    try:
        yield conn, problem
    finally:
        conn.close()


def test_cache_hit_does_not_re_execute_stage1(problem_and_db: tuple[sqlite3.Connection, ProblemDocument]) -> None:
    conn, problem = problem_and_db
    stage1_calls: list[str] = []
    code_snapshot = problem.starter_path.read_text(encoding="utf-8")

    def executor(
        executed_problem: ProblemDocument,
        solution_path: Path | None,
        test_cases: list | tuple | None,
    ) -> ExecutionSummary:
        stage1_calls.append(executed_problem.slug)
        assert solution_path == problem.starter_path
        assert test_cases is not None
        return build_execution(executed_problem, code_snapshot)

    first = solve_problem(
        conn,
        problem,
        requested_test_mode="both",
        executor=executor,
    )
    second = solve_problem(
        conn,
        problem,
        requested_test_mode="both",
        executor=executor,
    )

    assert stage1_calls == ["two-sum"]
    assert first.used_cache is False
    assert first.requested_test_mode == "both"
    assert first.effective_test_mode == "bundled"
    assert first.cache_key.test_mode == "both"
    assert second.used_cache is True
    assert second.cached_attempt is not None
    assert second.execution is None
    assert second.cached_attempt.id == first.attempt_id


def test_force_bypasses_existing_cache(problem_and_db: tuple[sqlite3.Connection, ProblemDocument]) -> None:
    conn, problem = problem_and_db
    stage1_calls: list[str] = []
    code_snapshot = problem.starter_path.read_text(encoding="utf-8")

    def executor(
        executed_problem: ProblemDocument,
        solution_path: Path | None,
        test_cases: list | tuple | None,
    ) -> ExecutionSummary:
        stage1_calls.append(executed_problem.slug)
        assert solution_path == problem.starter_path
        assert test_cases is not None
        return build_execution(executed_problem, code_snapshot)

    first = solve_problem(
        conn,
        problem,
        requested_test_mode="llm",
        executor=executor,
    )
    forced = solve_problem(
        conn,
        problem,
        requested_test_mode="llm",
        force=True,
        executor=executor,
    )

    assert first.used_cache is False
    assert forced.used_cache is False
    assert stage1_calls == ["two-sum", "two-sum"]
    assert forced.requested_test_mode == "llm"
    assert forced.effective_test_mode == "bundled"
    assert forced.cached_attempt is None


def test_solve_marks_problem_auto_solved_on_first_full_bundled_pass(
    problem_and_db: tuple[sqlite3.Connection, ProblemDocument],
) -> None:
    conn, problem = problem_and_db
    code_snapshot = problem.starter_path.read_text(encoding="utf-8")

    def executor(
        executed_problem: ProblemDocument,
        solution_path: Path | None,
        test_cases: list | tuple | None,
    ) -> ExecutionSummary:
        assert solution_path == problem.starter_path
        assert test_cases is not None
        return build_execution(executed_problem, code_snapshot)

    solve_problem(
        conn,
        problem,
        requested_test_mode="bundled",
        executor=executor,
    )

    record = fetch_problem_record(conn, problem.slug)
    assert record is not None
    assert int(record["auto_solved"]) == 1
    assert record["auto_solved_at"] is not None


def test_llm_mode_runs_generated_cases_when_backend_available(
    problem_and_db: tuple[sqlite3.Connection, ProblemDocument],
) -> None:
    conn, problem = problem_and_db
    code_snapshot = problem.starter_path.read_text(encoding="utf-8")
    backend = MockBackend(
        responses=[
            '{"test_cases":[{"name":"llm-1","input":{"nums":[],"target":0},"expected":[],"validator":"exact_match","source":"llm","rationale":"empty"}]}'
        ]
    )
    seen_test_case_count: list[int] = []

    def executor(
        executed_problem: ProblemDocument,
        solution_path: Path | None,
        test_cases: list | tuple | None,
    ) -> ExecutionSummary:
        assert solution_path == problem.starter_path
        assert test_cases is not None
        seen_test_case_count.append(len(test_cases))
        return build_execution(executed_problem, code_snapshot, bundled_passed=1, bundled_total=1, llm_passed=1, llm_total=1)

    result = solve_problem(
        conn,
        problem,
        requested_test_mode="llm",
        llm=backend,
        executor=executor,
    )

    assert seen_test_case_count == [1]
    assert result.requested_test_mode == "llm"
    assert result.effective_test_mode == "llm"
    assert result.attempt.llm_passed == 1
    assert result.attempt.llm_total == 1
    assert result.test_generation_result is not None
    assert result.test_generation_result.llm_used is True


def test_both_mode_falls_back_to_bundled_only_when_backend_unavailable(
    problem_and_db: tuple[sqlite3.Connection, ProblemDocument],
) -> None:
    conn, problem = problem_and_db
    code_snapshot = problem.starter_path.read_text(encoding="utf-8")
    seen_test_case_count: list[int] = []

    def executor(
        executed_problem: ProblemDocument,
        solution_path: Path | None,
        test_cases: list | tuple | None,
    ) -> ExecutionSummary:
        assert solution_path == problem.starter_path
        assert test_cases is not None
        seen_test_case_count.append(len(test_cases))
        return build_execution(executed_problem, code_snapshot, bundled_passed=1, bundled_total=1)

    result = solve_problem(
        conn,
        problem,
        requested_test_mode="both",
        llm=None,
        executor=executor,
    )

    assert seen_test_case_count == [0]
    assert result.requested_test_mode == "both"
    assert result.effective_test_mode == "bundled"
    assert result.attempt.llm_passed == 0
    assert result.attempt.llm_total == 0
    assert result.test_generation_result is not None
    assert result.test_generation_result.warning == "LLM backend unavailable; using bundled tests only."
