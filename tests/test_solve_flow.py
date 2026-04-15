from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from lcgrade.db import bootstrap_database, fetch_problem_record, upsert_problem_record
from lcgrade.execution import ExecutionSummary, hash_text
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


def build_execution(problem: ProblemDocument, code_snapshot: str) -> ExecutionSummary:
    return ExecutionSummary(
        verdicts=[],
        bundled_passed=2,
        bundled_total=2,
        runtime_ms=1.5,
        status="pass",
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

    def executor(executed_problem: ProblemDocument, solution_path: Path | None) -> ExecutionSummary:
        stage1_calls.append(executed_problem.slug)
        assert solution_path == problem.starter_path
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

    def executor(executed_problem: ProblemDocument, solution_path: Path | None) -> ExecutionSummary:
        stage1_calls.append(executed_problem.slug)
        assert solution_path == problem.starter_path
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

    def executor(executed_problem: ProblemDocument, solution_path: Path | None) -> ExecutionSummary:
        assert solution_path == problem.starter_path
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
