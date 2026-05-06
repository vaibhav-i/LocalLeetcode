from __future__ import annotations

from pathlib import Path

import pytest

from lcgrade.db import bootstrap_database, fetch_problem_record, insert_attempt
from lcgrade.execution import hash_text
from lcgrade.llm import LLMBackend, LLMResponse, MockBackend, ModelInfo
from lcgrade.problems import index_problem_bank, load_problem_by_slug
from lcgrade.reviews import (
    fetch_latest_attempt,
    generate_beta_stage2_review,
    persist_extension_results,
    review_problem,
    run_registered_extensions,
)


@pytest.fixture()
def db_conn(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    conn = bootstrap_database(tmp_path / "lcgrade.db")
    index_problem_bank(conn, project_root / "problems")
    try:
        yield conn
    finally:
        conn.close()


def _seed_attempt(conn, *, slug: str, code_snapshot: str, status: str = "pass") -> int:
    return insert_attempt(
        conn,
        slug=slug,
        test_mode="both",
        code_hash=hash_text(code_snapshot),
        tests_hash="tests-hash",
        bundled_passed=2,
        bundled_total=2,
        llm_passed=0,
        llm_total=0,
        runtime_ms=1.25,
        status=status,
        code_snapshot=code_snapshot,
    )


def test_fetch_latest_attempt_returns_most_recent_row(db_conn) -> None:
    first_id = _seed_attempt(db_conn, slug="two-sum", code_snapshot="def a():\n    return 1\n")
    second_id = _seed_attempt(db_conn, slug="two-sum", code_snapshot="def b():\n    return 2\n")

    latest = fetch_latest_attempt(db_conn, "two-sum")

    assert latest is not None
    assert int(latest["id"]) == second_id
    assert int(latest["id"]) != first_id
    assert latest["code_snapshot"] == "def b():\n    return 2\n"


def test_review_problem_generates_stage2_and_extensions(db_conn) -> None:
    code_snapshot = (Path(__file__).resolve().parents[1] / "problems" / "two-sum" / "solutions" / "reference.py").read_text(encoding="utf-8")
    attempt_id = _seed_attempt(db_conn, slug="two-sum", code_snapshot=code_snapshot)

    backend = MockBackend(
        responses=[
            "## Complexity\nO(n)\n\n## Correctness\nThe hash map approach is correct.\n\n## Code Quality\nClear and concise.\n\n## Edge Cases\nHandles duplicates and repeated values.\n\n## Verdict\nStrong pass.",
            "Follow up on why the hash map lookup is safe.",
            "Try to explain any space-usage trade-offs.",
        ]
    )

    result = review_problem(db_conn, "two-sum", llm=backend)

    assert result.skipped_reason is None
    assert result.attempt_id == attempt_id
    assert result.review_id is not None
    assert result.review_text is not None
    assert "hash map" in result.review_text
    assert [item.title for item in result.extension_results] == [
        "Interview Follow-ups",
        "Optimization Suggestions",
    ]
    assert [item.name for item in result.extension_results] == [
        "interview",
        "optimize",
    ]
    assert backend.prompts[0]["prompt"].startswith("You are lcgrade's Stage 2 reviewer")
    assert "Submitted code:" in backend.prompts[0]["prompt"]
    assert code_snapshot.strip() in backend.prompts[0]["prompt"]
    assert "Verdicts:" in backend.prompts[1]["prompt"]
    assert "bundled-tests" in backend.prompts[1]["prompt"]

    review_row = db_conn.execute("SELECT * FROM reviews WHERE attempt_id = ?", (attempt_id,)).fetchone()
    assert review_row is not None
    assert "hash map" in review_row["review_text"]

    extension_rows = db_conn.execute(
        "SELECT extension_name, output_text FROM extension_results WHERE review_id = ? ORDER BY id",
        (result.review_id,),
    ).fetchall()
    assert [row["extension_name"] for row in extension_rows] == [
        "interview",
        "optimize",
    ]
    assert "hash map lookup" in extension_rows[0]["output_text"]
    assert "space-usage trade-offs" in extension_rows[1]["output_text"]


def test_review_problem_skips_when_backend_unavailable(db_conn) -> None:
    code_snapshot = "def two_sum(nums, target):\n    return []\n"
    _seed_attempt(db_conn, slug="two-sum", code_snapshot=code_snapshot)

    class UnavailableBackend(LLMBackend):
        def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.7, max_tokens: int = 2048) -> LLMResponse:
            raise AssertionError("generate() should not be called when backend is unavailable")

        def available(self) -> bool:
            return False

        def model_info(self) -> ModelInfo:
            return ModelInfo(name="unavailable", context_window=0, quantization="unknown", backend="mock")

    result = review_problem(db_conn, "two-sum", llm=UnavailableBackend())

    assert result.review_text is None
    assert result.review_id is None
    assert result.skipped_reason is not None
    assert "This feature needs a local LLM backend." in result.skipped_reason
    assert "Core lcgrade solving still works without one." in result.skipped_reason
    assert db_conn.execute("SELECT COUNT(*) AS count FROM reviews").fetchone()["count"] == 0
    assert db_conn.execute("SELECT COUNT(*) AS count FROM extension_results").fetchone()["count"] == 0


def test_review_problem_marks_problem_review_generated(db_conn) -> None:
    code_snapshot = (Path(__file__).resolve().parents[1] / "problems" / "two-sum" / "solutions" / "reference.py").read_text(encoding="utf-8")
    _seed_attempt(db_conn, slug="two-sum", code_snapshot=code_snapshot)
    backend = MockBackend(
        responses=[
            "## Complexity\nO(n)\n\n## Correctness\nCorrect.\n\n## Code Quality\nClear.\n\n## Edge Cases\nReasonable.\n\n## Verdict\nPass.",
            "One follow-up.",
            "One optimization.",
        ]
    )

    result = review_problem(db_conn, "two-sum", llm=backend)

    assert result.review_id is not None
    record = fetch_problem_record(db_conn, "two-sum")
    assert record is not None
    assert int(record["review_generated"]) == 1
    assert record["review_generated_at"] is not None
