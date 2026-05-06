from __future__ import annotations

from pathlib import Path
from random import Random
from textwrap import dedent

import pytest

from lcgrade.db import (
    bootstrap_database,
    fetch_global_progress_counts,
    fetch_progress_counts_by_difficulty,
    fetch_problem_history_view,
    fetch_weakest_tags_by_bundled_pass_rate,
    insert_attempt,
    select_random_unsolved_problem,
    set_problem_milestone,
)
from lcgrade.execution import hash_text
from lcgrade.problems import index_problem_bank
from lcgrade.reviews import persist_extension_result, persist_review


def _write_problem(
    root: Path,
    *,
    slug: str,
    title: str,
    difficulty: str,
    tags: list[str],
    category: str = "blind75",
) -> None:
    problem_dir = root / slug
    problem_dir.mkdir(parents=True, exist_ok=True)
    tags_text = ", ".join(tags)
    statement = dedent(
        f"""\
        ---
        schema_version: 1
        title: {title}
        slug: {slug}
        difficulty: {difficulty}
        tags: [{tags_text}]
        category: {category}
        function_name: solve
        params: []
        return_type: int
        validator: exact_match
        ---

        # {title}

        Synthetic problem used for database query tests.
        """
    )
    (problem_dir / "statement.md").write_text(statement, encoding="utf-8")


@pytest.fixture()
def seeded_conn(tmp_path: Path):
    problems_root = tmp_path / "problems"
    _write_problem(problems_root, slug="alpha-array", title="Alpha Array", difficulty="easy", tags=["array"])
    _write_problem(
        problems_root,
        slug="beta-array-graph",
        title="Beta Array Graph",
        difficulty="medium",
        tags=["array", "graph"],
    )
    _write_problem(
        problems_root,
        slug="gamma-graph-dp",
        title="Gamma Graph DP",
        difficulty="hard",
        tags=["graph", "dp"],
    )
    _write_problem(
        problems_root,
        slug="delta-queue",
        title="Delta Queue",
        difficulty="medium",
        tags=["queue"],
    )

    conn = bootstrap_database(tmp_path / "lcgrade.db")
    index_problem_bank(conn, problems_root)
    try:
        yield conn
    finally:
        conn.close()


def _seed_attempt(
    conn,
    *,
    slug: str,
    code_snapshot: str,
    bundled_passed: int,
    bundled_total: int,
    status: str = "pass",
) -> int:
    return insert_attempt(
        conn,
        slug=slug,
        test_mode="both",
        code_hash=hash_text(code_snapshot),
        tests_hash="tests-hash",
        bundled_passed=bundled_passed,
        bundled_total=bundled_total,
        llm_passed=0,
        llm_total=0,
        runtime_ms=1.0,
        status=status,
        code_snapshot=code_snapshot,
    )


def test_global_and_difficulty_progress_counts_reflect_problem_state(seeded_conn) -> None:
    _seed_attempt(
        seeded_conn,
        slug="alpha-array",
        code_snapshot="def alpha():\n    return 1\n",
        bundled_passed=2,
        bundled_total=2,
    )
    _seed_attempt(
        seeded_conn,
        slug="beta-array-graph",
        code_snapshot="def beta():\n    return 0\n",
        bundled_passed=0,
        bundled_total=4,
        status="fail",
    )
    _seed_attempt(
        seeded_conn,
        slug="gamma-graph-dp",
        code_snapshot="def gamma():\n    return 3\n",
        bundled_passed=1,
        bundled_total=2,
        status="fail",
    )

    set_problem_milestone(seeded_conn, slug="alpha-array", flag_column="auto_solved", timestamp_column="auto_solved_at")
    set_problem_milestone(
        seeded_conn,
        slug="alpha-array",
        flag_column="review_generated",
        timestamp_column="review_generated_at",
    )
    set_problem_milestone(
        seeded_conn,
        slug="alpha-array",
        flag_column="review_acknowledged",
        timestamp_column="review_acknowledged_at",
    )
    set_problem_milestone(
        seeded_conn,
        slug="alpha-array",
        flag_column="followup_completed",
        timestamp_column="followup_completed_at",
    )
    set_problem_milestone(
        seeded_conn,
        slug="gamma-graph-dp",
        flag_column="manual_solved",
        timestamp_column="manual_solved_at",
    )

    global_counts = fetch_global_progress_counts(seeded_conn)
    assert global_counts == {
        "total_problems": 4,
        "attempted": 3,
        "solved": 2,
        "auto_solved": 1,
        "manual_solved": 1,
        "review_generated": 1,
        "review_acknowledged": 1,
        "followup_completed": 1,
    }

    by_difficulty = {item["difficulty"]: item for item in fetch_progress_counts_by_difficulty(seeded_conn)}
    assert by_difficulty == {
        "easy": {
            "difficulty": "easy",
            "total": 1,
            "attempted": 1,
            "solved": 1,
            "review_generated": 1,
            "review_acknowledged": 1,
            "followup_completed": 1,
        },
        "medium": {
            "difficulty": "medium",
            "total": 2,
            "attempted": 1,
            "solved": 0,
            "review_generated": 0,
            "review_acknowledged": 0,
            "followup_completed": 0,
        },
        "hard": {
            "difficulty": "hard",
            "total": 1,
            "attempted": 1,
            "solved": 1,
            "review_generated": 0,
            "review_acknowledged": 0,
            "followup_completed": 0,
        },
    }


def test_weakest_tags_rank_by_bundled_pass_rate(seeded_conn) -> None:
    _seed_attempt(
        seeded_conn,
        slug="alpha-array",
        code_snapshot="def alpha():\n    return 1\n",
        bundled_passed=2,
        bundled_total=2,
    )
    _seed_attempt(
        seeded_conn,
        slug="beta-array-graph",
        code_snapshot="def beta():\n    return 0\n",
        bundled_passed=0,
        bundled_total=4,
        status="fail",
    )
    _seed_attempt(
        seeded_conn,
        slug="gamma-graph-dp",
        code_snapshot="def gamma():\n    return 3\n",
        bundled_passed=1,
        bundled_total=2,
        status="fail",
    )

    weakest = fetch_weakest_tags_by_bundled_pass_rate(seeded_conn, limit=3)

    assert [item["tag"] for item in weakest] == ["graph", "array", "dp"]
    assert weakest[0]["attempts"] == 2
    assert weakest[0]["bundled_passed"] == 1
    assert weakest[0]["bundled_total"] == 6
    assert weakest[0]["pass_rate"] == pytest.approx(1 / 6)
    assert weakest[1]["pass_rate"] == pytest.approx(2 / 6)
    assert weakest[2]["pass_rate"] == pytest.approx(1 / 2)


def test_problem_history_view_returns_attempts_reviews_and_milestones(seeded_conn) -> None:
    older_attempt = _seed_attempt(
        seeded_conn,
        slug="alpha-array",
        code_snapshot="def alpha():\n    return 0\n",
        bundled_passed=0,
        bundled_total=2,
        status="fail",
    )
    newer_attempt = _seed_attempt(
        seeded_conn,
        slug="alpha-array",
        code_snapshot="def alpha():\n    return 1\n",
        bundled_passed=2,
        bundled_total=2,
    )
    review_id = persist_review(
        seeded_conn,
        attempt_id=newer_attempt,
        review_text="Concise pass review.",
        complexity_time="O(n)",
        complexity_space="O(1)",
    )
    persist_extension_result(
        seeded_conn,
        review_id=review_id,
        extension_name="interview",
        output_text="Follow up on trade-offs.",
    )
    persist_extension_result(
        seeded_conn,
        review_id=review_id,
        extension_name="optimize",
        output_text="Consider an early return optimization.",
    )
    set_problem_milestone(seeded_conn, slug="alpha-array", flag_column="auto_solved", timestamp_column="auto_solved_at")
    set_problem_milestone(
        seeded_conn,
        slug="alpha-array",
        flag_column="review_generated",
        timestamp_column="review_generated_at",
    )
    set_problem_milestone(
        seeded_conn,
        slug="alpha-array",
        flag_column="review_acknowledged",
        timestamp_column="review_acknowledged_at",
    )

    history = fetch_problem_history_view(seeded_conn, "alpha-array")

    assert history is not None
    assert history["problem"]["slug"] == "alpha-array"
    assert history["milestones"]["auto_solved"] is True
    assert history["milestones"]["review_generated"] is True
    assert history["milestones"]["review_acknowledged"] is True
    assert history["milestones"]["followup_completed"] is False
    assert history["milestones"]["auto_solved_at"] is not None
    assert len(history["attempts"]) == 2
    assert history["attempts"][0]["attempt_number"] == 1
    assert history["attempts"][0]["attempt"]["id"] == newer_attempt
    assert history["attempts"][0]["review"]["attempt_id"] == newer_attempt
    assert history["attempts"][0]["review"]["review_text"] == "Concise pass review."
    assert [item["extension_name"] for item in history["attempts"][0]["extensions"]] == [
        "interview",
        "optimize",
    ]
    assert history["attempts"][1]["attempt_number"] == 2
    assert history["attempts"][1]["attempt"]["id"] == older_attempt
    assert history["attempts"][1]["review"] is None
    assert history["attempts"][1]["extensions"] == []


def test_problem_history_view_handles_empty_attempt_history(seeded_conn) -> None:
    history = fetch_problem_history_view(seeded_conn, "delta-queue")

    assert history is not None
    assert history["problem"]["slug"] == "delta-queue"
    assert history["attempts"] == []
    assert history["milestones"] == {
        "auto_solved": False,
        "auto_solved_at": None,
        "manual_solved": False,
        "manual_solved_at": None,
        "review_generated": False,
        "review_generated_at": None,
        "review_acknowledged": False,
        "review_acknowledged_at": None,
        "followup_completed": False,
        "followup_completed_at": None,
    }


def test_random_unsolved_problem_respects_filters_and_is_repeatable(seeded_conn) -> None:
    _seed_attempt(
        seeded_conn,
        slug="alpha-array",
        code_snapshot="def alpha():\n    return 1\n",
        bundled_passed=2,
        bundled_total=2,
    )
    _seed_attempt(
        seeded_conn,
        slug="gamma-graph-dp",
        code_snapshot="def gamma():\n    return 3\n",
        bundled_passed=1,
        bundled_total=2,
        status="fail",
    )
    set_problem_milestone(seeded_conn, slug="alpha-array", flag_column="auto_solved", timestamp_column="auto_solved_at")
    set_problem_milestone(
        seeded_conn,
        slug="gamma-graph-dp",
        flag_column="manual_solved",
        timestamp_column="manual_solved_at",
    )

    first_pick = select_random_unsolved_problem(seeded_conn, rng=Random(0))
    second_pick = select_random_unsolved_problem(seeded_conn, rng=Random(0))
    assert first_pick is not None
    assert second_pick is not None
    assert first_pick["slug"] == second_pick["slug"]
    assert first_pick["slug"] in {"beta-array-graph", "delta-queue"}

    filtered = select_random_unsolved_problem(
        seeded_conn,
        difficulty="medium",
        tag="queue",
        rng=Random(0),
    )
    assert filtered is not None
    assert filtered["slug"] == "delta-queue"

    no_match = select_random_unsolved_problem(
        seeded_conn,
        difficulty="hard",
        tag="dp",
        rng=Random(0),
    )
    assert no_match is None
