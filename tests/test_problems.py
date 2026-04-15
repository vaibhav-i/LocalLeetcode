from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from lcgrade import db
from lcgrade.problems import index_problem_bank, parse_statement


def _write_problem(problem_dir: Path, *, tests_payload: list[dict[str, object]]) -> Path:
    problem_dir.mkdir(parents=True, exist_ok=True)
    statement_path = problem_dir / "statement.md"
    statement_path.write_text(
        dedent(
            """
            ---
            # beta frontmatter keeps comments and blank lines out of the payload
            schema_version: 2
            title: Two Sum
            slug: two-sum
            difficulty: Easy
            tags:
              - Array
              - Hash Map
            category: Arrays
            function_name: two_sum
            params:
              - name: nums
                type: list[int]
              - name: target
                type: int
            return_type: list[int]
            validator: set
            scaling_inputs:
              sizes: [10, 100]
              generator: pair_generator
              strategy: mixed
            extra_note: kept for downstream tools
            ---

            Find two indices whose values add up to the target.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (problem_dir / "tests.json").write_text(
        json.dumps(tests_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return statement_path


def test_parse_statement_reads_beta_frontmatter(tmp_path: Path) -> None:
    statement_path = _write_problem(
        tmp_path / "two-sum",
        tests_payload=[
            {
                "input": {"nums": [2, 7, 11, 15], "target": 9},
                "expected": [0, 1],
            }
        ],
    )

    document = parse_statement(statement_path)

    assert document.slug == "two-sum"
    assert document.body == "Find two indices whose values add up to the target."
    assert document.metadata.schema_version == 2
    assert document.metadata.difficulty == "easy"
    assert document.metadata.validator == "set_equality"
    assert document.metadata.tags == ("Array", "Hash Map")
    assert document.metadata.params[0].name == "nums"
    assert document.metadata.params[1].type == "int"
    assert document.metadata.scaling_inputs is not None
    assert document.metadata.scaling_inputs.sizes == (10, 100)
    assert document.metadata.scaling_inputs.extra == {"strategy": "mixed"}
    assert document.metadata.extras == {"extra_note": "kept for downstream tools"}
    assert document.statement_path == statement_path.resolve()
    assert document.tests_path == (statement_path.parent / "tests.json").resolve()
    assert document.statement_hash
    assert document.tests_hash


def test_index_problem_bank_reindexes_only_when_files_change(tmp_path: Path) -> None:
    problems_root = tmp_path / "problems"
    statement_path = _write_problem(
        problems_root / "two-sum",
        tests_payload=[
            {
                "input": {"nums": [2, 7, 11, 15], "target": 9},
                "expected": [0, 1],
            }
        ],
    )
    db_path = tmp_path / "lcgrade.db"
    conn = db.bootstrap_database(db_path)

    first = index_problem_bank(conn, problems_root)
    first_record = db.fetch_problem_record(conn, "two-sum")
    assert first.scanned == 1
    assert first.inserted == 1
    assert first.updated == 0
    assert first.skipped == 0
    assert first_record is not None

    second = index_problem_bank(conn, problems_root)
    assert second.scanned == 1
    assert second.inserted == 0
    assert second.updated == 0
    assert second.skipped == 1

    previous_tests_hash = first_record["tests_hash"]
    (statement_path.parent / "tests.json").write_text(
        json.dumps(
            [
                {
                    "input": {"nums": [3, 2, 4], "target": 6},
                    "expected": [1, 2],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    third = index_problem_bank(conn, problems_root)
    third_record = db.fetch_problem_record(conn, "two-sum")
    assert third.scanned == 1
    assert third.inserted == 0
    assert third.updated == 1
    assert third.skipped == 0
    assert third_record is not None
    assert third_record["tests_hash"] != previous_tests_hash
