"""Stage 2/3 review helpers for lcgrade.

This module keeps the review pipeline small and testable:
it loads the latest saved attempt, builds a Stage 2 prompt from the
problem statement plus saved execution facts, generates a qualitative
review through ``LLMBackend``, and optionally runs the built-in
extensions against the generated review.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from .db import get_metadata_value, mark_problem_review_generated
from .extensions import ExtensionResult, ExtensionContext, get_extension
from .llm import LLMBackend
from .problems import ProblemDocument, load_problem_by_slug
from .types import TestVerdict

DEFAULT_STAGE3_EXTENSIONS: tuple[str, ...] = ("interview", "optimize")


@dataclass(slots=True, frozen=True)
class Stage2ReviewResult:
    """Outcome of generating a beta Stage 2 review."""

    generated: bool
    review_text: str | None
    reason: str | None = None


@dataclass(slots=True, frozen=True)
class ReviewPipelineResult:
    """Complete output of the review pipeline slice."""

    slug: str
    attempt_id: int | None
    review_id: int | None
    review_text: str | None
    extension_results: tuple[ExtensionResult, ...]
    skipped_reason: str | None = None


def fetch_latest_attempt(conn: sqlite3.Connection, slug: str) -> dict[str, Any] | None:
    """Return the most recent saved attempt for ``slug``."""

    row = conn.execute(
        """
        SELECT *
        FROM attempts
        WHERE slug = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (slug,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def _attempt_value(attempt: Mapping[str, Any], key: str, default: Any = "") -> Any:
    value = attempt.get(key, default)
    if value is None:
        return default
    return value


def build_stage2_prompt(problem: ProblemDocument, attempt: Mapping[str, Any]) -> str:
    """Build the Stage 2 prompt from the problem and saved attempt."""

    code_snapshot = str(_attempt_value(attempt, "code_snapshot", ""))
    prompt_lines = [
        "You are lcgrade's Stage 2 reviewer for interview preparation.",
        "Write a concise Markdown review with these sections:",
        "## Complexity",
        "## Correctness",
        "## Code Quality",
        "## Edge Cases",
        "## Verdict",
        "",
        f"Problem: {problem.metadata.title} ({problem.slug})",
        f"Difficulty: {problem.metadata.difficulty}",
        f"Function: {problem.metadata.function_name}",
        f"Attempt status: {_attempt_value(attempt, 'status', 'unknown')}",
        f"Test mode: {_attempt_value(attempt, 'test_mode', 'unknown')}",
        (
            "Bundled tests: "
            f"{int(_attempt_value(attempt, 'bundled_passed', 0))}/"
            f"{int(_attempt_value(attempt, 'bundled_total', 0))}"
        ),
        (
            "LLM tests: "
            f"{int(_attempt_value(attempt, 'llm_passed', 0))}/"
            f"{int(_attempt_value(attempt, 'llm_total', 0))}"
        ),
        f"Runtime ms: {float(_attempt_value(attempt, 'runtime_ms', 0.0)):.3f}",
        "",
        "Problem statement:",
        problem.body.strip(),
        "",
        "Submitted code:",
        code_snapshot.strip(),
    ]
    return "\n".join(prompt_lines).strip() + "\n"


def _default_stage2_system_prompt() -> str:
    return "You are a strict but supportive coding interview reviewer."


def generate_beta_stage2_review(
    problem: ProblemDocument,
    attempt: Mapping[str, Any],
    llm: LLMBackend | None,
) -> Stage2ReviewResult:
    """Generate the Stage 2 review, falling back cleanly when unavailable."""

    if llm is None:
        return Stage2ReviewResult(generated=False, review_text=None, reason="No LLM backend provided.")

    try:
        if not llm.available():
            return Stage2ReviewResult(
                generated=False,
                review_text=None,
                reason="LLM backend unavailable.",
            )
    except Exception as exc:
        return Stage2ReviewResult(
            generated=False,
            review_text=None,
            reason=f"LLM availability check failed: {exc}",
        )

    prompt = build_stage2_prompt(problem, attempt)
    try:
        response = llm.generate(
            prompt=prompt,
            system_prompt=_default_stage2_system_prompt(),
            temperature=0.25,
            max_tokens=1200,
        )
    except Exception as exc:
        return Stage2ReviewResult(
            generated=False,
            review_text=None,
            reason=f"LLM review failed: {exc}",
        )

    text = response.text.strip()
    if not text:
        return Stage2ReviewResult(
            generated=False,
            review_text=None,
            reason="LLM returned an empty review.",
        )
    return Stage2ReviewResult(generated=True, review_text=text, reason=None)


def persist_review(
    conn: sqlite3.Connection,
    *,
    attempt_id: int,
    review_text: str,
    complexity_time: str = "",
    complexity_space: str = "",
) -> int:
    """Insert or update the Stage 2 review linked to ``attempt_id``."""

    timestamp = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO reviews (
                attempt_id,
                timestamp,
                complexity_time,
                complexity_space,
                review_text
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(attempt_id) DO UPDATE SET
                timestamp = excluded.timestamp,
                complexity_time = excluded.complexity_time,
                complexity_space = excluded.complexity_space,
                review_text = excluded.review_text
            """,
            (attempt_id, timestamp, complexity_time, complexity_space, review_text),
        )
    row = conn.execute("SELECT id FROM reviews WHERE attempt_id = ?", (attempt_id,)).fetchone()
    if row is None:  # pragma: no cover - defensive only
        raise RuntimeError(f"Failed to persist review for attempt {attempt_id}")
    return int(row["id"])


def persist_extension_result(
    conn: sqlite3.Connection,
    *,
    review_id: int,
    extension_name: str,
    output_text: str,
) -> int:
    """Persist one Stage 3 extension result."""

    timestamp = datetime.now(timezone.utc).isoformat()
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO extension_results (
                review_id,
                extension_name,
                output_text,
                timestamp
            )
            VALUES (?, ?, ?, ?)
            """,
            (review_id, extension_name, output_text, timestamp),
        )
    return int(cursor.lastrowid)


def persist_extension_results(
    conn: sqlite3.Connection,
    *,
    review_id: int,
    results: Iterable[ExtensionResult],
) -> list[int]:
    """Replace all extension results for a review with the provided set."""

    inserted_ids: list[int] = []
    with conn:
        conn.execute("DELETE FROM extension_results WHERE review_id = ?", (review_id,))
    for result in results:
        inserted_ids.append(
            persist_extension_result(
                conn,
                review_id=review_id,
                extension_name=result.name,
                output_text=result.content,
            )
        )
    return inserted_ids


def build_extension_verdicts(attempt: Mapping[str, Any]) -> list[TestVerdict]:
    """Build a typed Stage 1 verdict summary from saved attempt facts."""

    verdicts: list[TestVerdict] = []
    bundled_total = int(_attempt_value(attempt, "bundled_total", 0))
    bundled_passed = int(_attempt_value(attempt, "bundled_passed", 0))
    if bundled_total > 0:
        verdicts.append(
            TestVerdict(
                name="bundled-tests",
                passed=bundled_passed == bundled_total,
                input={"scope": "bundled", "test_mode": str(_attempt_value(attempt, "test_mode", "unknown"))},
                expected={"passed": bundled_total, "total": bundled_total},
                actual={"passed": bundled_passed, "total": bundled_total},
                source="verified",
            )
        )

    llm_total = int(_attempt_value(attempt, "llm_total", 0))
    llm_passed = int(_attempt_value(attempt, "llm_passed", 0))
    if llm_total > 0:
        verdicts.append(
            TestVerdict(
                name="llm-tests",
                passed=llm_passed == llm_total,
                input={"scope": "llm", "test_mode": str(_attempt_value(attempt, "test_mode", "unknown"))},
                expected={"passed": llm_total, "total": llm_total},
                actual={"passed": llm_passed, "total": llm_total},
                source="llm_verified",
            )
        )

    status = str(_attempt_value(attempt, "status", "unknown"))
    verdicts.append(
        TestVerdict(
            name="attempt-status",
            passed=status == "pass",
            input={"scope": "attempt"},
            expected="pass",
            actual=status,
            source="verified",
        )
    )
    return verdicts


def run_registered_extensions(
    problem: ProblemDocument,
    attempt: Mapping[str, Any],
    review_text: str,
    llm: LLMBackend,
    extension_names: Sequence[str] = DEFAULT_STAGE3_EXTENSIONS,
) -> list[ExtensionResult]:
    """Run the requested Stage 3 extensions from the registry."""

    context = ExtensionContext(
        problem_statement=problem.body,
        user_code=str(_attempt_value(attempt, "code_snapshot", "")),
        verdicts=build_extension_verdicts(attempt),
        review_output=review_text,
        timing_data=None,
    )

    results: list[ExtensionResult] = []
    for extension_name in extension_names:
        try:
            extension = get_extension(extension_name)
        except KeyError as exc:
            results.append(
                ExtensionResult(
                    name=extension_name,
                    title=extension_name,
                    content=f"Unknown extension {extension_name!r}: {exc}",
                )
            )
            continue

        try:
            results.append(extension.run(context, llm))
        except Exception as exc:
            results.append(
                ExtensionResult(
                    name=extension.name(),
                    title=extension.name(),
                    content=f"Extension {extension_name!r} failed: {exc}",
                )
            )
    return results


def review_problem(
    conn: sqlite3.Connection,
    slug: str,
    *,
    llm: LLMBackend | None = None,
    extension_names: Sequence[str] = DEFAULT_STAGE3_EXTENSIONS,
) -> ReviewPipelineResult:
    """Run the review slice against the latest saved attempt for a slug."""

    problem = load_problem_by_slug(conn, slug)
    if problem is None:
        raise ValueError(f"Unknown problem slug: {slug}")

    attempt = fetch_latest_attempt(conn, slug)
    if attempt is None:
        return ReviewPipelineResult(
            slug=slug,
            attempt_id=None,
            review_id=None,
            review_text=None,
            extension_results=(),
            skipped_reason="No saved attempt found.",
        )

    stage2 = generate_beta_stage2_review(problem, attempt, llm)
    if not stage2.generated or stage2.review_text is None:
        return ReviewPipelineResult(
            slug=slug,
            attempt_id=int(attempt["id"]),
            review_id=None,
            review_text=None,
            extension_results=(),
            skipped_reason=stage2.reason,
        )

    review_id = persist_review(
        conn,
        attempt_id=int(attempt["id"]),
        review_text=stage2.review_text,
    )
    mark_problem_review_generated(conn, slug)
    extension_results: list[ExtensionResult] = []
    if llm is not None:
        try:
            if llm.available():
                extension_results = run_registered_extensions(
                    problem,
                    attempt,
                    stage2.review_text,
                    llm,
                    extension_names=extension_names,
                )
                persist_extension_results(conn, review_id=review_id, results=extension_results)
        except Exception:
            extension_results = []

    return ReviewPipelineResult(
        slug=slug,
        attempt_id=int(attempt["id"]),
        review_id=review_id,
        review_text=stage2.review_text,
        extension_results=tuple(extension_results),
        skipped_reason=None,
    )


def problem_review_status(conn: sqlite3.Connection, slug: str) -> dict[str, Any]:
    """Small convenience helper for command-layer callers."""

    problem = load_problem_by_slug(conn, slug)
    attempt = fetch_latest_attempt(conn, slug)
    review_row = None
    if attempt is not None:
        review_row = conn.execute(
            "SELECT * FROM reviews WHERE attempt_id = ?",
            (attempt["id"],),
        ).fetchone()
    return {
        "problem": problem,
        "attempt": attempt,
        "review": dict(review_row) if review_row is not None else None,
        "active_slug": get_metadata_value(conn, "active_slug"),
    }
