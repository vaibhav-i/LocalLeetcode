from __future__ import annotations

from dataclasses import dataclass

from .problems import ProblemDocument
from .types import AttemptSummary


@dataclass(slots=True)
class SolveResult:
    problem: ProblemDocument
    attempt: AttemptSummary
    used_cache: bool
    review_summary: str | None = None


def build_pending_attempt(problem: ProblemDocument, test_mode: str) -> SolveResult:
    attempt = AttemptSummary(
        slug=problem.slug,
        test_mode=test_mode,
        bundled_passed=0,
        bundled_total=0,
        llm_passed=0,
        llm_total=0,
        status="pending",
        runtime_ms=0.0,
    )
    return SolveResult(problem=problem, attempt=attempt, used_cache=False)
