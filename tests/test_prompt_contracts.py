from __future__ import annotations

from pathlib import Path

from lcgrade.chat import build_chat_prompt, default_chat_system_prompt, hint_system_prompt
from lcgrade.problems import parse_problem_directory
from lcgrade.reviews import build_stage2_prompt, generate_beta_stage2_review
from lcgrade.llm import MockBackend


def _two_sum_problem():
    project_root = Path(__file__).resolve().parents[1]
    return parse_problem_directory(project_root / "problems" / "two-sum")


def _reference_code() -> str:
    project_root = Path(__file__).resolve().parents[1]
    return (project_root / "problems" / "two-sum" / "solutions" / "reference.py").read_text(encoding="utf-8")


def test_stage2_prompt_is_structured_and_concise() -> None:
    problem = _two_sum_problem()
    prompt = build_stage2_prompt(
        problem,
        {
            "status": "pass",
            "test_mode": "both",
            "bundled_passed": 2,
            "bundled_total": 2,
            "llm_passed": 1,
            "llm_total": 1,
            "runtime_ms": 12.5,
            "code_snapshot": _reference_code(),
        },
    )

    assert prompt.startswith("You are lcgrade's Stage 2 reviewer for interview preparation.")
    assert "Write a concise Markdown review with exactly these sections:" in prompt
    assert "Keep each section short, specific, and non-repetitive." in prompt
    assert "Prefer concrete findings and fixes over praise." in prompt
    for heading in ("## Complexity", "## Correctness", "## Code Quality", "## Edge Cases", "## Verdict"):
        assert heading in prompt
    assert "Facts:" in prompt
    assert "Bundled tests: 2/2" in prompt
    assert "LLM tests: 1/1" in prompt
    assert "Submitted code:" in prompt


def test_stage2_review_generation_uses_concise_system_prompt() -> None:
    problem = _two_sum_problem()
    backend = MockBackend(
        responses=[
            "## Complexity\nO(n)\n\n## Correctness\nLooks good.\n\n## Code Quality\nClear.\n\n## Edge Cases\nHandled.\n\n## Verdict\nPass.",
        ]
    )

    result = generate_beta_stage2_review(
        problem,
        {
            "status": "pass",
            "test_mode": "both",
            "bundled_passed": 2,
            "bundled_total": 2,
            "llm_passed": 1,
            "llm_total": 1,
            "runtime_ms": 1.0,
            "code_snapshot": _reference_code(),
        },
        backend,
    )

    assert result.generated is True
    assert backend.prompts[0]["system_prompt"] is not None
    assert "strict but supportive coding interview reviewer" in str(backend.prompts[0]["system_prompt"])
    assert "Stay concise" in str(backend.prompts[0]["system_prompt"])
    assert "non-repetitive" in backend.prompts[0]["prompt"]


def test_chat_prompt_and_system_prompt_feel_like_an_interview_coach() -> None:
    problem = _two_sum_problem()
    prompt = build_chat_prompt(
        problem=problem,
        user_message="How should I think about this?",
        latest_attempt={
            "status": "pass",
            "test_mode": "both",
            "bundled_passed": 2,
            "bundled_total": 2,
            "llm_passed": 1,
            "llm_total": 1,
            "code_snapshot": _reference_code(),
        },
        latest_review_text="Solid pass.",
        recent_messages=[
            {"role": "user", "message": "I am stuck.", "hint_tier": None},
            {"role": "assistant", "message": "Try a hash map.", "hint_tier": None},
        ],
    )

    assert prompt.startswith("You are lcgrade's interview coach.")
    assert "Keep the reply problem-scoped, concise, and action-oriented." in prompt
    assert "Prefer hints, invariants, and next steps over full solutions." in prompt
    assert "Problem statement:" in prompt
    assert "Latest attempt summary:" in prompt
    assert "Latest review:" in prompt
    assert "Recent conversation:" in prompt
    assert "User request:" in prompt
    assert default_chat_system_prompt().startswith("You are lcgrade's problem-scoped interview coach.")
    assert "avoid giving away a full solution" in default_chat_system_prompt()


def test_hint_system_prompt_varies_by_tier() -> None:
    tier_1 = hint_system_prompt(1)
    tier_2 = hint_system_prompt(2)
    tier_3 = hint_system_prompt(3)

    assert "interview coach for hints" in tier_1
    assert "light nudge" in tier_1
    assert "medium-strength hint" in tier_2
    assert "strong hint" in tier_3
    assert "concise" in tier_1
    assert "concise" in tier_2
    assert "concise" in tier_3
    assert tier_1 != tier_2 != tier_3
