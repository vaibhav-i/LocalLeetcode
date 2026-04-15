from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Sequence

from .db import fetch_recent_chat_messages, get_active_slug, insert_chat_message
from .llm import LLMBackend
from .problems import ProblemDocument, load_problem_by_slug
from .reviews import fetch_latest_attempt, problem_review_status

DEFAULT_CHAT_SESSION_ID = "default"
VALID_HINT_TIERS = (1, 2, 3)


@dataclass(slots=True, frozen=True)
class ChatTurnResult:
    slug: str | None
    response_text: str | None
    session_id: str
    skipped_reason: str | None = None


def _history_lines(messages: Sequence[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        hint_tier = message.get("hint_tier")
        tier_suffix = f" (hint tier {hint_tier})" if hint_tier is not None else ""
        lines.append(f"{role}{tier_suffix}: {str(message.get('message', '')).strip()}")
    return lines


def _attempt_summary_block(attempt: dict[str, object] | None) -> str:
    if attempt is None:
        return "No saved attempt yet."
    return "\n".join(
        [
            f"Status: {attempt['status']}",
            f"Test mode: {attempt['test_mode']}",
            f"Bundled tests: {attempt['bundled_passed']}/{attempt['bundled_total']}",
            f"LLM tests: {attempt['llm_passed']}/{attempt['llm_total']}",
            "Code snapshot:",
            str(attempt["code_snapshot"]).strip(),
        ]
    )


def build_chat_prompt(
    *,
    problem: ProblemDocument,
    user_message: str,
    latest_attempt: dict[str, object] | None,
    latest_review_text: str | None,
    recent_messages: Sequence[dict[str, object]],
) -> str:
    history = _history_lines(recent_messages)
    review_text = latest_review_text.strip() if latest_review_text else "No review generated yet."
    prompt_lines = [
        f"Problem: {problem.metadata.title} ({problem.slug})",
        f"Difficulty: {problem.metadata.difficulty}",
        f"Function: {problem.metadata.function_name}",
        "",
        "Problem statement:",
        problem.body.strip(),
        "",
        "Latest attempt summary:",
        _attempt_summary_block(latest_attempt),
        "",
        "Latest review:",
        review_text,
        "",
        "Recent conversation:",
        *(history if history else ["No prior conversation."]),
        "",
        "User request:",
        user_message.strip(),
    ]
    return "\n".join(prompt_lines).strip() + "\n"


def default_chat_system_prompt() -> str:
    return (
        "You are lcgrade's problem-scoped coding coach. "
        "Be concise, actionable, and avoid giving away a full solution unless directly asked."
    )


def hint_system_prompt(tier: int) -> str:
    if tier == 1:
        style = "Give only a light nudge. Focus on framing, not steps."
    elif tier == 2:
        style = "Give a medium-strength hint with directional guidance, but not a full algorithm."
    else:
        style = "Give a strong hint that outlines the key approach, but do not write the final code."
    return (
        "You are lcgrade's hint assistant. "
        f"{style} "
        "Keep the tone supportive and concise."
    )


def _resolve_active_problem(conn: sqlite3.Connection) -> tuple[str | None, ProblemDocument | None]:
    slug = get_active_slug(conn)
    if slug is None:
        return None, None
    return slug, load_problem_by_slug(conn, slug)


def run_chat_turn(
    conn: sqlite3.Connection,
    *,
    message: str,
    llm: LLMBackend | None,
    session_id: str = DEFAULT_CHAT_SESSION_ID,
    system_prompt: str | None = None,
    hint_tier: int | None = None,
) -> ChatTurnResult:
    slug, problem = _resolve_active_problem(conn)
    if slug is None or problem is None:
        return ChatTurnResult(
            slug=None,
            response_text=None,
            session_id=session_id,
            skipped_reason="No active problem. Run `lcgrade start <slug>` first.",
        )

    if llm is None:
        return ChatTurnResult(
            slug=slug,
            response_text=None,
            session_id=session_id,
            skipped_reason="LLM backend unavailable.",
        )

    try:
        if not llm.available():
            reason = llm.unavailable_reason() or "LLM backend unavailable."
            return ChatTurnResult(
                slug=slug,
                response_text=None,
                session_id=session_id,
                skipped_reason=reason,
            )
    except Exception as exc:
        return ChatTurnResult(
            slug=slug,
            response_text=None,
            session_id=session_id,
            skipped_reason=f"LLM backend unavailable: {exc}",
        )

    status = problem_review_status(conn, slug)
    latest_attempt = fetch_latest_attempt(conn, slug)
    latest_review_text = None
    if status["review"] is not None:
        latest_review_text = str(status["review"]["review_text"])
    recent_messages = fetch_recent_chat_messages(conn, slug=slug, session_id=session_id)
    prompt = build_chat_prompt(
        problem=problem,
        user_message=message,
        latest_attempt=latest_attempt,
        latest_review_text=latest_review_text,
        recent_messages=recent_messages,
    )

    try:
        response = llm.generate(
            prompt=prompt,
            system_prompt=system_prompt or default_chat_system_prompt(),
            temperature=0.35,
            max_tokens=900,
        )
    except Exception as exc:
        return ChatTurnResult(
            slug=slug,
            response_text=None,
            session_id=session_id,
            skipped_reason=f"LLM chat failed: {exc}",
        )

    text = response.text.strip()
    if not text:
        return ChatTurnResult(
            slug=slug,
            response_text=None,
            session_id=session_id,
            skipped_reason="LLM returned an empty response.",
        )

    insert_chat_message(
        conn,
        slug=slug,
        session_id=session_id,
        role="user",
        message=message.strip(),
        hint_tier=hint_tier,
    )
    insert_chat_message(
        conn,
        slug=slug,
        session_id=session_id,
        role="assistant",
        message=text,
        hint_tier=hint_tier,
    )
    return ChatTurnResult(
        slug=slug,
        response_text=text,
        session_id=session_id,
        skipped_reason=None,
    )


def run_hint_turn(
    conn: sqlite3.Connection,
    *,
    tier: int,
    llm: LLMBackend | None,
    message: str | None = None,
    session_id: str = DEFAULT_CHAT_SESSION_ID,
) -> ChatTurnResult:
    if tier not in VALID_HINT_TIERS:
        return ChatTurnResult(
            slug=get_active_slug(conn),
            response_text=None,
            session_id=session_id,
            skipped_reason="Hint tier must be one of: 1, 2, 3.",
        )

    prompt_message = (
        message.strip()
        if message and message.strip()
        else f"Give me a tier {tier} hint for this problem and my latest solution state."
    )
    return run_chat_turn(
        conn,
        message=prompt_message,
        llm=llm,
        session_id=session_id,
        system_prompt=hint_system_prompt(tier),
        hint_tier=tier,
    )
