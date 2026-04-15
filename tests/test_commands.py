from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import lcgrade.cli as cli_module
from lcgrade.cli import app
from lcgrade.config import AppPaths
from lcgrade.db import (
    bootstrap_database,
    fetch_problem_record,
    get_active_slug,
    insert_attempt,
    insert_chat_message,
    open_database,
)
from lcgrade.execution import hash_text
from lcgrade.llm import LLMBackend, LLMResponse, MockBackend, ModelInfo
from lcgrade.problems import index_problem_bank
from lcgrade.reviews import persist_extension_result, persist_review


runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_app_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppPaths:
    project_root = Path(__file__).resolve().parents[1]
    package_root = project_root / "src" / "lcgrade"
    data_dir = tmp_path / ".lcgrade"
    paths = AppPaths(
        project_root=project_root,
        package_root=package_root,
        workspace_root=project_root,
        problems_dir=project_root / "problems",
        data_dir=data_dir,
        db_path=data_dir / "lcgrade.db",
    )
    monkeypatch.setattr(cli_module, "discover_paths", lambda: paths)
    return paths


def _seed_attempt(conn, *, slug: str, code_snapshot: str, status: str = "pass") -> int:
    return insert_attempt(
        conn,
        slug=slug,
        test_mode="both",
        code_hash=hash_text(code_snapshot),
        tests_hash="tests-hash",
        bundled_passed=2,
        bundled_total=2,
        llm_passed=1,
        llm_total=1,
        runtime_ms=1.0,
        status=status,
        code_snapshot=code_snapshot,
    )


def test_setup_reports_paths_and_unavailable_backend(
    monkeypatch: pytest.MonkeyPatch,
    isolated_app_paths: AppPaths,
) -> None:
    class UnavailableBackend:
        def available(self) -> bool:
            return False

    monkeypatch.setattr(cli_module, "OllamaBackend", lambda: UnavailableBackend())

    result = runner.invoke(app, ["setup"])

    assert result.exit_code == 0
    assert str(isolated_app_paths.workspace_root) in result.stdout
    assert ".lcgrade/lcgrade.db" in result.stdout
    assert "Problem bank indexable: yes" in result.stdout
    assert "Ollama reachable: no" in result.stdout
    assert "repo-local problems/" in result.stdout


def test_start_sets_active_slug(isolated_app_paths: AppPaths) -> None:
    result = runner.invoke(app, ["start", "two-sum"])

    assert result.exit_code == 0
    assert "Active problem: Two Sum (two-sum)" in result.stdout
    assert "starter.py" in result.stdout

    conn = open_database(isolated_app_paths.db_path)
    try:
        assert get_active_slug(conn) == "two-sum"
    finally:
        conn.close()


def test_start_rejects_unknown_slug() -> None:
    result = runner.invoke(app, ["start", "unknown-problem"])

    assert result.exit_code != 0
    assert "Unknown problem slug" in (result.stdout + result.stderr)


def test_reset_clears_problem_state_and_active_slug(isolated_app_paths: AppPaths) -> None:
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    conn.execute("UPDATE problems SET auto_solved = 1, auto_solved_at = 'now', review_generated = 1, review_generated_at = 'now' WHERE slug = 'two-sum'")
    cli_module.set_active_slug(conn, "two-sum")
    attempt_id = _seed_attempt(conn, slug="two-sum", code_snapshot="def two_sum(nums, target):\n    return [0, 1]\n")
    review_id = persist_review(conn, attempt_id=attempt_id, review_text="Solid work.")
    persist_extension_result(conn, review_id=review_id, extension_name="interview", output_text="Follow-up")
    insert_chat_message(conn, slug="two-sum", session_id="default", role="user", message="help")
    conn.close()

    result = runner.invoke(app, ["reset", "two-sum"])

    assert result.exit_code == 0
    assert "Attempts deleted: 1" in result.stdout
    assert "Chat messages deleted: 1" in result.stdout
    assert "Cleared active problem: yes" in result.stdout

    conn = open_database(isolated_app_paths.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) AS count FROM attempts WHERE slug = 'two-sum'").fetchone()["count"] == 0
        assert conn.execute("SELECT COUNT(*) AS count FROM reviews").fetchone()["count"] == 0
        assert conn.execute("SELECT COUNT(*) AS count FROM extension_results").fetchone()["count"] == 0
        assert conn.execute("SELECT COUNT(*) AS count FROM chat_messages WHERE slug = 'two-sum'").fetchone()["count"] == 0
        record = fetch_problem_record(conn, "two-sum")
        assert record is not None
        assert int(record["auto_solved"]) == 0
        assert record["auto_solved_at"] is None
        assert int(record["review_generated"]) == 0
        assert record["review_generated_at"] is None
        assert get_active_slug(conn) is None
    finally:
        conn.close()


def test_reset_uses_active_problem_when_slug_is_omitted(isolated_app_paths: AppPaths) -> None:
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    cli_module.set_active_slug(conn, "two-sum")
    _seed_attempt(conn, slug="two-sum", code_snapshot="def two_sum(nums, target):\n    return [0, 1]\n")
    conn.close()

    result = runner.invoke(app, ["reset"])

    assert result.exit_code == 0
    assert "Reset local state for two-sum." in result.stdout


def test_prune_keeps_newest_ten_attempts_and_cascades_reviews(
    isolated_app_paths: AppPaths,
) -> None:
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    review_ids: list[int] = []
    for index in range(12):
        attempt_id = _seed_attempt(
            conn,
            slug="two-sum",
            code_snapshot=f"def two_sum(nums, target):\n    return [{index}]\n",
        )
        review_id = persist_review(conn, attempt_id=attempt_id, review_text=f"review {index}")
        review_ids.append(review_id)
        persist_extension_result(conn, review_id=review_id, extension_name="interview", output_text=f"ext {index}")
    insert_chat_message(conn, slug="two-sum", session_id="default", role="user", message="keep me")
    conn.close()

    result = runner.invoke(app, ["prune"])

    assert result.exit_code == 0
    assert "Deleted attempts: 2" in result.stdout

    conn = open_database(isolated_app_paths.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) AS count FROM attempts WHERE slug = 'two-sum'").fetchone()["count"] == 10
        assert conn.execute("SELECT COUNT(*) AS count FROM reviews").fetchone()["count"] == 10
        assert conn.execute("SELECT COUNT(*) AS count FROM extension_results").fetchone()["count"] == 10
        assert conn.execute("SELECT COUNT(*) AS count FROM chat_messages WHERE slug = 'two-sum'").fetchone()["count"] == 1
    finally:
        conn.close()


def test_chat_requires_active_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "_build_backend", lambda name: MockBackend(default_response="unused"))

    result = runner.invoke(app, ["chat", "How should I think about this?"])

    assert result.exit_code == 1
    assert "Run `lcgrade start <slug>` first" in result.stdout


def test_solve_requires_active_problem_when_slug_is_omitted() -> None:
    result = runner.invoke(app, ["solve"])

    assert result.exit_code == 1
    assert "Run `lcgrade start <slug>` first" in result.stdout


def test_review_requires_active_problem_when_slug_is_omitted() -> None:
    result = runner.invoke(app, ["review"])

    assert result.exit_code == 1
    assert "Run `lcgrade start <slug>` first" in result.stdout


def test_chat_persists_messages_and_includes_attempt_review_context(
    monkeypatch: pytest.MonkeyPatch,
    isolated_app_paths: AppPaths,
) -> None:
    backend = MockBackend(default_response="Focus on using a hash map.")
    monkeypatch.setattr(cli_module, "_build_backend", lambda name: backend)

    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    cli_module.set_active_slug(conn, "two-sum")
    attempt_id = _seed_attempt(conn, slug="two-sum", code_snapshot="def two_sum(nums, target):\n    return [0, 1]\n")
    persist_review(conn, attempt_id=attempt_id, review_text="Your complexity is good.")
    conn.close()

    result = runner.invoke(app, ["chat", "What should I improve?"])

    assert result.exit_code == 0
    assert "Focus on using a hash map." in result.stdout
    assert "Your complexity is good." in backend.prompts[0]["prompt"]
    assert "Code snapshot:" in backend.prompts[0]["prompt"]

    conn = open_database(isolated_app_paths.db_path)
    try:
        rows = conn.execute(
            "SELECT role, message, hint_tier FROM chat_messages WHERE slug = 'two-sum' ORDER BY id"
        ).fetchall()
        assert [row["role"] for row in rows] == ["user", "assistant"]
        assert rows[0]["message"] == "What should I improve?"
        assert rows[1]["message"] == "Focus on using a hash map."
        assert rows[0]["hint_tier"] is None
    finally:
        conn.close()


def test_chat_reports_unavailable_backend(
    monkeypatch: pytest.MonkeyPatch,
    isolated_app_paths: AppPaths,
) -> None:
    class UnavailableBackend(LLMBackend):
        def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.7, max_tokens: int = 2048) -> LLMResponse:
            raise AssertionError("generate should not run")

        def available(self) -> bool:
            return False

        def model_info(self) -> ModelInfo:
            return ModelInfo(name="unavailable", context_window=0, quantization="unknown", backend="mock")

    monkeypatch.setattr(cli_module, "_build_backend", lambda name: UnavailableBackend())
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    cli_module.set_active_slug(conn, "two-sum")
    conn.close()

    result = runner.invoke(app, ["chat", "Need help"])

    assert result.exit_code == 1
    assert "LLM backend unavailable." in result.stdout


def test_solve_uses_active_problem_and_clears_active_slug_and_chat_on_success(
    isolated_app_paths: AppPaths,
) -> None:
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    cli_module.set_active_slug(conn, "two-sum")
    insert_chat_message(conn, slug="two-sum", session_id="default", role="user", message="keep?")
    conn.close()

    result = runner.invoke(app, ["solve", "--solution", "problems/two-sum/solutions/reference.py"])

    assert result.exit_code == 0
    assert "Problem: Two Sum (two-sum)" in result.stdout
    assert "Bundled tests: 2/2" in result.stdout

    conn = open_database(isolated_app_paths.db_path)
    try:
        assert get_active_slug(conn) is None
        assert conn.execute("SELECT COUNT(*) AS count FROM chat_messages WHERE slug = 'two-sum'").fetchone()["count"] == 0
    finally:
        conn.close()


def test_review_uses_active_problem_when_slug_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    isolated_app_paths: AppPaths,
) -> None:
    backend = MockBackend(
        responses=[
            "## Complexity\nO(n)\n\n## Correctness\nLooks good.\n\n## Code Quality\nClear.\n\n## Edge Cases\nHandled.\n\n## Verdict\nPass.",
            "Interview follow-up.",
            "Optimization note.",
        ]
    )
    monkeypatch.setattr(cli_module, "_build_backend", lambda name: backend)
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    cli_module.set_active_slug(conn, "two-sum")
    _seed_attempt(conn, slug="two-sum", code_snapshot="def two_sum(nums, target):\n    return [0, 1]\n")
    conn.close()

    result = runner.invoke(app, ["review"])

    assert result.exit_code == 0
    assert "## Interview Follow-ups" in result.stdout
    assert "Interview follow-up." in result.stdout


def test_hint_enforces_tier_and_persists_hint_tier(
    monkeypatch: pytest.MonkeyPatch,
    isolated_app_paths: AppPaths,
) -> None:
    backend = MockBackend(default_response="Try storing seen values in a set.")
    monkeypatch.setattr(cli_module, "_build_backend", lambda name: backend)
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    cli_module.set_active_slug(conn, "contains-duplicate")
    conn.close()

    bad = runner.invoke(app, ["hint", "4"])
    good = runner.invoke(app, ["hint", "2", "Nudge me without solving it"])

    assert bad.exit_code == 1
    assert "Hint tier must be one of: 1, 2, 3." in bad.stdout
    assert good.exit_code == 0
    assert "Try storing seen values in a set." in good.stdout
    assert "medium-strength hint" in backend.prompts[0]["system_prompt"]

    conn = open_database(isolated_app_paths.db_path)
    try:
        rows = conn.execute(
            "SELECT role, hint_tier FROM chat_messages WHERE slug = 'contains-duplicate' ORDER BY id"
        ).fetchall()
        assert [row["role"] for row in rows] == ["user", "assistant"]
        assert [row["hint_tier"] for row in rows] == [2, 2]
    finally:
        conn.close()
