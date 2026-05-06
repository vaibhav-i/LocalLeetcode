from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import lcgrade.cli as cli_module
from lcgrade.cli import app
from lcgrade.config import AppConfig, AppPaths, load_app_config, save_app_config
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
        config_path=data_dir / "config.yaml",
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
        model = "llama3.2"
        base_url = "http://localhost:11434"

        def available(self) -> bool:
            return False

        def installed_models(self) -> tuple[str, ...]:
            return ()

        def unavailable_reason(self) -> str:
            return "Ollama is not reachable."

    monkeypatch.setattr("lcgrade.setup_wizard.OllamaBackend", lambda model=None: UnavailableBackend())

    result = runner.invoke(app, ["setup", "--check"])

    assert result.exit_code == 0
    assert str(isolated_app_paths.workspace_root) in result.stdout
    assert ".lcgrade/lcgrade.db" in result.stdout
    assert "Core features: not ready" in result.stdout
    assert "Problem bank readable: yes" in result.stdout
    assert "Resolved model: qwen2.5-coder:7b" in result.stdout
    assert "Ollama reachable: yes" in result.stdout
    assert "Installed Ollama models: none detected" in result.stdout
    assert "LLM features: No LLM configured" in result.stdout
    assert "ollama pull qwen2.5-coder:7b" in result.stdout
    assert "repo-local problems/" in result.stdout


def test_setup_check_does_not_create_data_dir(isolated_app_paths: AppPaths) -> None:
    result = runner.invoke(app, ["setup", "--check"])

    assert result.exit_code == 0
    assert isolated_app_paths.data_dir.exists() is False


def test_setup_reports_missing_ollama_binary(
    monkeypatch: pytest.MonkeyPatch,
    isolated_app_paths: AppPaths,
) -> None:
    monkeypatch.setattr("lcgrade.setup_wizard.detect_ollama_binary", lambda: None)

    result = runner.invoke(app, ["setup"])

    assert result.exit_code == 0
    assert "Core lcgrade is ready." in result.stdout
    assert "LLM features are unavailable until a local backend is configured." in result.stdout
    assert "brew install ollama" in result.stdout
    assert "python3 -m lcgrade.cli solve" in result.stdout


def test_setup_reports_ollama_serve_guidance(
    monkeypatch: pytest.MonkeyPatch,
    isolated_app_paths: AppPaths,
) -> None:
    monkeypatch.setattr("lcgrade.setup_wizard.detect_ollama_binary", lambda: "/usr/local/bin/ollama")

    class OfflineBackend:
        def __init__(self, model: str | None = None):
            self.model = model or "qwen2.5-coder:7b"
            self.base_url = "http://localhost:11434"

        def installed_models(self) -> tuple[str, ...]:
            raise RuntimeError("daemon down")

    monkeypatch.setattr("lcgrade.setup_wizard.OllamaBackend", OfflineBackend)

    result = runner.invoke(app, ["setup"])

    assert result.exit_code == 1
    assert "ollama serve" in result.stdout
    assert "python3 -m lcgrade.cli setup" in result.stdout


def test_setup_pulls_missing_model_and_persists_config(
    monkeypatch: pytest.MonkeyPatch,
    isolated_app_paths: AppPaths,
) -> None:
    state = {"installed": False}
    monkeypatch.setattr("lcgrade.setup_wizard.detect_total_ram_gb", lambda: 16.0)
    monkeypatch.setattr("lcgrade.setup_wizard.detect_ollama_binary", lambda: "/usr/local/bin/ollama")

    class FakeBackend:
        def __init__(self, model: str | None = None):
            self.model = model or "qwen2.5-coder:7b"
            self.base_url = "http://localhost:11434"

        def installed_models(self) -> tuple[str, ...]:
            return (self.model,) if state["installed"] else ()

        def model_installed(self) -> bool:
            return state["installed"]

        def unavailable_reason(self) -> str:
            return f"Ollama is running, but model {self.model!r} is not installed."

    monkeypatch.setattr("lcgrade.setup_wizard.OllamaBackend", FakeBackend)
    monkeypatch.setattr(cli_module, "OllamaBackend", FakeBackend)
    monkeypatch.setattr(cli_module.typer, "prompt", lambda message, default="", show_default=True: "1")
    monkeypatch.setattr(cli_module.typer, "confirm", lambda message, default=True: True)

    def fake_pull(model: str) -> tuple[bool, str | None]:
        state["installed"] = True
        return True, None

    monkeypatch.setattr(cli_module, "pull_ollama_model", fake_pull)

    result = runner.invoke(app, ["setup"])

    assert result.exit_code == 0
    assert "Setup complete." in result.stdout
    assert "qwen2.5-coder:7b" in result.stdout
    config = load_app_config(isolated_app_paths.config_path)
    assert config == AppConfig(backend="ollama", model="qwen2.5-coder:7b")


def test_setup_can_select_already_installed_model_without_pull(
    monkeypatch: pytest.MonkeyPatch,
    isolated_app_paths: AppPaths,
) -> None:
    monkeypatch.setattr("lcgrade.setup_wizard.detect_total_ram_gb", lambda: 16.0)
    monkeypatch.setattr("lcgrade.setup_wizard.detect_ollama_binary", lambda: "/usr/local/bin/ollama")

    class FakeBackend:
        def __init__(self, model: str | None = None):
            self.model = model or "qwen2.5-coder:7b"
            self.base_url = "http://localhost:11434"

        def installed_models(self) -> tuple[str, ...]:
            return ("llama3.2:3b",)

        def model_installed(self) -> bool:
            return self.model == "llama3.2:3b"

        def unavailable_reason(self) -> str:
            return f"Ollama is running, but model {self.model!r} is not installed."

    monkeypatch.setattr("lcgrade.setup_wizard.OllamaBackend", FakeBackend)
    monkeypatch.setattr(cli_module, "OllamaBackend", FakeBackend)
    monkeypatch.setattr(cli_module.typer, "prompt", lambda message, default="", show_default=True: "1")

    pulled: list[str] = []

    def fake_pull(model: str) -> tuple[bool, str | None]:
        pulled.append(model)
        return True, None

    monkeypatch.setattr(cli_module, "pull_ollama_model", fake_pull)

    result = runner.invoke(app, ["setup"])

    assert result.exit_code == 0
    assert "llama3.2:3b" in result.stdout
    assert pulled == []
    config = load_app_config(isolated_app_paths.config_path)
    assert config == AppConfig(backend="ollama", model="llama3.2:3b")


def test_setup_check_prints_remediation_commands(
    monkeypatch: pytest.MonkeyPatch,
    isolated_app_paths: AppPaths,
) -> None:
    class UnavailableBackend:
        model = "qwen2.5-coder:7b"
        base_url = "http://localhost:11434"

        def installed_models(self) -> tuple[str, ...]:
            return ()

        def model_installed(self) -> bool:
            return False

        def unavailable_reason(self) -> str:
            return "Ollama is not reachable."

    monkeypatch.setattr("lcgrade.setup_wizard.OllamaBackend", lambda model=None: UnavailableBackend())
    monkeypatch.setattr("lcgrade.setup_wizard.detect_ollama_binary", lambda: None)

    result = runner.invoke(app, ["setup", "--check"])

    assert result.exit_code == 0
    assert "brew install ollama" in result.stdout
    assert "ollama serve" in result.stdout
    assert "ollama pull qwen2.5-coder:7b" in result.stdout
    assert "Core lcgrade solving works without any LLM backend." in result.stdout
    assert "python3 -m lcgrade.cli start two-sum" in result.stdout
    assert "python3 -m lcgrade.cli solve" in result.stdout


def test_setup_uses_configured_model_for_solve(
    monkeypatch: pytest.MonkeyPatch,
    isolated_app_paths: AppPaths,
) -> None:
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    cli_module.set_active_slug(conn, "two-sum")
    conn.close()
    save_app_config(isolated_app_paths.config_path, AppConfig(backend="ollama", model="custom-model"))

    captured: dict[str, str] = {}

    class FakeBackend:
        def __init__(self, model: str | None = None):
            captured["model"] = model or ""

        def available(self) -> bool:
            return False

        def unavailable_reason(self) -> str:
            return "LLM backend unavailable."

    monkeypatch.setattr(cli_module, "OllamaBackend", FakeBackend)

    result = runner.invoke(app, ["solve", "--tests", "llm", "--solution", "problems/two-sum/solutions/reference.py"])

    assert result.exit_code == 0
    assert captured["model"] == "custom-model"


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


def test_describe_explicit_slug_renders_problem_statement() -> None:
    result = runner.invoke(app, ["describe", "two-sum"])

    assert result.exit_code == 0
    assert "Title: Two Sum" in result.stdout
    assert "Slug: two-sum" in result.stdout
    assert "Function: two_sum" in result.stdout
    assert "Given an array of integers" in result.stdout


def test_describe_uses_active_problem_when_slug_is_omitted(isolated_app_paths: AppPaths) -> None:
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    cli_module.set_active_slug(conn, "contains-duplicate")
    conn.close()

    result = runner.invoke(app, ["describe"])

    assert result.exit_code == 0
    assert "Title: Contains Duplicate" in result.stdout
    assert "Slug: contains-duplicate" in result.stdout


def test_history_renders_empty_state_for_problem_without_attempts(
    isolated_app_paths: AppPaths,
) -> None:
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    cli_module.set_active_slug(conn, "two-sum")
    conn.close()

    result = runner.invoke(app, ["history"])

    assert result.exit_code == 0
    assert "No saved attempts yet." in result.stdout
    assert "Milestones: none yet." in result.stdout


def test_history_renders_attempts_and_milestones(
    isolated_app_paths: AppPaths,
) -> None:
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    cli_module.set_active_slug(conn, "two-sum")
    _seed_attempt(conn, slug="two-sum", code_snapshot="def two_sum(nums, target):\n    return [0, 1]\n", status="fail")
    second_attempt = _seed_attempt(conn, slug="two-sum", code_snapshot="def two_sum(nums, target):\n    return [1, 2]\n", status="pass")
    review_id = persist_review(conn, attempt_id=second_attempt, review_text="Solid review.", complexity_time="O(n)")
    persist_extension_result(conn, review_id=review_id, extension_name="interview", output_text="Follow-up")
    conn.execute(
        """
        UPDATE problems
        SET
            auto_solved = 1,
            auto_solved_at = '2026-01-01T12:00:00+00:00',
            review_generated = 1,
            review_generated_at = '2026-01-01T12:05:00+00:00',
            review_acknowledged = 1,
            review_acknowledged_at = '2026-01-01T12:10:00+00:00',
            followup_completed = 1,
            followup_completed_at = '2026-01-01T12:15:00+00:00'
        WHERE slug = 'two-sum'
        """
    )
    conn.commit()
    conn.close()

    result = runner.invoke(app, ["history"])

    assert result.exit_code == 0
    assert "History · Two Sum" in result.stdout
    assert "O(n)" in result.stdout
    assert "✓ ack" in result.stdout
    assert "✓" in result.stdout
    assert "Solved on" in result.stdout
    assert "Review acknowledged on" in result.stdout
    assert "Follow-up completed on" in result.stdout


def test_stats_renders_global_progress_and_weakest_tags(
    isolated_app_paths: AppPaths,
) -> None:
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    conn.execute(
        """
        UPDATE problems
        SET
            auto_solved = CASE slug WHEN 'two-sum' THEN 1 ELSE 0 END,
            review_generated = CASE slug WHEN 'two-sum' THEN 1 ELSE 0 END,
            review_acknowledged = CASE slug WHEN 'two-sum' THEN 1 ELSE 0 END,
            followup_completed = CASE slug WHEN 'two-sum' THEN 1 ELSE 0 END
        """
    )
    _seed_attempt(conn, slug="two-sum", code_snapshot="def two_sum(nums, target):\n    return [0, 1]\n", status="pass")
    _seed_attempt(conn, slug="contains-duplicate", code_snapshot="def contains_duplicate(nums):\n    return False\n", status="fail")
    conn.execute(
        """
        UPDATE attempts
        SET bundled_passed = CASE slug WHEN 'two-sum' THEN 2 ELSE 0 END,
            bundled_total = 2
        """
    )
    conn.commit()
    conn.close()

    result = runner.invoke(app, ["stats"])

    assert result.exit_code == 0
    assert "Solved: 1/3" in result.stdout
    assert "Attempted: 2/3" in result.stdout
    assert "Review Generated: 1" in result.stdout
    assert "By Difficulty:" in result.stdout
    assert "easy: 1/3 solved" in result.stdout
    assert "Weakest Tags:" in result.stdout
    assert "hash-table" in result.stdout or "two-pointers" in result.stdout or "array" in result.stdout


def test_random_sets_active_problem(
    isolated_app_paths: AppPaths,
) -> None:
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    conn.execute("UPDATE problems SET auto_solved = 1 WHERE slug IN ('contains-duplicate', 'two-sum-sorted')")
    conn.commit()
    conn.close()

    result = runner.invoke(app, ["random"])

    assert result.exit_code == 0
    assert "Active problem:" in result.stdout

    conn = open_database(isolated_app_paths.db_path)
    try:
        assert get_active_slug(conn) is not None
    finally:
        conn.close()


def test_random_respects_filters_and_reports_empty_state(
    isolated_app_paths: AppPaths,
) -> None:
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    conn.execute("UPDATE problems SET auto_solved = 1 WHERE slug = 'two-sum'")
    conn.execute("UPDATE problems SET auto_solved = 1 WHERE slug = 'two-sum-sorted'")
    conn.commit()
    conn.close()

    filtered = runner.invoke(app, ["random", "--tag", "hash-table"])
    empty = runner.invoke(app, ["random", "--difficulty", "hard"])

    assert filtered.exit_code == 0
    assert "contains-duplicate" in filtered.stdout or "two-sum" in filtered.stdout
    assert empty.exit_code == 1
    assert "No unsolved problem matched the current filters." in empty.stdout


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
    monkeypatch.setattr(cli_module, "_build_backend", lambda paths, name: MockBackend(default_response="unused"))

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
    monkeypatch.setattr(cli_module, "_build_backend", lambda paths, name: backend)

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

    monkeypatch.setattr(cli_module, "_build_backend", lambda paths, name: UnavailableBackend())
    conn = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(conn, isolated_app_paths.problems_dir)
    cli_module.set_active_slug(conn, "two-sum")
    conn.close()

    result = runner.invoke(app, ["chat", "Need help"])

    assert result.exit_code == 1
    assert "This feature needs a local LLM backend." in result.stdout
    assert "Core lcgrade solving still works without one." in result.stdout
    assert "brew install ollama" in result.stdout


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
    monkeypatch.setattr(cli_module, "_build_backend", lambda paths, name: backend)
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
    monkeypatch.setattr(cli_module, "_build_backend", lambda paths, name: backend)
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
