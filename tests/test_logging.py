from __future__ import annotations

import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

import lcgrade.cli as cli_module
from lcgrade.cli import app
from lcgrade.config import AppConfig, AppPaths, save_app_config
from lcgrade.db import bootstrap_database, set_active_slug
from lcgrade.logging_utils import LOG_FILE_NAME, configure_logging
from lcgrade.problems import index_problem_bank


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


def test_configure_logging_levels_and_debug_file(isolated_app_paths: AppPaths) -> None:
    default_state = configure_logging(isolated_app_paths, verbose=False, debug=False)
    assert default_state.level == logging.WARNING
    assert default_state.log_file_path is None

    verbose_state = configure_logging(isolated_app_paths, verbose=True, debug=False)
    assert verbose_state.level == logging.INFO
    assert verbose_state.log_file_path is None

    debug_state = configure_logging(isolated_app_paths, verbose=False, debug=True)
    assert debug_state.level == logging.DEBUG
    assert debug_state.log_file_path == isolated_app_paths.data_dir / LOG_FILE_NAME
    assert debug_state.log_file_path.exists()
    root_logger = logging.getLogger("lcgrade")
    assert any(getattr(handler, "stream", None) is not None for handler in root_logger.handlers)


def test_debug_setup_logs_to_stderr_and_file(
    monkeypatch: pytest.MonkeyPatch,
    isolated_app_paths: AppPaths,
) -> None:
    class FakeBackend:
        def __init__(self, model: str | None = None):
            self.model = model or "qwen2.5-coder:7b"
            self.base_url = "http://localhost:11434"

        def installed_models(self) -> tuple[str, ...]:
            return (self.model,)

        def model_installed(self) -> bool:
            return True

        def unavailable_reason(self) -> str:
            return ""

    monkeypatch.setattr("lcgrade.setup_wizard.detect_ollama_binary", lambda: "/usr/local/bin/ollama")
    monkeypatch.setattr("lcgrade.setup_wizard.OllamaBackend", FakeBackend)

    result = runner.invoke(app, ["--debug", "setup", "--check"])

    assert result.exit_code == 0
    assert "lcgrade setup --check" in result.stdout
    log_file = isolated_app_paths.data_dir / LOG_FILE_NAME
    assert log_file.exists()
    log_text = log_file.read_text(encoding="utf-8")
    assert "Running setup" in result.output
    assert "Inspecting setup state" in log_text


def test_debug_solve_logs_fallback_reason(
    monkeypatch: pytest.MonkeyPatch,
    isolated_app_paths: AppPaths,
) -> None:
    connection = bootstrap_database(isolated_app_paths.db_path)
    index_problem_bank(connection, isolated_app_paths.problems_dir)
    set_active_slug(connection, "two-sum")
    connection.close()
    save_app_config(isolated_app_paths.config_path, AppConfig(backend="ollama", model="missing-model"))

    class FakeBackend:
        def __init__(self, model: str | None = None):
            self.model = model or "missing-model"

        def available(self) -> bool:
            return False

        def unavailable_reason(self) -> str:
            return "Configured model missing in Ollama."

    monkeypatch.setattr(cli_module, "OllamaBackend", FakeBackend)

    result = runner.invoke(
        app,
        [
            "--debug",
            "solve",
            "--tests",
            "llm",
            "--solution",
            "problems/two-sum/solutions/reference.py",
        ],
    )

    assert result.exit_code == 0
    assert "Effective tests mode: bundled" in result.stdout
    assert "Configured model missing in Ollama." in result.output
    assert "LLM test generation unavailable" in result.output
