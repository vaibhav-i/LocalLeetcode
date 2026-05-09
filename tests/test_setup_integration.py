from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import lcgrade.cli as cli_module
from lcgrade.cli import app
from lcgrade.config import AppPaths
from lcgrade.llm import OllamaBackend


runner = CliRunner()
MODEL = "qwen2.5-coder:7b"


def ollama_ready() -> bool:
    try:
        return OllamaBackend(model=MODEL).available()
    except Exception:
        return False


@pytest.fixture(autouse=True)
def isolated_app_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    package_root = project_root / "src" / "lcgrade"
    data_dir = tmp_path / ".lcgrade"
    monkeypatch.setattr(
        cli_module,
        "discover_paths",
        lambda: AppPaths(
            project_root=project_root,
            package_root=package_root,
            workspace_root=project_root,
            problems_dir=project_root / "problems",
            data_dir=data_dir,
            db_path=data_dir / "lcgrade.db",
            config_path=data_dir / "config.yaml",
        ),
    )


@pytest.mark.skipif(not ollama_ready(), reason="Ollama with qwen2.5-coder:7b is not available")
def test_setup_completes_with_live_ollama() -> None:
    result = runner.invoke(app, ["setup"], input="1\n")

    assert result.exit_code == 0
    assert "Setup complete." in result.stdout
    assert MODEL in result.stdout


@pytest.mark.skipif(not ollama_ready(), reason="Ollama with qwen2.5-coder:7b is not available")
def test_post_setup_solve_uses_live_model() -> None:
    setup_result = runner.invoke(app, ["setup"], input="1\n")
    solve_result = runner.invoke(
        app,
        ["solve", "two-sum", "--tests", "llm", "--solution", "problems/two-sum/solutions/reference.py"],
    )

    assert setup_result.exit_code == 0
    assert solve_result.exit_code == 0
    assert "Backend: ollama" in solve_result.stdout
    assert "Using bundled tests only." not in solve_result.stdout
