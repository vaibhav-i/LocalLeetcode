from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import lcgrade.cli as cli_module
from lcgrade.cli import app
from lcgrade.config import AppPaths


runner = CliRunner()


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


def test_list_problems_smoke() -> None:
    result = runner.invoke(app, ["list-problems"])
    assert result.exit_code == 0
    assert "Contains" in result.stdout
    assert "Duplicate" in result.stdout
    assert "two-sum" in result.stdout
    assert "two-sum-sorted" in result.stdout


def test_solve_smoke() -> None:
    result = runner.invoke(app, ["solve", "two-sum"])
    assert result.exit_code == 0
    assert "two-sum" in result.stdout
    assert "Requested tests mode: both" in result.stdout
    assert "Effective tests mode: bundled" in result.stdout
    assert "Backend: ollama" in result.stdout
    assert "LLM tests: 0/0" in result.stdout
    assert "Bundled tests: 2/2" in result.stdout
    assert "Status: pass" in result.stdout
    assert "Using bundled tests only." in result.stdout
    assert "Active problem cleared after successful solve." in result.stdout
    assert "lcgrade review two-sum" in result.stdout


def test_solve_uses_cache_for_unchanged_solution() -> None:
    first = runner.invoke(
        app,
        ["solve", "two-sum", "--solution", "problems/two-sum/solutions/reference.py"],
    )
    second = runner.invoke(
        app,
        ["solve", "two-sum", "--solution", "problems/two-sum/solutions/reference.py"],
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "Saved a new attempt snapshot." in first.stdout
    assert "Using cached results." in second.stdout


def test_review_without_attempt_reports_missing_saved_attempt() -> None:
    result = runner.invoke(app, ["review", "two-sum"])

    assert result.exit_code == 1
    assert "No saved attempt found." in result.stdout or "No saved attempt found" in result.stdout


def test_solve_accepts_explicit_mlx_backend_and_falls_back_for_llm_tests() -> None:
    result = runner.invoke(
        app,
        ["solve", "two-sum", "--tests", "llm", "--backend", "mlx", "--solution", "problems/two-sum/solutions/reference.py"],
    )

    assert result.exit_code == 0
    assert "Backend: mlx" in result.stdout
    assert "Using bundled tests only." in result.stdout


def test_review_accepts_explicit_mlx_backend() -> None:
    solve_result = runner.invoke(
        app,
        ["solve", "two-sum", "--solution", "problems/two-sum/solutions/reference.py"],
    )
    review_result = runner.invoke(app, ["review", "two-sum", "--backend", "mlx"])

    assert solve_result.exit_code == 0
    assert review_result.exit_code == 0
    assert "LLM backend unavailable." in review_result.stdout


def test_solve_reports_missing_expected_function(tmp_path: Path) -> None:
    broken = tmp_path / "broken_solution.py"
    broken.write_text("def not_two_sum(nums, target):\n    return []\n", encoding="utf-8")

    result = runner.invoke(app, ["solve", "two-sum", "--solution", str(broken)])

    assert result.exit_code == 1
