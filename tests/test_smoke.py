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
        ),
    )


def test_list_problems_smoke() -> None:
    result = runner.invoke(app, ["list-problems"])
    assert result.exit_code == 0
    assert "two-sum" in result.stdout


def test_solve_smoke() -> None:
    result = runner.invoke(app, ["solve", "two-sum"])
    assert result.exit_code == 0
    assert "two-sum" in result.stdout
    assert "Bundled tests: 0/2" in result.stdout


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


def test_solve_reports_missing_expected_function(tmp_path: Path) -> None:
    broken = tmp_path / "broken_solution.py"
    broken.write_text("def not_two_sum(nums, target):\n    return []\n", encoding="utf-8")

    result = runner.invoke(app, ["solve", "two-sum", "--solution", str(broken)])

    assert result.exit_code == 1
