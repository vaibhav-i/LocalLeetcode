from __future__ import annotations

from typer.testing import CliRunner

from lcgrade.cli import app


runner = CliRunner()


def test_list_problems_smoke() -> None:
    result = runner.invoke(app, ["list-problems"])
    assert result.exit_code == 0
    assert "two-sum" in result.stdout


def test_solve_smoke() -> None:
    result = runner.invoke(app, ["solve", "two-sum"])
    assert result.exit_code == 0
    assert "two-sum" in result.stdout
    assert "Bundled tests: 0/2" in result.stdout
