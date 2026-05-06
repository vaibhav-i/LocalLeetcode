from __future__ import annotations

from pathlib import Path
import tomllib


def test_pyproject_exposes_lcgrade_console_script() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["lcgrade"] == "lcgrade.cli:app"
    assert pyproject["project"]["requires-python"] == ">=3.11"
