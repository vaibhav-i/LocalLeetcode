from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


APP_NAME = "lcgrade"


@dataclass(frozen=True)
class AppPaths:
    project_root: Path
    package_root: Path
    workspace_root: Path
    problems_dir: Path
    data_dir: Path
    db_path: Path


def discover_paths() -> AppPaths:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent.parent
    workspace_root = project_root
    data_dir = workspace_root / ".lcgrade"
    problems_dir = workspace_root / "problems"
    db_path = data_dir / "lcgrade.db"
    return AppPaths(
        project_root=project_root,
        package_root=package_root,
        workspace_root=workspace_root,
        problems_dir=problems_dir,
        data_dir=data_dir,
        db_path=db_path,
    )
