from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess


APP_NAME = "lcgrade"
DEFAULT_BACKEND = "ollama"
DEFAULT_MODEL = "qwen2.5-coder:7b"


@dataclass(frozen=True)
class AppPaths:
    project_root: Path
    package_root: Path
    workspace_root: Path
    problems_dir: Path
    data_dir: Path
    db_path: Path
    config_path: Path


@dataclass(frozen=True)
class AppConfig:
    backend: str = DEFAULT_BACKEND
    model: str = "auto"


def discover_paths() -> AppPaths:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent.parent
    workspace_root = project_root
    data_dir = workspace_root / ".lcgrade"
    problems_dir = workspace_root / "problems"
    db_path = data_dir / "lcgrade.db"
    config_path = data_dir / "config.yaml"
    return AppPaths(
        project_root=project_root,
        package_root=package_root,
        workspace_root=workspace_root,
        problems_dir=problems_dir,
        data_dir=data_dir,
        db_path=db_path,
        config_path=config_path,
    )


def load_app_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        return AppConfig()

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        values[key.strip()] = raw_value.strip().strip("'\"")

    backend = values.get("backend", DEFAULT_BACKEND) or DEFAULT_BACKEND
    model = values.get("model", "auto") or "auto"
    return AppConfig(backend=backend, model=model)


def save_app_config(path: str | Path, config: AppConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"backend: {config.backend}",
                f"model: {config.model}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def detect_total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except (AttributeError, OSError, ValueError):
        pass

    try:
        output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(output) / (1024 ** 3)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return 16.0


def get_default_model(ram_gb: float | None = None) -> str:
    del ram_gb
    return DEFAULT_MODEL


def resolve_model_name(config: AppConfig, *, ram_gb: float | None = None) -> str:
    if config.model != "auto":
        return config.model
    return get_default_model(ram_gb=ram_gb)
