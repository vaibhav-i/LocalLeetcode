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


@dataclass(frozen=True)
class OllamaModelOption:
    name: str
    min_ram_gb: float
    recommended: bool
    summary: str


OLLAMA_MODEL_CATALOG = (
    OllamaModelOption(
        name="qwen2.5-coder:7b",
        min_ram_gb=12.0,
        recommended=True,
        summary="Best v0 default for coding on a 16 GB machine.",
    ),
    OllamaModelOption(
        name="llama3.2:3b",
        min_ram_gb=6.0,
        recommended=False,
        summary="Lighter fallback when memory is tighter or faster responses matter.",
    ),
    OllamaModelOption(
        name="deepseek-r1:14b",
        min_ram_gb=24.0,
        recommended=False,
        summary="Heavier reasoning model for larger-memory systems; not the v0 default.",
    ),
)


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
    if ram_gb is not None and ram_gb < 12.0:
        return "llama3.2:3b"
    return DEFAULT_MODEL


def recommended_ollama_models(ram_gb: float) -> tuple[OllamaModelOption, ...]:
    options: list[OllamaModelOption] = []
    for option in OLLAMA_MODEL_CATALOG:
        if ram_gb >= option.min_ram_gb:
            options.append(option)

    if not options:
        return (OLLAMA_MODEL_CATALOG[1],)

    options.sort(
        key=lambda option: (
            0 if option.name == get_default_model(ram_gb=ram_gb) else 1,
            0 if option.recommended else 1,
            option.min_ram_gb,
        )
    )
    return tuple(options)


def resolve_model_name(config: AppConfig, *, ram_gb: float | None = None) -> str:
    if config.model != "auto":
        return config.model
    return get_default_model(ram_gb=ram_gb)
