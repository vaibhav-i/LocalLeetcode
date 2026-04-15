from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

from .config import AppConfig, AppPaths, detect_total_ram_gb, load_app_config, resolve_model_name, save_app_config
from .db import bootstrap_database
from .llm import OllamaBackend
from .problems import index_problem_bank, problem_documents_from_root


@dataclass(slots=True, frozen=True)
class SetupStatus:
    data_dir_exists: bool
    db_ready: bool
    db_error: str | None
    indexed_count: int
    indexing_error: str | None
    ollama_binary_path: str | None
    ollama_reachable: bool
    installed_models: tuple[str, ...]
    configured_backend: str
    configured_model: str
    resolved_model: str
    model_available: bool
    ollama_detail: str | None
    ram_gb: float


def detect_ollama_binary() -> str | None:
    return shutil.which("ollama")


def inspect_setup(paths: AppPaths, *, check_only: bool) -> SetupStatus:
    config = load_app_config(paths.config_path)
    ram_gb = detect_total_ram_gb()
    resolved_model = resolve_model_name(config, ram_gb=ram_gb)
    backend = OllamaBackend(model=resolved_model)

    db_ready = False
    db_error: str | None = None
    indexed_count = 0
    indexing_error: str | None = None

    if check_only:
        try:
            indexed_count = len(problem_documents_from_root(paths.problems_dir))
        except Exception as exc:
            indexing_error = str(exc)
        db_ready = paths.db_path.exists()
        if not db_ready:
            db_error = "Database not initialized yet."
    else:
        try:
            paths.data_dir.mkdir(parents=True, exist_ok=True)
            connection = bootstrap_database(paths.db_path)
            try:
                report = index_problem_bank(connection, paths.problems_dir)
                indexed_count = report.scanned
                db_ready = True
            finally:
                connection.close()
        except Exception as exc:
            db_error = str(exc)

    ollama_binary_path = detect_ollama_binary()
    installed_models: tuple[str, ...] = ()
    ollama_reachable = False
    model_available = False
    ollama_detail: str | None = None

    if ollama_binary_path is not None:
        try:
            installed_models = backend.installed_models()
            ollama_reachable = True
            model_available = backend.model_installed()
            if not model_available:
                ollama_detail = backend.unavailable_reason()
        except Exception as exc:
            ollama_detail = str(exc)
    else:
        ollama_detail = "Ollama binary not found."

    return SetupStatus(
        data_dir_exists=paths.data_dir.exists(),
        db_ready=db_ready,
        db_error=db_error,
        indexed_count=indexed_count,
        indexing_error=indexing_error,
        ollama_binary_path=ollama_binary_path,
        ollama_reachable=ollama_reachable,
        installed_models=installed_models,
        configured_backend=config.backend,
        configured_model=config.model,
        resolved_model=resolved_model,
        model_available=model_available,
        ollama_detail=ollama_detail,
        ram_gb=ram_gb,
    )


def pull_ollama_model(model: str) -> tuple[bool, str | None]:
    try:
        completed = subprocess.run(
            ["ollama", "pull", model],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return False, str(exc)

    if completed.returncode == 0:
        return True, None

    error_text = (completed.stderr or completed.stdout).strip()
    if not error_text:
        error_text = f"`ollama pull {model}` failed with exit code {completed.returncode}"
    return False, error_text


def persist_setup_config(paths: AppPaths, *, backend: str, model: str) -> AppConfig:
    config = AppConfig(backend=backend, model=model)
    save_app_config(paths.config_path, config)
    return config
