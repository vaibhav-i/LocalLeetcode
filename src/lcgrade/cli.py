from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .chat import run_chat_turn, run_hint_turn
from .config import AppConfig, discover_paths, load_app_config
from .db import (
    bootstrap_database,
    clear_chat_messages,
    clear_active_slug,
    get_active_slug,
    prune_attempt_history,
    reset_problem_state,
    set_active_slug,
)
from .execution import ExecutionError
from .llm import MLXBackend, OllamaBackend
from .preflight import PreflightError
from .reviews import DEFAULT_STAGE3_EXTENSIONS, review_problem
from .setup_wizard import inspect_setup, persist_setup_config, pull_ollama_model
from .solve_flow import SolveFlowError, solve_problem

app = typer.Typer(help="Local-first CLI auto-grader for LeetCode-style problems.")
console = Console()


def _lazy_imports():
    from .problems import index_problem_bank, load_problem_by_slug

    return index_problem_bank, load_problem_by_slug


def _configured_app_config(paths) -> AppConfig:
    return load_app_config(paths.config_path)


def _resolve_backend_name(paths, requested_name: str | None) -> str:
    if requested_name is not None:
        return requested_name.strip().lower()
    return _configured_app_config(paths).backend.strip().lower()


def _build_backend(paths, name: str | None):
    normalized = _resolve_backend_name(paths, name)
    if normalized == "ollama":
        config = _configured_app_config(paths)
        model = config.model if config.model != "auto" else None
        return OllamaBackend(model=model)
    if normalized == "mlx":
        return MLXBackend()
    raise typer.BadParameter(f"Unsupported backend: {name}")


def _resolve_solution_path(paths, solution: Path | None) -> Path | None:
    if solution is None or solution.is_absolute():
        return solution
    return (paths.workspace_root / solution).resolve()


def _resolve_target_slug(
    connection,
    slug: str | None,
    *,
    require_active_message: str = "No active problem. Run `lcgrade start <slug>` first.",
) -> str:
    if slug is not None:
        return slug
    active_slug = get_active_slug(connection)
    if active_slug is None:
        console.print(Panel.fit(require_active_message, title="lcgrade"))
        raise typer.Exit(code=1)
    return active_slug


@app.command()
def init() -> None:
    paths = discover_paths()
    index_problem_bank, _ = _lazy_imports()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        report = index_problem_bank(connection, paths.problems_dir)
    finally:
        connection.close()
    console.print(
        Panel.fit(
            f"Indexed {report.scanned} problem(s)\nDB: {paths.db_path}",
            title="lcgrade init",
        )
    )


@app.command()
def list_problems() -> None:
    paths = discover_paths()
    index_problem_bank, _ = _lazy_imports()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        rows = connection.execute(
            """
            SELECT slug, title, difficulty, category, validator
            FROM problems
            ORDER BY difficulty, slug
            """
        ).fetchall()
    finally:
        connection.close()

    table = Table(title="Problems")
    table.add_column("Slug")
    table.add_column("Title")
    table.add_column("Difficulty")
    table.add_column("Category")
    table.add_column("Validator")
    for row in rows:
        table.add_row(*[str(value) for value in row])
    console.print(table)


@app.command()
def setup(
    check: bool = typer.Option(False, "--check", help="Report current environment without mutating setup state."),
) -> None:
    paths = discover_paths()
    status = inspect_setup(paths, check_only=check)

    if check:
        lines = [
            f"Workspace: {paths.workspace_root}",
            f"Data dir: {paths.data_dir}",
            f"DB path: {paths.db_path}",
            f"Config path: {paths.config_path}",
            f"Problem bank: {paths.problems_dir}",
            f"Data dir exists: {'yes' if status.data_dir_exists else 'no'}",
            f"Database ready: {'yes' if status.db_ready else 'no'}",
            *([f"Database detail: {status.db_error}"] if status.db_error else []),
            (
                f"Problem bank readable: yes ({status.indexed_count} problem(s))"
                if status.indexing_error is None
                else f"Problem bank readable: no ({status.indexing_error})"
            ),
            f"Configured backend: {status.configured_backend}",
            f"Configured model: {status.configured_model}",
            f"Resolved model: {status.resolved_model}",
            f"Detected RAM: {status.ram_gb:.1f} GB",
            f"Ollama binary: {status.ollama_binary_path or 'not found'}",
            f"Ollama reachable: {'yes' if status.ollama_reachable else 'no'}",
            (
                "Installed Ollama models: " + ", ".join(status.installed_models)
                if status.installed_models
                else "Installed Ollama models: none detected"
            ),
            *(["Ollama detail: " + status.ollama_detail] if status.ollama_detail else []),
            "v0 uses repo-local problems/ and .lcgrade/ paths.",
        ]
        console.print(Panel.fit("\n".join(lines), title="lcgrade setup --check"))
        return

    if status.ollama_binary_path is None:
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        "Ollama is not installed.",
                        "Install Ollama first, then re-run `lcgrade setup`.",
                        "Expected command after install: `ollama serve`",
                    ]
                ),
                title="lcgrade setup",
            )
        )
        raise typer.Exit(code=1)

    if not status.ollama_reachable:
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        "Ollama is installed but not reachable.",
                        "Run `ollama serve` in another terminal, then re-run `lcgrade setup`.",
                        *(["Detail: " + status.ollama_detail] if status.ollama_detail else []),
                    ]
                ),
                title="lcgrade setup",
            )
        )
        raise typer.Exit(code=1)

    if not status.model_available:
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"Recommended model: {status.resolved_model}",
                        f"Detected RAM: {status.ram_gb:.1f} GB",
                        "The configured model is not installed yet.",
                        *(["Detail: " + status.ollama_detail] if status.ollama_detail else []),
                    ]
                ),
                title="lcgrade setup",
            )
        )
        if not typer.confirm(f"Pull `{status.resolved_model}` now?", default=True):
            console.print(
                Panel.fit(
                    f"Setup incomplete. Run `ollama pull {status.resolved_model}` and then `lcgrade setup`.",
                    title="lcgrade setup",
                )
            )
            raise typer.Exit(code=1)

        console.print(f"Pulling {status.resolved_model} via Ollama...")
        pulled, pull_error = pull_ollama_model(status.resolved_model)
        if not pulled:
            console.print(
                Panel.fit(
                    "\n".join(
                        [
                            f"Failed to pull {status.resolved_model}.",
                            "Run this manually and retry setup:",
                            f"`ollama pull {status.resolved_model}`",
                            *(["Error: " + pull_error] if pull_error else []),
                        ]
                    ),
                    title="lcgrade setup",
                )
            )
            raise typer.Exit(code=1)

    config = persist_setup_config(paths, backend="ollama", model=status.resolved_model)
    final_status = inspect_setup(paths, check_only=False)
    if not final_status.db_ready or final_status.indexing_error is not None or not final_status.model_available:
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        "Setup incomplete.",
                        *(["Database detail: " + final_status.db_error] if final_status.db_error else []),
                        *(["Indexing detail: " + final_status.indexing_error] if final_status.indexing_error else []),
                        *(["Ollama detail: " + final_status.ollama_detail] if final_status.ollama_detail else []),
                    ]
                ),
                title="lcgrade setup",
            )
        )
        raise typer.Exit(code=1)

    lines = [
        f"Workspace: {paths.workspace_root}",
        f"Data dir: {paths.data_dir}",
        f"DB path: {paths.db_path}",
        f"Config path: {paths.config_path}",
        f"Problem bank: {paths.problems_dir}",
        f"Database ready: {'yes' if final_status.db_ready else 'no'}",
        f"Problem bank indexable: yes ({final_status.indexed_count} problem(s))",
        f"Configured backend: {config.backend}",
        f"Ollama model: {config.model}",
        f"Detected RAM: {final_status.ram_gb:.1f} GB",
        f"Ollama reachable: {'yes' if final_status.ollama_reachable else 'no'}",
        (
            "Installed Ollama models: " + ", ".join(final_status.installed_models)
            if final_status.installed_models
            else "Installed Ollama models: none detected"
        ),
        "Setup complete.",
        "Next steps:",
        "1. lcgrade start two-sum",
        "2. edit the starter file",
        "3. lcgrade solve",
        "v0 uses repo-local problems/ and .lcgrade/ paths.",
    ]
    console.print(Panel.fit("\n".join(lines), title="lcgrade setup"))


@app.command()
def start(slug: str = typer.Argument(..., help="Problem slug to make active.")) -> None:
    paths = discover_paths()
    index_problem_bank, load_problem = _lazy_imports()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        problem = load_problem(connection, slug)
        if problem is None:
            raise typer.BadParameter(f"Unknown problem slug: {slug}")
        set_active_slug(connection, slug)
    finally:
        connection.close()

    starter_path = problem.starter_path if problem.starter_path is not None else "No starter.py found."
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Active problem: {problem.metadata.title} ({problem.slug})",
                    f"Function: {problem.metadata.function_name}",
                    f"Starter: {starter_path}",
                ]
            ),
            title="lcgrade start",
        )
    )


@app.command()
def reset(slug: str | None = typer.Argument(None, help="Problem slug whose local state should be cleared.")) -> None:
    paths = discover_paths()
    index_problem_bank, load_problem = _lazy_imports()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        slug = _resolve_target_slug(connection, slug)
        problem = load_problem(connection, slug)
        if problem is None:
            raise typer.BadParameter(f"Unknown problem slug: {slug}")
        summary = reset_problem_state(connection, slug)
    finally:
        connection.close()

    lines = [
        f"Reset local state for {slug}.",
        f"Attempts deleted: {summary['attempts_deleted']}",
        f"Chat messages deleted: {summary['chat_messages_deleted']}",
        f"Milestones reset: {'yes' if summary['milestones_reset'] else 'no'}",
        f"Cleared active problem: {'yes' if summary['cleared_active_slug'] else 'no'}",
    ]
    console.print(Panel.fit("\n".join(lines), title="lcgrade reset"))


@app.command()
def prune() -> None:
    paths = discover_paths()
    index_problem_bank, _ = _lazy_imports()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        deleted = prune_attempt_history(connection, keep_per_problem=10)
    finally:
        connection.close()

    console.print(
        Panel.fit(
            "\n".join(
                [
                    "Pruned attempt history.",
                    "Retention policy: keep newest 10 attempts per problem.",
                    f"Deleted attempts: {deleted}",
                ]
            ),
            title="lcgrade prune",
        )
    )


@app.command()
def solve(
    slug: str | None = typer.Argument(None, help="Problem slug to evaluate."),
    tests: str = typer.Option("both", "--tests", help="bundled, llm, or both"),
    solution: Path | None = typer.Option(None, "--solution", help="Path to the solution file to evaluate."),
    force: bool = typer.Option(False, "--force", help="Bypass cached attempts and re-run Stage 1."),
    backend: str | None = typer.Option(None, "--backend", help="LLM backend to use for test generation."),
) -> None:
    paths = discover_paths()
    index_problem_bank, load_problem = _lazy_imports()

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        slug = _resolve_target_slug(connection, slug)
        problem = load_problem(connection, slug)
        if problem is None:
            raise typer.BadParameter(f"Unknown problem slug: {slug}")

        try:
            flow = solve_problem(
                connection,
                problem,
                solution_path=_resolve_solution_path(paths, solution),
                requested_test_mode=tests,
                force=force,
                llm=_build_backend(paths, backend) if tests in {"llm", "both"} else None,
            )
        except (ExecutionError, PreflightError, SolveFlowError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        if flow.attempt.bundled_total > 0 and flow.attempt.bundled_passed == flow.attempt.bundled_total:
            clear_chat_messages(connection, problem.slug)
            if get_active_slug(connection) == problem.slug:
                clear_active_slug(connection)
    finally:
        connection.close()

    llm_tests_message = flow.test_generation_result.warning if flow.test_generation_result is not None else None

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Problem: {flow.problem.metadata.title} ({flow.problem.slug})",
                    f"Function: {flow.problem.metadata.function_name}",
                    f"Requested tests mode: {flow.requested_test_mode}",
                    f"Effective tests mode: {flow.effective_test_mode}",
                    f"Backend: {_resolve_backend_name(paths, backend)}",
                    (
                        f"Code unchanged since last attempt. "
                        f"Showing cached results."
                        if flow.used_cache
                        else "New code or test inputs detected."
                    ),
                    *([llm_tests_message] if llm_tests_message else []),
                    f"Bundled tests: {flow.attempt.bundled_passed}/{flow.attempt.bundled_total}",
                    f"LLM tests: {flow.attempt.llm_passed}/{flow.attempt.llm_total}",
                    f"Status: {flow.attempt.status}",
                    f"Runtime: {flow.attempt.runtime_ms:.2f} ms",
                    "Using cached results." if flow.used_cache else "Saved a new attempt snapshot.",
                ]
            ),
            title="lcgrade solve",
        )
    )


@app.command()
def review(
    slug: str | None = typer.Argument(None, help="Problem slug to review."),
    backend: str | None = typer.Option(None, "--backend", help="LLM backend to use for Stage 2/3."),
    extend: str = typer.Option(
        ",".join(DEFAULT_STAGE3_EXTENSIONS),
        "--extend",
        help="Comma-separated Stage 3 extensions to run.",
    ),
) -> None:
    paths = discover_paths()
    index_problem_bank, _ = _lazy_imports()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        slug = _resolve_target_slug(connection, slug)
        extension_names = tuple(
            item.strip()
            for item in extend.split(",")
            if item.strip()
        )
        result = review_problem(
            connection,
            slug,
            llm=_build_backend(paths, backend),
            extension_names=extension_names,
        )
    finally:
        connection.close()

    if result.attempt_id is None:
        console.print(
            Panel.fit(
                result.skipped_reason or f"No attempt found for {slug}.",
                title="lcgrade review",
            )
        )
        raise typer.Exit(code=1)

    if result.review_text is None:
        console.print(
            Panel.fit(
                result.skipped_reason or "Review was skipped.",
                title="lcgrade review",
            )
        )
        return

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Backend: {_resolve_backend_name(paths, backend)}",
                    "",
                    result.review_text,
                    "",
                    *(
                        [
                            f"## {extension.title}\n{extension.content}"
                            for extension in result.extension_results
                        ]
                    ),
                ]
            ).strip(),
            title="lcgrade review",
        )
    )


@app.command()
def chat(
    message: str = typer.Argument(..., help="Message to send about the active problem."),
) -> None:
    paths = discover_paths()
    index_problem_bank, _ = _lazy_imports()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        result = run_chat_turn(
            connection,
            message=message,
            llm=_build_backend(paths, None),
        )
    finally:
        connection.close()

    if result.response_text is None:
        console.print(Panel.fit(result.skipped_reason or "Chat was skipped.", title="lcgrade chat"))
        raise typer.Exit(code=1)

    console.print(Panel.fit(result.response_text, title="lcgrade chat"))


@app.command()
def hint(
    tier: int = typer.Argument(..., help="Hint tier: 1, 2, or 3."),
    message: str | None = typer.Argument(None, help="Optional hint request refinement."),
) -> None:
    paths = discover_paths()
    index_problem_bank, _ = _lazy_imports()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        result = run_hint_turn(
            connection,
            tier=tier,
            message=message,
            llm=_build_backend(paths, None),
        )
    finally:
        connection.close()

    if result.response_text is None:
        console.print(Panel.fit(result.skipped_reason or "Hint was skipped.", title="lcgrade hint"))
        raise typer.Exit(code=1)

    console.print(Panel.fit(result.response_text, title="lcgrade hint"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
