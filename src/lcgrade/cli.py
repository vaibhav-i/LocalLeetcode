from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .chat import run_chat_turn, run_hint_turn
from .config import discover_paths
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
from .solve_flow import SolveFlowError, solve_problem

app = typer.Typer(help="Local-first CLI auto-grader for LeetCode-style problems.")
console = Console()


def _lazy_imports():
    from .problems import index_problem_bank, load_problem_by_slug

    return index_problem_bank, load_problem_by_slug


def _build_backend(name: str):
    normalized = name.strip().lower()
    if normalized == "ollama":
        return OllamaBackend()
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
def setup() -> None:
    paths = discover_paths()
    index_problem_bank, _ = _lazy_imports()
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    db_ready = False
    indexed_count = 0
    indexing_error: str | None = None
    connection = None
    try:
        connection = bootstrap_database(paths.db_path)
        db_ready = True
        try:
            report = index_problem_bank(connection, paths.problems_dir)
            indexed_count = report.scanned
        except Exception as exc:  # pragma: no cover - defensive status reporting
            indexing_error = str(exc)
    finally:
        if connection is not None:
            connection.close()

    ollama_available = False
    ollama_error: str | None = None
    backend = OllamaBackend()
    try:
        ollama_available = backend.available()
        if not ollama_available:
            ollama_error = "Ollama is not reachable."
    except Exception as exc:  # pragma: no cover - defensive status reporting
        ollama_error = str(exc)

    lines = [
        f"Workspace: {paths.workspace_root}",
        f"Data dir: {paths.data_dir}",
        f"DB path: {paths.db_path}",
        f"Problem bank: {paths.problems_dir}",
        f"Database ready: {'yes' if db_ready else 'no'}",
        (
            f"Problem bank indexable: yes ({indexed_count} problem(s))"
            if indexing_error is None
            else f"Problem bank indexable: no ({indexing_error})"
        ),
        f"Ollama reachable: {'yes' if ollama_available else 'no'}",
        *(["Ollama detail: " + ollama_error] if ollama_error else []),
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
    backend: str = typer.Option("ollama", "--backend", help="LLM backend to use for test generation."),
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
                llm=_build_backend(backend) if tests in {"llm", "both"} else None,
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
                    f"Backend: {backend}",
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
    backend: str = typer.Option("ollama", "--backend", help="LLM backend to use for Stage 2/3."),
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
            llm=_build_backend(backend),
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
                    f"Backend: {backend}",
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
            llm=_build_backend("ollama"),
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
            llm=_build_backend("ollama"),
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
