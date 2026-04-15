from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import discover_paths
from .execution import ExecutionError
from .llm import MLXBackend, OllamaBackend
from .preflight import PreflightError
from .reviews import DEFAULT_STAGE3_EXTENSIONS, review_problem
from .solve_flow import SolveFlowError, solve_problem

app = typer.Typer(help="Local-first CLI auto-grader for LeetCode-style problems.")
console = Console()


def _lazy_imports():
    from .db import bootstrap_database
    from .problems import index_problem_bank, load_problem_by_slug

    return bootstrap_database, index_problem_bank, load_problem_by_slug


def _build_backend(name: str):
    normalized = name.strip().lower()
    if normalized == "ollama":
        return OllamaBackend()
    if normalized == "mlx":
        return MLXBackend()
    raise typer.BadParameter(f"Unsupported backend: {name}")


@app.command()
def init() -> None:
    paths = discover_paths()
    bootstrap_database, index_problem_bank, _ = _lazy_imports()
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
    bootstrap_database, index_problem_bank, _ = _lazy_imports()
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
def solve(
    slug: str = typer.Argument(..., help="Problem slug to evaluate."),
    tests: str = typer.Option("both", "--tests", help="bundled, llm, or both"),
    solution: Path | None = typer.Option(None, "--solution", help="Path to the solution file to evaluate."),
    force: bool = typer.Option(False, "--force", help="Bypass cached attempts and re-run Stage 1."),
    backend: str = typer.Option("ollama", "--backend", help="LLM backend to use for test generation."),
) -> None:
    paths = discover_paths()
    bootstrap_database, index_problem_bank, load_problem = _lazy_imports()

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        problem = load_problem(connection, slug)
        if problem is None:
            raise typer.BadParameter(f"Unknown problem slug: {slug}")

        try:
            flow = solve_problem(
                connection,
                problem,
                solution_path=solution,
                requested_test_mode=tests,
                force=force,
                llm=_build_backend(backend) if tests in {"llm", "both"} else None,
            )
        except (ExecutionError, PreflightError, SolveFlowError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
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
    slug: str = typer.Argument(..., help="Problem slug to review."),
    backend: str = typer.Option("ollama", "--backend", help="LLM backend to use for Stage 2/3."),
    extend: str = typer.Option(
        ",".join(DEFAULT_STAGE3_EXTENSIONS),
        "--extend",
        help="Comma-separated Stage 3 extensions to run.",
    ),
) -> None:
    paths = discover_paths()
    bootstrap_database, index_problem_bank, _ = _lazy_imports()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
