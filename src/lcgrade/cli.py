from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import discover_paths
from .solver import build_pending_attempt

app = typer.Typer(help="Local-first CLI auto-grader for LeetCode-style problems.")
console = Console()


def _lazy_imports():
    from .db import bootstrap_database
    from .problems import index_problem_bank, load_problem_by_slug

    return bootstrap_database, index_problem_bank, load_problem_by_slug


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
) -> None:
    paths = discover_paths()
    bootstrap_database, index_problem_bank, load_problem = _lazy_imports()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        problem = load_problem(connection, slug)
    finally:
        connection.close()
    if problem is None:
        raise typer.BadParameter(f"Unknown problem slug: {slug}")

    result = build_pending_attempt(problem, tests)
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Problem: {result.problem.metadata.title} ({result.problem.slug})",
                    f"Function: {result.problem.metadata.function_name}",
                    f"Tests mode: {result.attempt.test_mode}",
                    "Execution pipeline scaffolding is in place.",
                ]
            ),
            title="lcgrade solve",
        )
    )


@app.command()
def review(
    slug: str = typer.Argument(..., help="Problem slug to review."),
) -> None:
    paths = discover_paths()
    bootstrap_database, index_problem_bank, load_problem = _lazy_imports()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        problem = load_problem(connection, slug)
    finally:
        connection.close()
    if problem is None:
        raise typer.BadParameter(f"Unknown problem slug: {slug}")

    console.print(
        Panel.fit(
            f"Review pipeline placeholder ready for {problem.metadata.title}.",
            title="lcgrade review",
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
