from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .chat import run_chat_turn, run_hint_turn
from .config import AppConfig, discover_paths, load_app_config, recommended_ollama_models
from .db import (
    bootstrap_database,
    clear_chat_messages,
    clear_active_slug,
    fetch_global_progress_counts,
    fetch_problem_history_view,
    fetch_progress_counts_by_difficulty,
    fetch_weakest_tags_by_bundled_pass_rate,
    get_active_slug,
    prune_attempt_history,
    reset_problem_state,
    select_random_unsolved_problem,
    set_active_slug,
)
from .execution import ExecutionError
from .llm import MLXBackend, OllamaBackend, local_llm_enablement_commands
from .logging_utils import configure_logging, get_logger
from .preflight import PreflightError
from .reviews import DEFAULT_STAGE3_EXTENSIONS, review_problem
from .setup_wizard import inspect_setup, persist_setup_config, pull_ollama_model
from .solve_flow import SolveFlowError, solve_problem

app = typer.Typer(help="Local-first CLI auto-grader for LeetCode-style problems.")
console = Console()
logger = get_logger("lcgrade.cli")


def _lazy_imports():
    from .problems import index_problem_bank, load_problem_by_slug

    return index_problem_bank, load_problem_by_slug


@app.callback()
def app_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show info-level logs on stderr."),
    debug: bool = typer.Option(False, "--debug", help="Show debug logs on stderr and write .lcgrade/lcgrade.log."),
) -> None:
    paths = discover_paths()
    log_state = configure_logging(paths, verbose=verbose, debug=debug)
    ctx.obj = {"paths": paths, "log_state": log_state}
    logger.info(
        "CLI startup",
        extra={
            "verbose": verbose,
            "debug": debug,
            "log_file_path": str(log_state.log_file_path) if log_state.log_file_path else None,
        },
    )


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


def _format_timestamp(value: str | None) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(value).strftime("%b %d, %I:%M%p")
    except ValueError:
        return value


def _history_review_state(item: dict[str, object], milestones: dict[str, object]) -> str:
    review = item.get("review")
    if review is None:
        return "·"
    if milestones.get("review_acknowledged"):
        return "✓ ack"
    return "✓ gen"


def _history_followup_state(item: dict[str, object], milestones: dict[str, object]) -> str:
    if milestones.get("followup_completed"):
        return "✓"
    extensions = item.get("extensions") or []
    return "✓ gen" if extensions else "·"


def _milestone_lines(milestones: dict[str, object]) -> list[str]:
    solved_at = milestones.get("auto_solved_at") or milestones.get("manual_solved_at")
    lines: list[str] = []
    if solved_at:
        lines.append(f"Solved on {_format_timestamp(str(solved_at))}")
    if milestones.get("review_generated_at"):
        lines.append(f"Review generated on {_format_timestamp(str(milestones['review_generated_at']))}")
    if milestones.get("review_acknowledged_at"):
        lines.append(f"Review acknowledged on {_format_timestamp(str(milestones['review_acknowledged_at']))}")
    if milestones.get("followup_completed_at"):
        lines.append(f"Follow-up completed on {_format_timestamp(str(milestones['followup_completed_at']))}")
    return lines


def _setup_model_choices(status) -> list[dict[str, str]]:
    choices: list[dict[str, str]] = []
    seen: set[str] = set()

    for model in status.installed_models:
        normalized = str(model).strip()
        if not normalized or normalized in seen:
            continue
        choices.append(
            {
                "name": normalized,
                "source": "installed",
                "summary": "Already available locally.",
            }
        )
        seen.add(normalized)

    for option in recommended_ollama_models(status.ram_gb):
        if option.name in seen:
            continue
        choices.append(
            {
                "name": option.name,
                "source": "recommended",
                "summary": option.summary,
            }
        )
        seen.add(option.name)

    if status.resolved_model not in seen:
        choices.insert(
            0,
            {
                "name": status.resolved_model,
                "source": "configured",
                "summary": "Current lcgrade-configured model.",
            },
        )
    return choices


def _select_setup_model(status) -> str:
    choices = _setup_model_choices(status)
    table = Table(title="Ollama Model Choices")
    table.add_column("#")
    table.add_column("Model")
    table.add_column("Source")
    table.add_column("Notes")

    default_index = 1
    for index, choice in enumerate(choices, start=1):
        if choice["name"] == status.resolved_model:
            default_index = index
        table.add_row(str(index), choice["name"], choice["source"], choice["summary"])
    console.print(table)

    selected_index = typer.prompt(
        "Select an Ollama model by number",
        default=str(default_index),
        show_default=True,
    )
    try:
        selected_choice = choices[int(selected_index) - 1]
    except (ValueError, IndexError):
        raise typer.BadParameter("Invalid model selection.")
    return selected_choice["name"]


def _core_setup_ready(status) -> bool:
    return status.db_ready and status.indexing_error is None


def _llm_backend_state(status) -> str:
    if status.ollama_binary_path and status.ollama_reachable and status.model_available:
        return "Ollama configured"
    return "No LLM configured"


@app.command()
def init() -> None:
    paths = discover_paths()
    logger.info("Running init", extra={"workspace_root": str(paths.workspace_root)})
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
    logger.info("Listing problems", extra={"workspace_root": str(paths.workspace_root)})
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
def describe(slug: str | None = typer.Argument(None, help="Problem slug to describe.")) -> None:
    paths = discover_paths()
    index_problem_bank, load_problem = _lazy_imports()
    logger.info("Describing problem", extra={"slug": slug})

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        slug = _resolve_target_slug(connection, slug)
        problem = load_problem(connection, slug)
        if problem is None:
            raise typer.BadParameter(f"Unknown problem slug: {slug}")
    finally:
        connection.close()

    tags_text = ", ".join(problem.metadata.tags) if problem.metadata.tags else "none"
    lines = [
        f"Title: {problem.metadata.title}",
        f"Slug: {problem.slug}",
        f"Difficulty: {problem.metadata.difficulty}",
        f"Category: {problem.metadata.category or 'uncategorized'}",
        f"Tags: {tags_text}",
        f"Function: {problem.metadata.function_name}",
        "",
        problem.body.strip(),
    ]
    console.print(Panel.fit("\n".join(lines), title="lcgrade describe"))


@app.command()
def history(slug: str | None = typer.Argument(None, help="Problem slug whose attempt history should be shown.")) -> None:
    paths = discover_paths()
    index_problem_bank, _ = _lazy_imports()
    logger.info("Showing history", extra={"slug": slug})

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        slug = _resolve_target_slug(connection, slug)
        history_view = fetch_problem_history_view(connection, slug)
    finally:
        connection.close()

    if history_view is None:
        raise typer.BadParameter(f"Unknown problem slug: {slug}")

    problem = history_view["problem"]
    milestones = history_view["milestones"]
    attempts = history_view["attempts"]

    if not attempts:
        lines = [
            f"History: {problem['title']} ({problem['slug']})",
            "No saved attempts yet.",
            *([f"Milestones: {line}" for line in _milestone_lines(milestones)] or ["Milestones: none yet."]),
        ]
        console.print(Panel.fit("\n".join(lines), title="lcgrade history"))
        return

    table = Table(title=f"History · {problem['title']}")
    table.add_column("#")
    table.add_column("Date")
    table.add_column("Bundled")
    table.add_column("LLM")
    table.add_column("Review")
    table.add_column("Follow-up")
    table.add_column("Complexity")
    table.add_column("Status")
    for item in attempts:
        attempt = item["attempt"]
        review = item["review"]
        complexity = review["complexity_time"] if review and review.get("complexity_time") else "—"
        table.add_row(
            str(item["attempt_number"]),
            _format_timestamp(str(attempt.get("timestamp"))),
            f"{attempt.get('bundled_passed', 0)}/{attempt.get('bundled_total', 0)}",
            f"{attempt.get('llm_passed', 0)}/{attempt.get('llm_total', 0)}",
            _history_review_state(item, milestones),
            _history_followup_state(item, milestones),
            str(complexity),
            str(attempt.get("status", "unknown")),
        )

    milestone_lines = _milestone_lines(milestones)
    console.print(table)
    console.print(
        Panel.fit(
            "\n".join(milestone_lines or ["No milestones completed yet."]),
            title="Milestones",
        )
    )


@app.command()
def stats() -> None:
    paths = discover_paths()
    index_problem_bank, _ = _lazy_imports()
    logger.info("Showing global stats")

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        counts = fetch_global_progress_counts(connection)
        by_difficulty = fetch_progress_counts_by_difficulty(connection)
        weakest_tags = fetch_weakest_tags_by_bundled_pass_rate(connection, limit=5)
    finally:
        connection.close()

    lines = [
        f"Solved: {counts['solved']}/{counts['total_problems']}",
        f"Attempted: {counts['attempted']}/{counts['total_problems']}",
        f"Review Generated: {counts['review_generated']}",
        f"Review Acknowledged: {counts['review_acknowledged']}",
        f"Follow-up Completed: {counts['followup_completed']}",
        "",
        "By Difficulty:",
    ]
    for row in by_difficulty:
        lines.append(f"  {row['difficulty']}: {row['solved']}/{row['total']} solved · {row['attempted']} attempted")
    if weakest_tags:
        lines.extend(
            [
                "",
                "Weakest Tags:",
                *[
                    f"  {row['tag']}: {int(round(float(row['pass_rate']) * 100))}% bundled pass rate across {row['attempts']} attempt(s)"
                    for row in weakest_tags
                ],
            ]
        )
    console.print(Panel.fit("\n".join(lines), title="lcgrade stats"))


@app.command("random")
def random_problem(
    difficulty: str | None = typer.Option(None, "--difficulty", help="Filter by difficulty."),
    tag: str | None = typer.Option(None, "--tag", help="Filter by tag."),
) -> None:
    paths = discover_paths()
    index_problem_bank, load_problem = _lazy_imports()
    logger.info("Selecting random problem", extra={"difficulty": difficulty, "tag": tag})

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        selected = select_random_unsolved_problem(connection, difficulty=difficulty, tag=tag)
        if selected is None:
            lines = [
                "No unsolved problem matched the current filters.",
                *(["Difficulty filter: " + difficulty] if difficulty else []),
                *(["Tag filter: " + tag] if tag else []),
            ]
            console.print(Panel.fit("\n".join(lines), title="lcgrade random"))
            raise typer.Exit(code=1)
        problem = load_problem(connection, str(selected["slug"]))
        if problem is None:
            raise typer.BadParameter(f"Unknown problem slug: {selected['slug']}")
        set_active_slug(connection, problem.slug)
    finally:
        connection.close()

    starter_path = problem.starter_path if problem.starter_path is not None else "No starter.py found."
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Active problem: {problem.metadata.title} ({problem.slug})",
                    f"Difficulty: {problem.metadata.difficulty}",
                    f"Function: {problem.metadata.function_name}",
                    f"Starter: {starter_path}",
                ]
            ),
            title="lcgrade random",
        )
    )


@app.command()
def setup(
    check: bool = typer.Option(False, "--check", help="Report current environment without mutating setup state."),
) -> None:
    paths = discover_paths()
    logger.info("Running setup", extra={"check_only": check, "workspace_root": str(paths.workspace_root)})
    status = inspect_setup(paths, check_only=check)

    if check:
        core_ready = _core_setup_ready(status)
        llm_state = _llm_backend_state(status)
        lines = [
            f"Workspace: {paths.workspace_root}",
            f"Data dir: {paths.data_dir}",
            f"DB path: {paths.db_path}",
            f"Config path: {paths.config_path}",
            f"Problem bank: {paths.problems_dir}",
            "",
            f"Core features: {'ready' if core_ready else 'not ready'}",
            f"Data dir exists: {'yes' if status.data_dir_exists else 'no'}",
            f"Database ready: {'yes' if status.db_ready else 'no'}",
            *([f"Database detail: {status.db_error}"] if status.db_error else []),
            (
                f"Problem bank readable: yes ({status.indexed_count} problem(s))"
                if status.indexing_error is None
                else f"Problem bank readable: no ({status.indexing_error})"
            ),
            "",
            f"LLM features: {llm_state}",
            f"Configured backend preference: {status.configured_backend}",
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
            "",
            "Core lcgrade solving works without any LLM backend.",
            "Enable local AI features:",
            *[f"  {command}" for command in local_llm_enablement_commands()],
            f"  `ollama pull {status.resolved_model}`",
            "Next product commands:",
            "  `python3 -m lcgrade.cli setup`",
            "  `python3 -m lcgrade.cli start two-sum`",
            "  `python3 -m lcgrade.cli solve`",
            "v0 uses repo-local problems/ and .lcgrade/ paths.",
        ]
        console.print(Panel.fit("\n".join(lines), title="lcgrade setup --check"))
        return

    if status.ollama_binary_path is None:
        if not _core_setup_ready(status):
            console.print(
                Panel.fit(
                    "\n".join(
                        [
                            "Core setup is incomplete.",
                            *(["Database detail: " + status.db_error] if status.db_error else []),
                            *(["Indexing detail: " + status.indexing_error] if status.indexing_error else []),
                            "Fix the core setup issues first, then re-run `python3 -m lcgrade.cli setup`.",
                        ]
                    ),
                    title="lcgrade setup",
                )
            )
            raise typer.Exit(code=1)
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        "Core lcgrade is ready.",
                        "LLM features are unavailable until a local backend is configured.",
                        "You can already use the offline workflow:",
                        "`python3 -m lcgrade.cli start two-sum`",
                        "`python3 -m lcgrade.cli solve`",
                        "",
                        "Enable local AI features later with:",
                        *local_llm_enablement_commands(),
                    ]
                ),
                title="lcgrade setup",
            )
        )
        return

    if not status.ollama_reachable:
        logger.warning("Ollama unreachable during setup", extra={"detail": status.ollama_detail})
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        "Ollama is installed but not reachable.",
                        "Run these commands, then re-run setup:",
                        "`ollama serve`",
                        "`python3 -m lcgrade.cli setup`",
                        *(["Detail: " + status.ollama_detail] if status.ollama_detail else []),
                    ]
                ),
                title="lcgrade setup",
            )
        )
        raise typer.Exit(code=1)

    selected_model = _select_setup_model(status)
    logger.info("Selected setup model", extra={"model": selected_model, "ram_gb": status.ram_gb})
    model_backend = OllamaBackend(model=selected_model)
    selected_model_installed = False
    try:
        selected_model_installed = model_backend.model_installed()
    except Exception as exc:
        logger.error("Selected model verification failed", extra={"model": selected_model, "error": str(exc)})
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"Could not verify selected model `{selected_model}`.",
                        "Run these commands, then re-run setup:",
                        "`ollama serve`",
                        f"`ollama pull {selected_model}`",
                        "`python3 -m lcgrade.cli setup`",
                        f"Detail: {exc}",
                    ]
                ),
                title="lcgrade setup",
            )
        )
        raise typer.Exit(code=1)

    if not selected_model_installed:
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"Selected model: {selected_model}",
                        f"Detected RAM: {status.ram_gb:.1f} GB",
                        "The selected model is not installed yet.",
                        "Recommended command:",
                        f"`ollama pull {selected_model}`",
                    ]
                ),
                title="lcgrade setup",
            )
        )
        if not typer.confirm(f"Pull `{selected_model}` now?", default=True):
            logger.warning("User declined Ollama pull", extra={"model": selected_model})
            console.print(
                Panel.fit(
                    "\n".join(
                        [
                            "Setup incomplete. Run these commands:",
                            f"`ollama pull {selected_model}`",
                            "`python3 -m lcgrade.cli setup`",
                        ]
                    ),
                    title="lcgrade setup",
                )
            )
            raise typer.Exit(code=1)

        console.print(f"Pulling {selected_model} via Ollama...")
        logger.info("Pulling Ollama model", extra={"model": selected_model})
        pulled, pull_error = pull_ollama_model(selected_model)
        if not pulled:
            logger.error("Ollama pull failed", extra={"model": selected_model, "error": pull_error})
            console.print(
                Panel.fit(
                    "\n".join(
                        [
                            f"Failed to pull {selected_model}.",
                            "Run this manually and retry setup:",
                            f"`ollama pull {selected_model}`",
                            "`python3 -m lcgrade.cli setup`",
                            *(["Error: " + pull_error] if pull_error else []),
                        ]
                    ),
                    title="lcgrade setup",
                )
            )
            raise typer.Exit(code=1)

    config = persist_setup_config(paths, backend="ollama", model=selected_model)
    logger.info("Persisted setup config", extra={"backend": config.backend, "model": config.model})
    final_status = inspect_setup(paths, check_only=False)
    if not final_status.db_ready or final_status.indexing_error is not None or not final_status.model_available:
        logger.error(
            "Setup validation failed",
            extra={
                "db_ready": final_status.db_ready,
                "indexing_error": final_status.indexing_error,
                "ollama_detail": final_status.ollama_detail,
            },
        )
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
        f"Core features: {'ready' if _core_setup_ready(final_status) else 'not ready'}",
        f"LLM features: {_llm_backend_state(final_status)}",
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
        "AI review, hints, and generated tests are optional local enhancements.",
        "v0 uses repo-local problems/ and .lcgrade/ paths.",
    ]
    console.print(Panel.fit("\n".join(lines), title="lcgrade setup"))


@app.command()
def start(slug: str = typer.Argument(..., help="Problem slug to make active.")) -> None:
    paths = discover_paths()
    logger.info("Starting problem", extra={"slug": slug})
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
        logger.info("Resetting problem state", extra={"slug": slug})
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
    logger.info("Pruning attempt history")
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
    logger.info(
        "Running solve",
        extra={"slug": slug, "requested_tests": tests, "force": force, "backend": backend or "configured"},
    )

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    connection = bootstrap_database(paths.db_path)
    try:
        index_problem_bank(connection, paths.problems_dir)
        slug = _resolve_target_slug(connection, slug)
        problem = load_problem(connection, slug)
        if problem is None:
            raise typer.BadParameter(f"Unknown problem slug: {slug}")

        try:
            llm_backend = _build_backend(paths, backend) if tests in {"llm", "both"} else None
            flow = solve_problem(
                connection,
                problem,
                solution_path=_resolve_solution_path(paths, solution),
                requested_test_mode=tests,
                force=force,
                llm=llm_backend,
            )
        except (ExecutionError, PreflightError, SolveFlowError) as exc:
            logger.error("Solve failed before completion", extra={"slug": slug, "error": str(exc)})
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        if flow.attempt.bundled_total > 0 and flow.attempt.bundled_passed == flow.attempt.bundled_total:
            clear_chat_messages(connection, problem.slug)
            if get_active_slug(connection) == problem.slug:
                clear_active_slug(connection)
            logger.info("Solve passed bundled tests and cleared active state", extra={"slug": problem.slug})
    finally:
        connection.close()

    llm_tests_message = flow.test_generation_result.warning if flow.test_generation_result is not None else None
    active_cleared = flow.attempt.bundled_total > 0 and flow.attempt.bundled_passed == flow.attempt.bundled_total
    if flow.test_generation_result is not None and flow.test_generation_result.generation_error:
        logger.warning(
            "LLM test generation fallback activated",
            extra={
                "slug": flow.problem.slug,
                "requested_mode": flow.requested_test_mode,
                "effective_mode": flow.effective_test_mode,
                "reason": flow.test_generation_result.generation_error,
            },
        )
    logger.info(
        "Solve completed",
        extra={
            "slug": flow.problem.slug,
            "used_cache": flow.used_cache,
            "requested_mode": flow.requested_test_mode,
            "effective_mode": flow.effective_test_mode,
            "status": flow.attempt.status,
        },
    )

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
                    *(
                        [
                            f"Active problem cleared after successful solve.",
                            f"Next: `lcgrade review {flow.problem.slug}` or `lcgrade start {flow.problem.slug}`.",
                        ]
                        if active_cleared
                        else []
                    ),
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
    logger.info("Running review", extra={"slug": slug, "backend": backend or "configured", "extend": extend})
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
    logger.info("Running chat", extra={"message_length": len(message)})
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
    logger.info("Running hint", extra={"tier": tier, "message_length": len(message or "")})
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
