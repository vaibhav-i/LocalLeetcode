from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .problems import ProblemDocument


class PreflightError(RuntimeError):
    """Raised when source code fails deterministic preflight checks."""


@dataclass(slots=True, frozen=True)
class PreflightResult:
    solution_path: Path
    function_name: str
    parameter_count: int


def run_preflight(problem: ProblemDocument, solution_path: Path) -> PreflightResult:
    source = solution_path.read_text(encoding="utf-8")
    try:
        module = ast.parse(source, filename=str(solution_path))
    except SyntaxError as exc:
        location = f"line {exc.lineno}, column {exc.offset}"
        raise PreflightError(f"Syntax error in {solution_path} at {location}: {exc.msg}") from exc

    function = _find_function(module, problem.metadata.function_name)
    if function is None:
        raise PreflightError(
            f"Expected function {problem.metadata.function_name!r} in {solution_path}"
        )

    actual_arity = len(function.args.args)
    expected_arity = len(problem.metadata.params)
    if actual_arity != expected_arity:
        raise PreflightError(
            f"Function {problem.metadata.function_name!r} expected {expected_arity} parameter(s) "
            f"but found {actual_arity} in {solution_path}"
        )

    return PreflightResult(
        solution_path=solution_path,
        function_name=problem.metadata.function_name,
        parameter_count=actual_arity,
    )


def _find_function(module: ast.Module, function_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    return None
