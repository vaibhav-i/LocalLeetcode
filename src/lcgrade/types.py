from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FunctionParam:
    name: str
    type: str


@dataclass(slots=True)
class ProblemDefinition:
    schema_version: int
    title: str
    slug: str
    difficulty: str
    tags: list[str]
    category: str
    function_name: str
    params: list[FunctionParam]
    return_type: str
    validator: str
    statement_path: Path
    problem_dir: Path


@dataclass(slots=True)
class TestCase:
    name: str
    input: dict[str, Any]
    expected: Any
    validator: str
    source: str = "verified"


@dataclass(slots=True)
class TestVerdict:
    name: str
    passed: bool
    input: dict[str, Any]
    expected: Any
    actual: Any = None
    error: str | None = None
    source: str = "verified"


@dataclass(slots=True)
class AttemptSummary:
    slug: str
    test_mode: str
    bundled_passed: int
    bundled_total: int
    llm_passed: int
    llm_total: int
    status: str
    runtime_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    review_generated: bool = False
