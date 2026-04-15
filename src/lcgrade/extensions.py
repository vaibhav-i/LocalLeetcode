"""Extension abstractions and beta extension implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from .llm import LLMBackend


@dataclass(slots=True, frozen=True)
class TimingData:
    """Empirical profiling data for a solution input size."""

    input_size: int
    runtime_ms: float


@dataclass(slots=True, frozen=True)
class ExtensionContext:
    """Typed boundary passed into extensions."""

    problem_statement: str
    user_code: str
    verdicts: list[Any]
    review_output: str
    timing_data: list[TimingData] | None = None


@dataclass(slots=True, frozen=True)
class ExtensionResult:
    """A rendered extension section."""

    title: str
    content: str


class Extension(ABC):
    """Base class for pluggable analysis steps."""

    extension_name: str | None = None

    @abstractmethod
    def name(self) -> str:
        """Return the unique registry key for this extension."""

    @abstractmethod
    def run(self, ctx: ExtensionContext, llm: LLMBackend) -> ExtensionResult:
        """Execute the extension and return rendered content."""


_EXTENSION_REGISTRY: dict[str, type[Extension]] = {}
_T = TypeVar("_T", bound=type[Extension])


def register_extension(*names: str) -> Callable[[_T], _T]:
    """Register an extension class under one or more lookup names."""

    def decorator(cls: _T) -> _T:
        keys = list(names) if names else []
        keys.extend([(cls.extension_name or cls.__name__).lower(), cls.__name__.lower()])
        for key in keys:
            _EXTENSION_REGISTRY[key.lower()] = cls
        return cls

    return decorator


def available_extensions() -> tuple[str, ...]:
    return tuple(sorted(_EXTENSION_REGISTRY))


def get_extension(name: str, **kwargs: Any) -> Extension:
    try:
        cls = _EXTENSION_REGISTRY[name.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown extension: {name!r}") from exc
    return cls(**kwargs)


def extension_registry() -> dict[str, type[Extension]]:
    return dict(_EXTENSION_REGISTRY)


def _context_prompt(ctx: ExtensionContext) -> str:
    lines = [
        "Problem statement:",
        ctx.problem_statement.strip(),
        "",
        "User code:",
        ctx.user_code.strip(),
        "",
        "Stage 2 review:",
        ctx.review_output.strip(),
    ]
    if ctx.verdicts:
        lines.extend(["", "Verdicts:", "\n".join(f"- {verdict!r}" for verdict in ctx.verdicts)])
    if ctx.timing_data:
        timing_lines = [
            f"- input_size={item.input_size}, runtime_ms={item.runtime_ms:.3f}"
            for item in ctx.timing_data
        ]
        lines.extend(["", "Timing data:", *timing_lines])
    return "\n".join(lines)


@register_extension("interview")
class InterviewQuestions(Extension):
    """Generate follow-up interview questions."""

    extension_name = "interview"

    def name(self) -> str:
        return "interview"

    def run(self, ctx: ExtensionContext, llm: LLMBackend) -> ExtensionResult:
        prompt = (
            "Generate 2-3 concise follow-up interview questions for the candidate.\n"
            "Focus on reasoning, trade-offs, edge cases, and implementation choices.\n"
            "Do not provide solutions.\n\n"
            f"{_context_prompt(ctx)}"
        )
        response = llm.generate(
            prompt=prompt,
            system_prompt="You are an interview coach. Ask short, high-signal follow-up questions.",
            temperature=0.4,
            max_tokens=512,
        )
        return ExtensionResult(title="Interview Follow-ups", content=response.text)


@register_extension("optimize")
class OptimizationPrompt(Extension):
    """Generate optimization suggestions for the submitted solution."""

    extension_name = "optimize"

    def name(self) -> str:
        return "optimize"

    def run(self, ctx: ExtensionContext, llm: LLMBackend) -> ExtensionResult:
        prompt = (
            "Analyze the solution for potential optimizations.\n"
            "Cover time complexity, space complexity, readability, and practical trade-offs.\n"
            "Keep the response concrete and actionable.\n\n"
            f"{_context_prompt(ctx)}"
        )
        response = llm.generate(
            prompt=prompt,
            system_prompt="You are a senior engineer reviewing algorithmic optimizations.",
            temperature=0.3,
            max_tokens=700,
        )
        return ExtensionResult(title="Optimization Suggestions", content=response.text)
