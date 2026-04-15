"""Validation strategies for lcgrade test comparisons."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
import importlib.util
import re
from pathlib import Path
from typing import Any, Callable, TypeVar


class Validator(ABC):
    """Base class for test output comparison."""

    validator_name: str | None = None

    @abstractmethod
    def check(
        self,
        expected: Any,
        actual: Any,
        input_data: Mapping[str, Any] | None = None,
    ) -> bool:
        """Return True when actual is correct for the given input."""


_VALIDATOR_REGISTRY: dict[str, type[Validator]] = {}
_T = TypeVar("_T", bound=type[Validator])


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def register_validator(*names: str) -> Callable[[_T], _T]:
    """Register a validator class under one or more lookup names."""

    def decorator(cls: _T) -> _T:
        keys = list(names) if names else []
        keys.extend(
            [
                cls.validator_name or _snake_case(cls.__name__),
                cls.__name__,
                _snake_case(cls.__name__),
            ]
        )
        for key in keys:
            _VALIDATOR_REGISTRY[key.lower()] = cls
        return cls

    return decorator


def available_validators() -> tuple[str, ...]:
    return tuple(sorted(_VALIDATOR_REGISTRY))


def get_validator(name: str, **kwargs: Any) -> Validator:
    try:
        cls = _VALIDATOR_REGISTRY[name.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown validator: {name!r}") from exc
    return cls(**kwargs)


def validator_registry() -> dict[str, type[Validator]]:
    """Return a copy of the validator registry."""

    return dict(_VALIDATOR_REGISTRY)


@register_validator("exact", "exact_match")
class ExactMatch(Validator):
    """Strict equality validator."""

    validator_name = "exact_match"

    def check(
        self,
        expected: Any,
        actual: Any,
        input_data: Mapping[str, Any] | None = None,
    ) -> bool:
        del input_data
        return expected == actual


@register_validator("set", "set_equality")
class SetEquality(Validator):
    """Order-insensitive comparison for list-like outputs."""

    validator_name = "set_equality"

    def check(
        self,
        expected: Any,
        actual: Any,
        input_data: Mapping[str, Any] | None = None,
    ) -> bool:
        del input_data
        if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            try:
                return set(expected) == set(actual)
            except TypeError:
                return expected == actual
        return expected == actual


@dataclass(slots=True, frozen=True)
class ValidatorRecord:
    """Lightweight registry snapshot for debugging and tests."""

    name: str
    validator: type[Validator]


def load_custom_validator(validator_path: str | Path) -> Validator:
    """Load a problem-specific validator module and instantiate it.

    This is intentionally minimal beta scaffolding. If the module exposes a
    callable named `check`, we wrap it in a tiny Validator adapter.
    """

    path = Path(validator_path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load validator from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    check = getattr(module, "check", None)
    if not callable(check):
        raise AttributeError(f"{path} does not define a callable check()")

    class _CustomValidator(Validator):
        validator_name = path.stem

        def check(
            self,
            expected: Any,
            actual: Any,
            input_data: Mapping[str, Any] | None = None,
        ) -> bool:
            return bool(check(expected, actual, input_data))

    return _CustomValidator()
