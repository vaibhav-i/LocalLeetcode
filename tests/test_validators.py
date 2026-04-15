from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from lcgrade.validators import ExactMatch, SetEquality, get_validator, load_custom_validator


def test_exact_and_set_validators_follow_beta_registry_contract() -> None:
    exact = get_validator("exact")
    set_validator = get_validator("set_equality")

    assert isinstance(exact, ExactMatch)
    assert isinstance(set_validator, SetEquality)
    assert exact.check({"items": [1, 2]}, {"items": [1, 2]})
    assert not exact.check({"items": [1, 2]}, {"items": [2, 1]})
    assert set_validator.check([1, 2, 3], [3, 2, 1])
    assert not set_validator.check([1, 2, 3], [1, 2, 4])


def test_load_custom_validator_from_tempfile(tmp_path: Path) -> None:
    validator_path = tmp_path / "custom_rule.py"
    validator_path.write_text(
        dedent(
            """
            def check(expected, actual, input_data):
                return actual == expected + input_data["offset"]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    validator = load_custom_validator(validator_path)

    assert validator.validator_name == "custom_rule"
    assert validator.check(3, 5, {"offset": 2})
    assert not validator.check(3, 4, {"offset": 2})
