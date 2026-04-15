from __future__ import annotations

import json
from pathlib import Path

from lcgrade.llm import LLMBackend, LLMResponse, MockBackend, ModelInfo
from lcgrade.problems import ProblemDocument, ProblemMetadata, ProblemParam
from lcgrade.test_generation import generate_test_generation


def build_problem(tmp_path: Path) -> ProblemDocument:
    problem_dir = tmp_path / "two-sum"
    problem_dir.mkdir()
    statement_path = problem_dir / "statement.md"
    statement_path.write_text(
        "---\n"
        "schema_version: 1\n"
        "title: Two Sum\n"
        "slug: two-sum\n"
        "difficulty: easy\n"
        "function_name: two_sum\n"
        "params:\n"
        "  - name: nums\n"
        "    type: list[int]\n"
        "  - name: target\n"
        "    type: int\n"
        "return_type: list[int]\n"
        "---\n"
        "Find two indices.\n",
        encoding="utf-8",
    )
    tests_path = problem_dir / "tests.json"
    tests_path.write_text(
        json.dumps(
            [
                {
                    "input": {"nums": [2, 7, 11, 15], "target": 9},
                    "expected": [0, 1],
                    "validator": "exact_match",
                    "source": "verified",
                }
            ]
        ),
        encoding="utf-8",
    )
    starter_path = problem_dir / "starter.py"
    starter_path.write_text(
        "def two_sum(nums, target):\n    return []\n",
        encoding="utf-8",
    )

    metadata = ProblemMetadata(
        schema_version=1,
        title="Two Sum",
        slug="two-sum",
        difficulty="easy",
        tags=(),
        category="Arrays",
        function_name="two_sum",
        params=(
            ProblemParam(name="nums", type="list[int]"),
            ProblemParam(name="target", type="int"),
        ),
        return_type="list[int]",
        validator="exact_match",
    )
    return ProblemDocument(
        metadata=metadata,
        body="Find two indices.",
        problem_dir=problem_dir,
        statement_path=statement_path,
        tests_path=tests_path,
        starter_path=starter_path,
        validator_path=None,
        reference_solution_path=None,
        statement_hash="statement-hash",
        statement_mtime=1.0,
        tests_hash="tests-hash",
        tests_mtime=1.0,
    )


def test_generate_test_generation_falls_back_when_backend_unavailable(tmp_path: Path) -> None:
    problem = build_problem(tmp_path)

    class UnavailableBackend(LLMBackend):
        def generate(
            self,
            prompt: str,
            system_prompt: str | None = None,
            temperature: float = 0.7,
            max_tokens: int = 2048,
        ) -> LLMResponse:
            raise AssertionError("generate() should not be called when unavailable")

        def available(self) -> bool:
            return False

        def model_info(self) -> ModelInfo:
            return ModelInfo(
                name="unavailable",
                context_window=0,
                quantization="unknown",
                backend="mock",
            )

    result = generate_test_generation(problem, requested_test_mode="both", llm=UnavailableBackend())

    assert result.requested_test_mode == "both"
    assert result.effective_test_mode == "bundled"
    assert result.llm_requested is True
    assert result.llm_available is False
    assert result.llm_used is False
    assert result.warning == "LLM backend unavailable; using bundled tests only."
    assert len(result.bundled_test_cases) == 1
    assert result.generated_test_cases == ()
    assert len(result.combined_test_cases()) == 1
    metadata = result.to_pipeline_metadata()
    assert metadata["generated_count"] == 0
    assert metadata["llm_available"] is False


def test_generate_test_generation_uses_mockbackend_json(tmp_path: Path) -> None:
    problem = build_problem(tmp_path)
    payload = {
        "test_cases": [
            {
                "name": "llm-empty",
                "input": {"nums": [], "target": 0},
                "expected": [],
                "validator": "exact_match",
                "source": "llm",
                "rationale": "Empty input should be handled cleanly.",
            }
        ],
        "notes": "one edge case",
    }
    backend = MockBackend(responses=[json.dumps(payload)])

    result = generate_test_generation(problem, requested_test_mode="llm", llm=backend)

    assert result.requested_test_mode == "llm"
    assert result.effective_test_mode == "llm"
    assert result.llm_requested is True
    assert result.llm_available is True
    assert result.llm_used is True
    assert result.warning is None
    assert result.generated_test_cases[0].name == "llm-empty"
    assert result.generated_test_cases[0].input == {"nums": [], "target": 0}
    assert result.generated_test_cases[0].expected == []
    assert result.generated_test_cases[0].source == "llm"
    assert result.generated_test_cases[0].rationale == "Empty input should be handled cleanly."
    assert result.raw_payload == payload
    assert backend.prompts[0]["system_prompt"] == "You are a structured test-case generator for lcgrade."
    assert "Return only JSON." in backend.prompts[0]["prompt"]
    assert "llm-empty" in json.dumps(result.to_pipeline_metadata(), sort_keys=True)
