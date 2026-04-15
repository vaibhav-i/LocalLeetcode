"""LLM backend abstractions for lcgrade.

This module keeps the core inference contract small and stable:
`LLMBackend` handles shared JSON parsing/retry logic, while concrete
backends only implement transport-specific generation and metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import time
from typing import Any, Mapping, Sequence
from urllib import error, request


@dataclass(slots=True, frozen=True)
class LLMResponse:
    """A single model response with lightweight usage metadata."""

    text: str
    tokens_used: int = 0
    latency_ms: float = 0.0


@dataclass(slots=True, frozen=True)
class ModelInfo:
    """Model metadata used for token budgeting and UX."""

    name: str
    context_window: int
    quantization: str
    backend: str


class LLMBackend(ABC):
    """Base class for LLM inference backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a prompt to the backend and return a text response."""

    @abstractmethod
    def available(self) -> bool:
        """Return True when this backend can serve requests."""

    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Return metadata about the active model."""

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        retries: int = 3,
        max_tokens: int = 2048,
    ) -> Any:
        """Generate JSON output with a small retry loop.

        The method is intentionally backend-agnostic. Subclasses can override
        this if they later expose native JSON modes, but the beta path works
        with any text-only model.
        """

        last_error: json.JSONDecodeError | None = None
        base_prompt = prompt
        for attempt in range(retries):
            response = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                return self._parse_json_response(response.text)
            except json.JSONDecodeError as exc:
                last_error = exc
                if attempt + 1 < retries:
                    prompt = self._json_retry_prompt(base_prompt, response.text)

        if last_error is not None:
            raise last_error
        raise RuntimeError("JSON generation failed without a parse error.")

    @staticmethod
    def _json_retry_prompt(original_prompt: str, bad_output: str) -> str:
        """Add a lightweight correction hint for another JSON attempt."""

        del bad_output
        return (
            f"{original_prompt.rstrip()}\n\n"
            "Return only valid JSON. Do not wrap the answer in prose or markdown."
        )

    @classmethod
    def _parse_json_response(cls, text: str) -> Any:
        """Best-effort JSON extraction for model outputs.

        This strips fenced code blocks, then falls back to extracting the first
        balanced-looking object/array payload from the response.
        """

        cleaned = cls._strip_markdown_fences(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as first_error:
            extracted = cls._extract_json_substring(cleaned)
            if extracted is not None and extracted != cleaned:
                return json.loads(extracted)
            raise first_error

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()
        if cleaned.lower().startswith("json\n"):
            cleaned = cleaned[4:].lstrip()
        return cleaned

    @staticmethod
    def _extract_json_substring(text: str) -> str | None:
        first_object = text.find("{")
        first_array = text.find("[")
        starts = [index for index in (first_object, first_array) if index != -1]
        if not starts:
            return None

        start = min(starts)
        end_object = text.rfind("}")
        end_array = text.rfind("]")
        end = max(end_object, end_array)
        if end == -1 or end <= start:
            return None
        return text[start : end + 1].strip()


class OllamaBackend(LLMBackend):
    """Small HTTP-backed Ollama backend with no extra dependencies."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        start = time.perf_counter()
        data = self._request_json("POST", "/api/generate", payload)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        text = str(data.get("response", ""))
        tokens_used = self._coerce_int(
            data.get("eval_count")
            or data.get("prompt_eval_count")
            or data.get("total_tokens")
            or 0
        )
        latency_ms = self._duration_to_ms(data.get("total_duration"))
        if latency_ms <= 0:
            latency_ms = elapsed_ms

        return LLMResponse(text=text, tokens_used=tokens_used, latency_ms=latency_ms)

    def available(self) -> bool:
        try:
            self._request_json("GET", "/api/tags", None)
        except Exception:
            return False
        return True

    def model_info(self) -> ModelInfo:
        try:
            data = self._request_json("POST", "/api/show", {"model": self.model})
        except Exception:
            return ModelInfo(
                name=self.model,
                context_window=4096,
                quantization="unknown",
                backend="ollama",
            )

        context_window = self._extract_context_window(data)
        quantization = self._extract_quantization(data)
        return ModelInfo(
            name=self.model,
            context_window=context_window,
            quantization=quantization,
            backend="ollama",
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body: bytes | None = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(url, data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except error.URLError as exc:
            raise ConnectionError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        if not raw.strip():
            return {}
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object from Ollama, got {type(parsed).__name__}")
        return parsed

    @staticmethod
    def _duration_to_ms(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return float(value) / 1_000_000.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coerce_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _extract_context_window(data: Mapping[str, Any]) -> int:
        candidates = [
            data.get("context_window"),
            data.get("context_length"),
        ]
        model_info = data.get("model_info")
        if isinstance(model_info, Mapping):
            candidates.extend(
                [
                    model_info.get("llama.context_length"),
                    model_info.get("context_length"),
                    model_info.get("context_window"),
                ]
            )
        details = data.get("details")
        if isinstance(details, Mapping):
            candidates.extend([details.get("context_length"), details.get("num_ctx")])

        for candidate in candidates:
            try:
                if candidate is not None:
                    return int(candidate)
            except (TypeError, ValueError):
                continue
        return 4096

    @staticmethod
    def _extract_quantization(data: Mapping[str, Any]) -> str:
        candidates: Sequence[Any] = [data.get("quantization"), data.get("quantization_level")]
        details = data.get("details")
        if isinstance(details, Mapping):
            candidates = (*candidates, details.get("quantization_level"), details.get("quantization"))

        for candidate in candidates:
            if candidate:
                return str(candidate)
        return "unknown"


class MockBackend(LLMBackend):
    """Deterministic backend for tests and local development."""

    def __init__(
        self,
        responses: Sequence[str | LLMResponse] | None = None,
        default_response: str | LLMResponse = "",
        model: str = "mock-model",
        context_window: int = 8192,
        quantization: str = "none",
    ) -> None:
        self._responses = list(responses or [])
        self._default_response = default_response
        self._index = 0
        self._prompts: list[dict[str, Any]] = []
        self._model_info = ModelInfo(
            name=model,
            context_window=context_window,
            quantization=quantization,
            backend="mock",
        )

    @property
    def prompts(self) -> list[dict[str, Any]]:
        """Captured calls for assertions in tests."""

        return list(self._prompts)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self._prompts.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
        else:
            response = self._default_response

        if isinstance(response, LLMResponse):
            return response
        return LLMResponse(text=str(response), tokens_used=0, latency_ms=0.0)

    def available(self) -> bool:
        return True

    def model_info(self) -> ModelInfo:
        return self._model_info
