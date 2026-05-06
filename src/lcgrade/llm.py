"""LLM backend abstractions for lcgrade.

This module keeps the core inference contract small and stable:
`LLMBackend` handles shared JSON parsing/retry logic, while concrete
backends only implement transport-specific generation and metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
import time
from typing import Any, Mapping, Sequence
from urllib import error, request

from .logging_utils import get_logger

logger = get_logger("lcgrade.llm")


def local_llm_enablement_commands() -> tuple[str, ...]:
    return (
        "`brew install ollama`",
        "`ollama serve`",
        "`lcgrade setup`",
    )


def local_llm_required_message(reason: str | None = None) -> str:
    lines = [
        "This feature needs a local LLM backend.",
        "Core lcgrade solving still works without one.",
        "To enable local AI features today, run:",
        *local_llm_enablement_commands(),
    ]
    cleaned_reason = (reason or "").strip()
    if cleaned_reason:
        lines.append(f"Detail: {cleaned_reason}")
    return "\n".join(lines)


def local_llm_test_fallback_message(reason: str | None = None) -> str:
    cleaned_reason = (reason or "").strip()
    lines = [
        "Local AI-generated tests were skipped.",
        "Using bundled tests only.",
        "Run `python3 -m lcgrade.cli setup` to enable local AI features.",
    ]
    if cleaned_reason:
        lines.insert(1, f"Detail: {cleaned_reason.rstrip('.')}.")
    return " ".join(lines)

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

    def unavailable_reason(self) -> str | None:
        """Return a user-facing reason when the backend cannot serve requests."""

        try:
            if self.available():
                return None
        except Exception as exc:  # pragma: no cover - defensive fallback
            return f"LLM backend unavailable: {exc}"
        return "LLM backend unavailable."

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
        logger.debug("LLM generate_json prompt:\n%s", prompt)
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
                logger.warning("LLM JSON parse retry %s/%s failed: %s", attempt + 1, retries, exc)
                logger.debug("LLM JSON parse raw response:\n%s", response.text)
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
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model or os.environ.get("LCGRADE_OLLAMA_MODEL", "llama3.2")
        resolved_base_url = base_url or os.environ.get("LCGRADE_OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = resolved_base_url.rstrip("/")
        timeout_value = timeout
        if timeout_value is None:
            raw_timeout = os.environ.get("LCGRADE_OLLAMA_TIMEOUT")
            if raw_timeout:
                try:
                    timeout_value = float(raw_timeout)
                except ValueError:
                    timeout_value = 30.0
            else:
                timeout_value = 30.0
        self.timeout = float(timeout_value)
        logger.info("Initialized Ollama backend model=%s base_url=%s timeout=%.1f", self.model, self.base_url, self.timeout)

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

        logger.debug("Ollama generate payload for model=%s:\n%s", self.model, json.dumps(payload, ensure_ascii=False))
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
        logger.debug("Ollama raw response for model=%s:\n%s", self.model, json.dumps(data, ensure_ascii=False))

        return LLMResponse(text=text, tokens_used=tokens_used, latency_ms=latency_ms)

    def available(self) -> bool:
        try:
            return self.model_installed()
        except Exception:
            return False

    def unavailable_reason(self) -> str | None:
        try:
            models = self.installed_models()
        except Exception as exc:
            return f"Could not reach Ollama at {self.base_url}: {exc}"
        if self._model_matches_any(models):
            return None
        if models:
            return (
                f"Ollama is running, but model {self.model!r} is not installed. "
                f"Installed models: {', '.join(models)}"
            )
        return f"Ollama is running, but no models are installed. Expected {self.model!r}."

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
        logger.debug(
            "Ollama model info summary model=%s context_window=%s quantization=%s keys=%s",
            self.model,
            context_window,
            quantization,
            sorted(data.keys()),
        )
        return ModelInfo(
            name=self.model,
            context_window=context_window,
            quantization=quantization,
            backend="ollama",
        )

    def installed_models(self) -> tuple[str, ...]:
        data = self._request_json("GET", "/api/tags", None)
        models = data.get("models", [])
        if not isinstance(models, Sequence):
            return ()
        names: list[str] = []
        for item in models:
            if isinstance(item, Mapping):
                name = item.get("name")
                if name:
                    names.append(str(name))
        return tuple(names)

    def model_installed(self) -> bool:
        return self._model_matches_any(self.installed_models())

    def _model_matches_any(self, installed_models: Sequence[str]) -> bool:
        target = self.model.strip()
        if not target:
            return False
        target_base = target.split(":", 1)[0]
        for installed in installed_models:
            installed_text = str(installed).strip()
            if installed_text == target:
                return True
            if installed_text.split(":", 1)[0] == target_base:
                return True
        return False

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
        logger.debug("Ollama request method=%s url=%s", method, url)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except error.URLError as exc:
            logger.debug("Ollama request failed method=%s url=%s error=%s", method, url, exc)
            raise ConnectionError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        if not raw.strip():
            return {}
        if path == "/api/show":
            logger.debug("Ollama raw body received from %s (length=%s)", path, len(raw))
        else:
            logger.debug("Ollama raw body from %s:\n%s", path, raw)
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


class MLXBackend(LLMBackend):
    """Beta placeholder for future Apple Silicon-native inference support."""

    def __init__(self, model: str = "mlx-community") -> None:
        self.model = model

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        del prompt, system_prompt, temperature, max_tokens
        raise NotImplementedError("MLX backend is not implemented yet.")

    def available(self) -> bool:
        return False

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model,
            context_window=4096,
            quantization="unknown",
            backend="mlx",
        )
