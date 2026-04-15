from __future__ import annotations

from lcgrade.llm import OllamaBackend


def test_ollama_backend_uses_environment_configuration(monkeypatch) -> None:
    monkeypatch.setenv("LCGRADE_OLLAMA_MODEL", "llama3.1")
    monkeypatch.setenv("LCGRADE_OLLAMA_BASE_URL", "http://ollama.local:11434/")
    monkeypatch.setenv("LCGRADE_OLLAMA_TIMEOUT", "12.5")

    backend = OllamaBackend()

    assert backend.model == "llama3.1"
    assert backend.base_url == "http://ollama.local:11434"
    assert backend.timeout == 12.5


def test_ollama_backend_available_requires_configured_model(monkeypatch) -> None:
    backend = OllamaBackend(model="llama3.2")
    monkeypatch.setattr(backend, "installed_models", lambda: ("mistral:latest",))

    assert backend.available() is False
    assert backend.unavailable_reason() == (
        "Ollama is running, but model 'llama3.2' is not installed. Installed models: mistral:latest"
    )


def test_ollama_backend_matches_model_without_explicit_tag(monkeypatch) -> None:
    backend = OllamaBackend(model="llama3.2")
    monkeypatch.setattr(backend, "installed_models", lambda: ("llama3.2:latest",))

    assert backend.available() is True
    assert backend.unavailable_reason() is None
