# MLX Backend v1 Plan

## Goal

Add a local Apple Silicon backend that does not require the Ollama daemon. MLX is the first v1 backend priority because lcgrade's product value is offline, local interview practice.

## Minimum Backend Shape

- Implement `MLXBackend` behind the existing `LLMBackend` interface.
- Keep `generate()`, `available()`, and `model_info()` as the only required methods.
- Reuse the existing prompt builders for generated tests, review, chat, and hint.
- Keep Ollama as the v0 default until MLX install and model loading are reliable.

## Setup Behavior

- Detect Apple Silicon before recommending MLX.
- Show MLX as a local backend option alongside Ollama once implemented.
- Keep cloud API providers explicitly non-core and later.

## Validation

- Compare MLX and Ollama on the same prompts for test generation, review, chat, and hint.
- Record tokens/sec, first-token latency, memory usage, and failure modes.
- Keep the offline bundled-test workflow independent from any backend.
