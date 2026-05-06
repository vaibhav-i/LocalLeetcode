# lcgrade

`lcgrade` is a local-first CLI grader for LeetCode-style interview practice. Core solving works offline with bundled problems and tests. Local AI features such as generated tests, review, chat, and hints are optional enhancements backed by a local model.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

The installed command is `lcgrade`. You can also run the package with `python3 -m lcgrade.cli`.

## First Run

```bash
lcgrade setup
lcgrade list-problems
lcgrade start two-sum
```

Open the starter file printed by `start`, implement the function, then grade it:

```bash
lcgrade solve
```

You can also target a specific file:

```bash
lcgrade solve two-sum --solution problems/two-sum/solutions/reference.py
```

## Core Commands

```bash
lcgrade describe two-sum
lcgrade random
lcgrade random --difficulty easy
lcgrade history two-sum
lcgrade stats
lcgrade reset two-sum
lcgrade prune
```

These commands do not require Ollama or any LLM backend.

## Optional Local AI

`review`, `chat`, `hint`, and `solve --tests llm|both` need a local LLM backend. In v0, the supported backend is Ollama.

```bash
brew install ollama
ollama serve
ollama pull qwen2.5-coder:7b
lcgrade setup
```

Then run:

```bash
lcgrade solve two-sum --tests both
lcgrade review two-sum
lcgrade start two-sum
lcgrade chat "What should I improve?"
lcgrade hint 1
```

If Ollama is not installed, `lcgrade setup` still completes the offline core setup and prints the local AI enablement commands.

## Debugging

Use `--debug` for stderr diagnostics and `.lcgrade/lcgrade.log`:

```bash
lcgrade --debug setup --check
lcgrade --debug solve two-sum
```

## Development

```bash
pytest
lcgrade list-problems
lcgrade solve two-sum --solution problems/two-sum/solutions/reference.py
```

The bundled problem bank currently uses repo-local `problems/` and `.lcgrade/` paths for v0 development.
