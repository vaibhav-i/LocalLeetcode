# lcgrade Live Ollama Integration

- Date: 2026-05-06
- Scope: optional integration track, not a `v0` signoff blocker

## Environment Status

- `lcgrade setup --check` reported `Ollama reachable: no`
- Localhost access to the Ollama daemon is blocked from this Codex sandbox with `Operation not permitted`
- Because this is environment-level access restriction, live inference scenarios still need to be run from the user's terminal

## Requested Live Scenarios

- `solve --tests llm`
- `solve --tests both`
- `review`
- `chat`
- `hint`

## Recorded Outcome

- Live Ollama validation: skipped from Codex sandbox
- Graceful degradation behavior: verified
  - `solve` fell back to bundled-only mode with warning
  - LLM-dependent commands report that local AI features need a local backend
  - core offline setup and bundled solves remain usable without Ollama

## Next Live Validation Step

When Ollama is available, rerun the five scenarios above and record:

- model/backend metadata
- whether generation happened or fallback occurred
- whether outputs matched expected command-level behavior
