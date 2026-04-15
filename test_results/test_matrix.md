# lcgrade Test Matrix

- Date: 2026-04-15
- Scope: deterministic `v0` signoff plus optional live Ollama check

## Deterministic Automated Coverage

Command run:

```bash
python3 -m pytest tests -q
```

Result:

- Status: pass
- Summary: `44 passed in 2.18s`

Subsystem coverage summary:

- Problem parsing/indexing: pass
- Validators: pass
- Solve flow, cache, and preflight: pass
- Sandbox execution: pass
- Review pipeline: pass
- Command surface: pass
- Test-generation fallback behavior: pass
- Ollama backend configuration and model-availability checks: pass

## Manual CLI Coverage

- `setup`: pass
- `start`: pass
- `solve` with active problem default: pass
- `review` with active problem default: pass
- `chat`: pass for graceful unavailable-backend behavior
- `hint`: pass for graceful unavailable-backend behavior
- `reset`: pass
- `prune`: pass

## Optional Live Ollama Coverage

- Status: not available in this audit environment
- Outcome: deterministic graceful-degradation behavior verified instead
