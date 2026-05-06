# lcgrade Test Matrix

- Date: 2026-05-06
- Scope: expanded problem bank, offline-first setup, installed CLI smoke, and optional live Ollama check

## Deterministic Automated Coverage

Command run:

```bash
python3 -m pytest tests -q
```

Result:

- Status: pass
- Summary: `71 passed, 2 skipped in 1.64s`

Subsystem coverage summary:

- Problem parsing/indexing: pass
- Validators: pass
- Solve flow, cache, and preflight: pass
- Sandbox execution: pass
- Review pipeline: pass
- Command surface: pass
- Test-generation fallback behavior: pass
- Ollama backend configuration and model-availability checks: pass
- Expanded bundled problem bank indexing: pass, 12 problem(s)
- Installed `lcgrade` console command smoke: pass

## Manual CLI Coverage

- `setup`: pass
- `start`: pass
- `solve` with active problem default: pass
- `review` with active problem default: pass
- `chat`: pass for graceful unavailable-backend behavior
- `hint`: pass for graceful unavailable-backend behavior
- `reset`: pass
- `prune`: pass
- `describe`, `history`, `stats`, and `random`: pass
- New reference-solution bundled solves: pass

## Optional Live Ollama Coverage

- Status: blocked in this Codex sandbox by local daemon access restrictions
- Outcome: deterministic graceful-degradation behavior and installed CLI behavior verified instead
