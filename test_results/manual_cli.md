# lcgrade Manual CLI Results

- Date: 2026-04-15
- Workspace: repo-local `.lcgrade/`

## 1. `lcgrade setup`

- Command: `python3 -m lcgrade.cli setup`
- Expected: prints workspace/data/db/problem-bank paths, DB readiness, problem indexing, and Ollama availability
- Actual: reported DB ready, indexed 3 problems, and showed `Ollama reachable: no`
- Result: pass

## 2. `lcgrade start <slug>`

- Command: `python3 -m lcgrade.cli start two-sum`
- Expected: sets active problem and prints starter path
- Actual: reported `Active problem: Two Sum (two-sum)` and showed the repo-local `starter.py` path
- Result: pass

## 3. `lcgrade solve` with active problem default

- Command: `python3 -m lcgrade.cli solve --solution problems/two-sum/solutions/reference.py`
- Expected: uses active problem without requiring a slug, runs grading, and clears active/chat state on bundled pass
- Actual: graded `two-sum` successfully with `Bundled tests: 2/2`, saved a new attempt snapshot, then SQLite metadata showed no `active_slug` and zero `chat_messages` for `two-sum`
- Result: pass

## 4. `lcgrade review` with active problem default

- Prep: `python3 -m lcgrade.cli start two-sum`
- Command: `python3 -m lcgrade.cli review`
- Expected: targets the active problem and uses the latest saved attempt
- Actual: resolved the active problem and returned graceful Stage 2 skip output: `LLM backend unavailable.`
- Result: pass

## 5. `lcgrade chat`

- Command: `python3 -m lcgrade.cli chat "How should I approach this?"`
- Expected: active-problem scoped behavior with graceful backend error when Ollama is unavailable
- Actual: returned `LLM backend unavailable.`
- Result: pass

## 6. `lcgrade hint`

- Command: `python3 -m lcgrade.cli hint 2`
- Expected: active-problem scoped hint behavior with graceful backend error when Ollama is unavailable
- Actual: returned `LLM backend unavailable.`
- Result: pass

## 7. `lcgrade reset`

- Command: `python3 -m lcgrade.cli reset`
- Expected: uses active problem by default, clears attempts, review artifacts, chat state, and active problem
- Actual: reset `two-sum`, deleted one attempt, and reported `Cleared active problem: yes`
- Result: pass

## 8. `lcgrade prune`

- Prep: seeded 11 forced attempts for `two-sum`
- Command: `python3 -m lcgrade.cli prune`
- Expected: retain only the newest 10 attempts for the problem
- Actual: reported `Deleted attempts: 1`; post-check showed `10` remaining attempts for `two-sum`
- Result: pass

## Extra Lifecycle Check

- Command: `python3 -m lcgrade.cli review`
- Expected: after solve/reset cleanup, missing active context should fail clearly
- Actual: returned `No active problem. Run \`lcgrade start <slug>\` first.`
- Result: pass
