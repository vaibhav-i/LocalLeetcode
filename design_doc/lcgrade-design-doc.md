# lcgrade — Design Document

## Project Overview

**lcgrade** is a CLI auto-grader for LeetCode-style problems that runs entirely locally using local LLMs. Users write solutions in their editor of choice, run `lcgrade solve <slug>`, and get rich feedback in their terminal — test results, complexity analysis, qualitative code review, and follow-up interview questions — with no internet required after initial setup.

**Target audience:** Software engineers preparing for technical interviews, specifically those targeting MLE/AI engineering roles.

**Portfolio positioning:** This project demonstrates three pillars relevant to MLE/AI engineering roles:

1. **Local inference and optimization** — quantization tradeoffs, memory constraints, benchmarking inference backends on Apple Silicon
2. **LLM evaluation methodology** — hybrid scoring (deterministic + LLM), benchmarking the scorer against ground truth
3. **Production software design** — clean CLI UX, extensible architecture, data modeling, sandbox execution

---

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python | Primary audience (MLE/AI engineers), fastest to ship, rich ecosystem |
| CLI Framework | Typer + Rich | Type-hint-based CLI with colored terminal output, progress bars, tables |
| Local LLM (Beta) | Ollama | One dependency, handles model management, simple HTTP API. Gets beta shipped fast. |
| Local LLM (v1) | MLX | Direct Metal inference on Apple Silicon. Enables benchmarking, quantization analysis, KV cache control — the "depth" story for interviews. |
| Database | SQLite | User state, progress tracking, attempt history, problem index |
| Problem Bank | Markdown + YAML frontmatter | Git-trackable, human-editable, ships as package data |
| Target Hardware | Apple Silicon Mac (M1/M2/M3/M4) | Primary development and optimization target |

### Core Abstractions (all ship in beta)

Anywhere a future change would touch core logic, we build the abstraction now and implement only the beta-required concrete class. Adding new backends, validators, or extensions later is just adding a new class — no refactoring.

#### LLM Backend Abstraction

The LLM is called from multiple places: test case generation, review, extensibility, chat, hints, compaction. All calls go through a single `LLMBackend` interface.

**Design pattern: Template Method.** Only methods that genuinely vary per backend are abstract. Shared logic (JSON parsing, retries) lives as concrete methods on the base class. Subclasses inherit them for free but can override if needed.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json

@dataclass
class LLMResponse:
    text: str
    tokens_used: int
    latency_ms: float

@dataclass
class ModelInfo:
    name: str
    context_window: int     # max tokens
    quantization: str       # e.g., "4-bit", "8-bit", "none"
    backend: str            # e.g., "ollama", "mlx"

class LLMBackend(ABC):
    """Base class for LLM inference backends."""

    # --- Abstract: these vary per backend, subclasses MUST implement ---

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None, 
                 temperature: float = 0.7, max_tokens: int = 2048) -> LLMResponse:
        """Send a prompt and get a text response."""
        ...

    @abstractmethod
    def available(self) -> bool:
        """Check if this backend is ready (model downloaded, server running, etc.)."""
        ...

    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Return model metadata. Used by compaction system for token budgets."""
        ...

    # --- Concrete: shared logic, subclasses inherit for free ---

    def generate_json(self, prompt: str, system_prompt: str | None = None,
                      temperature: float = 0.2, retries: int = 3) -> dict:
        """Calls generate() and handles JSON parsing with retries.
        Override only if backend supports native JSON mode."""
        for attempt in range(retries):
            response = self.generate(prompt, system_prompt, temperature)
            try:
                cleaned = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                if attempt == retries - 1:
                    raise
        raise RuntimeError("JSON generation failed")
```

**`OllamaBackend` (ships in beta):** Implements `generate()`, `available()`, `model_info()`. Gets `generate_json()` for free.

**`MLXBackend` (v1):** Same three abstract methods. Could optionally override `generate_json()` to use MLX's native JSON mode if available.

**`MockBackend` (for testing):** Returns canned responses. Enables unit testing the full pipeline without a real LLM.

CLI flag: `lcgrade --backend ollama` (default) or `lcgrade --backend mlx`

#### Validator Abstraction

Test result comparison goes through a `Validator` interface. Adding new comparison strategies is just a new class.

```python
class Validator(ABC):
    """Base class for test output comparison."""

    @abstractmethod
    def check(self, expected: any, actual: any, input: dict) -> bool:
        """Return True if actual output is correct for the given input."""
        ...

class ExactMatch(Validator):
    def check(self, expected, actual, input):
        return expected == actual

class SetEquality(Validator):
    def check(self, expected, actual, input):
        return set(expected) == set(actual) if isinstance(expected, list) else expected == actual

class FloatTolerance(Validator):
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
    def check(self, expected, actual, input):
        return abs(expected - actual) < self.epsilon

class CustomValidator(Validator):
    """Loads a problem-specific validator.py and calls its check() function."""
    def __init__(self, validator_path: str):
        # Dynamically load the problem's validator module
        ...
    def check(self, expected, actual, input):
        return self._module.check(expected, actual, input)
```

Validators are registered by name. The problem metadata specifies which validator to use via the `validator` field. The test harness looks up the validator by name and calls `check()`.

**Beta:** Ship `ExactMatch` and `SetEquality`. Add `FloatTolerance` and `CustomValidator` as needed.

#### Extension Abstraction (Stage 3)

Extensions are pluggable analysis steps that run after the core LLM review. Each extension receives the full context and produces a section of output.

**Design principle: every data boundary is a typed dataclass, not a dict.** Anything crossing a component boundary should have a defined shape. Dicts are fine inside a component for temporary work, but extension authors (including community contributors) shouldn't have to guess what keys exist.

```python
@dataclass
class TimingData:
    """v1: empirical profiling data from scaling_inputs runs."""
    input_size: int
    runtime_ms: float

@dataclass
class ExtensionContext:
    problem_statement: str
    user_code: str
    verdicts: list[TestVerdict]          # Typed — from Evaluator
    review_output: str                   # Stage 2 LLM review text
    timing_data: list[TimingData] | None # v1: None in beta, populated when empirical profiling ships

@dataclass
class ExtensionResult:
    title: str                # Display name in terminal output
    content: str              # The extension's output text

class Extension(ABC):
    """Base class for Stage 3 extensibility plugins."""

    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this extension (used in --extend flag)."""
        ...

    @abstractmethod
    def run(self, ctx: ExtensionContext, llm: LLMBackend) -> ExtensionResult:
        """Execute this extension and return results."""
        ...

class InterviewQuestions(Extension):
    def name(self): return "interview"
    def run(self, ctx, llm):
        prompt = f"Given this problem and solution, generate 2-3 follow-up interview questions..."
        response = llm.generate(prompt)
        return ExtensionResult(title="Interview Follow-ups", content=response.text)

class OptimizationPrompt(Extension):
    def name(self): return "optimize"
    def run(self, ctx, llm):
        prompt = f"Analyze this solution for potential optimizations..."
        response = llm.generate(prompt)
        return ExtensionResult(title="Optimization Suggestions", content=response.text)
```

Extensions are discovered by scanning a registry. Users select which to run via flag: `lcgrade solve two-sum --extend interview,optimize` or config default.

**Beta:** Ship `InterviewQuestions` and `OptimizationPrompt`. Community can add their own by dropping a Python file into `~/.lcgrade/extensions/`. (Note: community extensions run with full permissions in the lcgrade process — this is a known risk. Sandboxing for community extensions is a v1 consideration.)

---

## Roadmap

| Phase | Scope |
|-------|-------|
| **Beta** | All core abstractions (Runner, LLMBackend, Validator, Extension) with Python/Ollama/ExactMatch+SetEquality/InterviewQuestions+OptimizationPrompt implementations. Core eval pipeline end-to-end, Blind 75 problem bank from cojudge, subprocess sandbox. |
| **v1** | Add MLXBackend + inference benchmarks, eval accuracy benchmarks, empirical complexity profiling (`scaling_inputs` generators + timing framework), additional Runners (Java, C++), additional Validators (FloatTolerance, Custom), Docker sandbox option, community extensions. Optional cloud-agent backends via user-provided Claude/OpenAI/Gemini API keys are explicitly non-core convenience modes and only follow the local-backend work. |
| **Polish** | `lcgrade setup` onboarding wizard, interactive TUI browser (textual), blog posts, README with demo GIF |

---

## Architecture Overview

### Three-Stage Evaluation Pipeline

When the user runs `lcgrade solve <slug>`, the pipeline executes in order. Each stage is a natural stopping point — the user can be satisfied with just test results, or opt into deeper LLM analysis.

#### Cache Check (run first)

Before doing any work, compute three hashes: SHA-256 of the user's solution file (`code_hash`), SHA-256 of the problem's `tests.json` (`tests_hash`), and the `test_mode` flag. Compare against the most recent attempt in SQLite. The cache key is `(code_hash, test_mode, tests_hash)` — all three must match for a cache hit.

**If cache hit:**
- Skip all stages, show cached results immediately
- Display: "Code unchanged since last attempt (2 min ago). Showing cached results."
- If a review exists, show it. If not, offer: "Run `lcgrade review two-sum` to add LLM analysis."
- User can bypass with `lcgrade solve two-sum --force` to re-run everything

**If cache miss:**
- Any of the three components changed → run the full pipeline from preflight onwards
- Common triggers: code edited, different `--tests` flag, problem bank updated after lcgrade upgrade

**Edge case — code unchanged but user wants new extensions:**
`lcgrade solve two-sum --extend optimize` with unchanged code should skip Stages 1-2 and only run the new extension. The `--extend` flag doesn't invalidate the test/review cache.

#### Preflight Checks (run before Stage 1)

Fast, deterministic checks that run before any test execution:

- **Syntax validation** — parse the source with Python's `ast.parse()`. If parsing fails, stop immediately and show the syntax error with line/column context.
- **Function signature validation** — check that the expected function exists and has the correct arity. Parameter names and type hints are advisory only in beta, since valid interview-style Python often omits hints or uses equivalent forms.
- **Optional lint notes (`ruff`)** — non-blocking style/code-quality feedback. If available, include it in the review output or a separate "Lint Notes" section, but do not fail execution because of stylistic issues.

#### Stage 1: Test Case Execution

Two sources of test cases, selectable via `--tests` flag:

- **`--tests bundled`** (no LLM required): Pre-shipped test cases with known input/output pairs. Works immediately after `pip install lcgrade` with zero setup. Source: converted from cojudge's Blind 75 problem bank.
- **`--tests llm`**: LLM generates additional edge-case test cases as structured JSON (input, expected output, rationale). Targets cases the provided tests might miss — empty inputs, duplicates, negative numbers, boundary values.
- **`--tests both`** (default): Runs both bundled and LLM-generated test cases.

User code is executed against test cases via the sandbox (see Code Execution Sandbox section). Results are collected as pass/fail per test case with captured output or error messages.

**LLM-generated test case trust hierarchy:**

1. **Verified** — expected output generated by a human-written reference solution. Tag: `"source": "verified"`
2. **LLM-generated, cross-checked** — LLM generates expected output, confirmed against reference solution. Tag: `"source": "llm_verified"`
3. **LLM-generated, unverified** — no reference solution available, LLM is sole source. Tag: `"source": "llm"`. These are weighted differently in scoring and flagged to the user.

**Some users may stop here.** Test results alone are valuable feedback — they tell you if your code works. Stages 2 and 3 are optional deeper analysis.

#### Stage 2: LLM Review

The LLM receives the code, test results, and optional lint output. It produces:

- **Time and space complexity analysis** inferred from code structure (LLM reads the code and identifies patterns — nested loops, hash map usage, recursion with memoization, etc.). No empirical timing needed; the LLM is good at this from code alone.
- **Correctness assessment** informed by actual test results
- **Code quality feedback** beyond what ruff catches
- **Edge cases missed** — what the user didn't handle

**v1 enhancement: empirical timing.** Run user code on synthetic inputs of varying sizes (n=10, 100, 1000, 10000) using `scaling_inputs` generators, measure wall-clock time, and pass the scaling curve to the LLM alongside the code. This gives the LLM both the *what* (your code scales quadratically) and it explains the *why* (nested loop on line 5). Deferred to v1 because it requires problem-specific input generators that don't exist yet.

This is a fixed, opinionated analysis — the same review criteria for every problem.

#### Stage 3: LLM Extensibility (Optional)

A separate, pluggable layer for open-ended analysis that goes beyond the core review. This is opt-in and customizable:

- **Interview-style follow-up questions** — "How would you modify this if the array was sorted?" or "Can you do this in O(1) space?"
- **Optimization prompts** — "Your solution works, but can you reduce the space complexity?"
- **Custom review criteria** — users or the community can define their own extensions (e.g., "review for production readiness," "review for system design implications," "review for concurrency safety")

Extensions are defined as prompt templates that receive the code, test results, and Stage 2 review. The user controls which extensions run via config or flags (e.g., `lcgrade solve two-sum --extend interview,optimize`).

This separation matters: Stage 2 always runs the same way for consistent scoring. Stage 3 is a playground for deeper exploration that the user opts into when they're ready.

#### Pipeline Error Recovery

**Design principle: save results as each stage completes, not after the whole pipeline finishes.** This prevents losing work when a later stage fails.

The flow:

1. **Preflight** passes → proceed
2. **Stage 1** completes → immediately write `attempts` row to SQLite (test results, code snapshot, timing). User can see results even if everything after this crashes.
3. **Stage 2** completes → write `reviews` row linked to the attempt. If Stage 2 fails (Ollama crashes, model OOMs), the attempt is already saved. User sees: "Test results saved. LLM review failed — run `lcgrade review two-sum` to retry."
4. **Stage 3** completes → insert rows into `extension_results` table (one per extension). If Stage 3 fails, user still has test results + core review. Extensions can be re-run independently without affecting prior results.

This also enables the `lcgrade review <slug>` command — re-run Stages 2/3 against the most recent saved attempt without re-executing tests. Useful for retrying after a crash, reviewing with a different model, or running different extensions.

#### Graceful Degradation When Ollama Is Unavailable

If the LLM backend is not available (`backend.available()` returns False):

- **`--tests bundled`**: Works fully. No LLM needed.
- **Core product remains usable**: `start`, `describe`, `solve` with bundled tests, `history`, `stats`, and `random` should all work offline without any LLM backend configured.
- **`--tests llm` or `--tests both`**: Falls back to provided-tests-only with a warning that local AI-generated tests were skipped and bundled tests still ran.
- **Stage 2/3**: Skipped with a message that this feature needs a local LLM backend, while core lcgrade solving still works without one.
- **`lcgrade chat` / `lcgrade hint`**: Clear error: "Chat requires a running LLM backend. Run `ollama serve` to start."

The tool should **always** do something useful rather than crash. Test results without LLM review are still valuable.

---

## Problem Bank

### Source Data

Initial problem bank is converted from [cojudge/cojudge](https://github.com/cojudge/cojudge), which contains 60+ Blind 75 problems with:

- `statement.md` — problem descriptions
- `metadata.json` — difficulty, input structure, starter code
- `official-tests.json` — test case inputs only (no expected outputs)
- `Marker.java` — Java reference solution + custom judge

### Conversion Process

cojudge stores only test inputs; their Java Marker computes expected outputs at runtime. A one-time conversion script:

1. Takes cojudge's `official-tests.json` inputs per problem
2. Runs Python reference solutions (written by the developer) against those inputs
3. Outputs complete `tests.json` files with input + expected output pairs
4. These ship with the lcgrade package as the "verified" test bank

### lcgrade Problem Format

Each problem lives in `~/.lcgrade/problems/<slug>/`:

```
~/.lcgrade/problems/two-sum/
├── statement.md          # Problem description with YAML frontmatter
├── tests.json            # Bundled test cases (input + expected output + validator type)
├── starter.py            # Function stub
├── validator.py          # (Optional) Custom validator for non-deterministic problems
└── solutions/
    └── reference.py      # (Optional) Reference solution for generating new test answers
```

### YAML Frontmatter Schema (in `statement.md`)

```yaml
---
schema_version: 1
title: Two Sum
slug: two-sum
difficulty: easy
tags: [array, hash-table]
category: blind75
function_name: two_sum
params:
  - name: nums
    type: List[int]
  - name: target
    type: int
return_type: List[int]
validator: set_equality
# v1: scaling_inputs for empirical complexity profiling
# scaling_inputs:
#   sizes: [10, 100, 1000, 10000]
#   generator: random_int_array
---
```

Key fields:

- `schema_version` — format version for migration support. If the YAML schema changes between lcgrade releases, the indexer detects old versions and either auto-migrates or prompts the user to update. Start at 1, increment on breaking changes.
- `function_name`, `params`, `return_type` — used by the test harness to generate the runner script
- `validator` — which comparison method to use (see Validators section)
- `scaling_inputs` — (v1) tells the empirical profiler how to generate varying-size inputs for complexity benchmarking

### tests.json Schema

```json
[
  {
    "input": {"nums": [2, 7, 11, 15], "target": 9},
    "expected": [0, 1],
    "validator": "set_equality",
    "source": "verified"
  }
]
```

### Validators

Most Blind 75 problems use exact match, but some accept multiple valid outputs:

| Validator | Use Case |
|-----------|----------|
| `exact_match` | Default. `output == expected` |
| `set_equality` | Order doesn't matter (e.g., Two Sum) |
| `float_tolerance` | Floating point comparison with epsilon |
| `any_valid` | Calls problem-specific `validator.py` that takes (input, output) → bool |

For beta, implement `exact_match` and `set_equality` only. Add more as needed.

For LLM-generated test cases on non-deterministic problems: skip expected output, let the LLM validate the output in Stage 3 qualitative review instead.

---

## Data Model

### Storage Architecture

- **Markdown files** = source of truth for problems (git-trackable, human-editable)
- **SQLite** = index for fast queries + all user state (attempts, scores, chat history)

Analogy: like a music player indexing MP3 files. The files are canonical; the database makes searching fast.

### Indexing Flow

1. On `lcgrade init` or first run, copy bundled problem bank to `~/.lcgrade/problems/`
2. Scan all problem folders, parse YAML frontmatter, populate `problems` table
3. On subsequent runs, check for new/modified problems (compare file mtimes or content hash), re-index only changed files
4. Adding a new problem = dropping a folder into the problems directory → auto-detected on next run

### SQLite Schema

#### `metadata` table (application state)

| Column | Type | Description |
|--------|------|-------------|
| key | TEXT PK | Setting name |
| value | TEXT | Setting value |

Stores: `db_version` (current schema version, for migrations), `active_slug` (currently active problem, set by `lcgrade start`).

**SQLite migration strategy:** On startup, lcgrade reads `db_version` from the metadata table. If it's lower than the current version baked into the code, it runs migration scripts sequentially (e.g., v1→v2, v2→v3). Migrations are simple `ALTER TABLE` / `CREATE TABLE` statements — no ORM, no Alembic. If the metadata table doesn't exist at all (fresh install), create everything from scratch.

```python
MIGRATIONS = {
    1: "CREATE TABLE metadata ...; CREATE TABLE problems ...; ...",
    2: "ALTER TABLE attempts ADD COLUMN new_field TEXT;",
    # each key is a version, value is the SQL to migrate FROM the previous version
}

def migrate(db):
    current = get_db_version(db)  # 0 if metadata table missing
    for version in sorted(MIGRATIONS):
        if version > current:
            db.executescript(MIGRATIONS[version])
            set_db_version(db, version)
```

#### `problems` table (indexed from files + durable user progress)

| Column | Type | Description |
|--------|------|-------------|
| slug | TEXT PK | Problem identifier |
| title | TEXT | Display name |
| difficulty | TEXT | easy / medium / hard |
| tags | TEXT | JSON array of tags |
| category | TEXT | e.g., blind75 |
| function_name | TEXT | Expected function name |
| validator | TEXT | Default validator type |
| schema_version | INTEGER | YAML schema version from frontmatter (for detecting old-format problems) |
| file_path | TEXT | Path to problem directory |
| auto_solved | INTEGER | 0/1 flag: has the user ever passed the provided tests? |
| auto_solved_at | DATETIME NULL | When the problem was first auto-marked solved from provided tests |
| manual_solved | INTEGER | 0/1 user override flag |
| manual_solved_at | DATETIME NULL | When the user manually marked the problem solved |
| review_generated | INTEGER | 0/1 flag: has Stage 2 ever completed successfully? |
| review_generated_at | DATETIME NULL | When the first review was generated |
| review_acknowledged | INTEGER | 0/1 flag: has the user acknowledged the review? |
| review_acknowledged_at | DATETIME NULL | When the user marked the review as completed/absorbed |
| followup_completed | INTEGER | 0/1 flag: has the user marked follow-up work complete? |
| followup_completed_at | DATETIME NULL | When the user marked follow-up complete |

#### `attempts` table (execution facts only)

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| slug | TEXT FK | Problem identifier |
| timestamp | DATETIME | When the attempt was made |
| test_mode | TEXT | bundled / llm / both |
| code_hash | TEXT | SHA-256 of the solution file (cache key component) |
| tests_hash | TEXT | SHA-256 of the provided-tests `tests.json` file (cache key component) |
| bundled_passed | INTEGER | Number of provided tests passed |
| bundled_total | INTEGER | Total provided tests |
| llm_passed | INTEGER | Number of LLM tests passed |
| llm_total | INTEGER | Total LLM tests |
| runtime_ms | REAL | Execution time |
| status | TEXT | pass / fail / TLE / MLE / error |
| code_snapshot | TEXT | Full source code at time of attempt |

**Cache key:** `(code_hash, test_mode, tests_hash)`. All three must match the most recent attempt for a cache hit.

**Cache invalidation — the cache is just the existing attempts/reviews rows, not separate storage:**

| Trigger | Why | What happens |
|---------|-----|--------------|
| Code changes | Different `code_hash` | Cache miss → full pipeline runs |
| Different `--tests` mode | Different `test_mode` | Cache miss → full pipeline runs |
| Problem bank updated (e.g., lcgrade upgrade) | Different `tests_hash` | Cache miss → full pipeline runs |
| `lcgrade solve --force` | Explicit bypass | Ignores cache → full pipeline runs |
| `lcgrade reset <slug>` | Wipes workspace | Attempts deleted → no cache exists |
| 11th attempt on same problem | Retention policy | Oldest attempt deleted, no impact on current cache |

**Retention policy:** Maximum 10 snapshots per problem. On the 11th attempt, the oldest is deleted (along with its linked review, if any). Users can run `lcgrade prune` for manual cleanup. Since the cache is just these rows, retention cleanup *is* cache cleanup — no separate cache layer to manage.

**Design principle: normalization.** This table stores only execution facts — what happened when the code ran. No LLM-produced analysis. An attempt can exist without a review (user only ran tests). This avoids NULL columns for users who skip Stages 2/3, and cleanly separates "I ran my code" from "I got it reviewed."

#### Problem completion model

The product tracks **three separate progress checks**, not one blended score:

1. **Solved** — durable correctness progress for the problem
2. **Review** — both the system event ("a review exists") and the user event ("I have acknowledged it")
3. **Follow-up** — a user progress marker for whether they feel they have taken the solution to a standard they are happy with

These checks are stored on the `problems` table as **sticky milestones**, not derived from only the latest attempt.

**Solved check**

- `auto_solved` becomes true the first time the user passes the **provided tests**
- `manual_solved` becomes true when the user explicitly marks the problem solved
- Effective solved state is: `auto_solved OR manual_solved`

**Review check**

- `review_generated` becomes true the first time Stage 2 completes successfully
- `review_acknowledged` becomes true when the user indicates they have absorbed the review feedback
- The UI may show both states separately, or use acknowledgment as the top-level "review complete" check while still preserving generated state

**Follow-up check**

- `followup_completed` is a user-controlled progress marker only
- Running Stage 3 extensions does **not** automatically mark follow-up complete
- `extension_results` rows are evidence/history of follow-up activity, not the canonical follow-up completion state

Implications for UX:

- `lcgrade solve` should show these as separate checkmarks/statuses
- history and stats should distinguish solved, review generated, review acknowledged, and follow-up completed
- if a future numeric score is added, it should represent analysis depth or practice completeness, not correctness
- once any of these fields becomes true, it stays true until explicitly cleared by the user

#### `reviews` table (LLM analysis, linked to attempt)

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| attempt_id | INTEGER FK | Links to `attempts.id` (one-to-one, optional) |
| timestamp | DATETIME | When the review was generated |
| complexity_time | TEXT | e.g., "O(n)" |
| complexity_space | TEXT | e.g., "O(n)" |
| review_text | TEXT | Full Stage 2 review output |

This separation enables error recovery: if Stage 2 fails (Ollama crashes), the attempt is already saved. User can retry the review with `lcgrade review <slug>` without re-running tests. Durable review progress lives on `problems` (`review_generated`, `review_acknowledged`); the `reviews` table remains an event/history table for the actual generated analysis.

Querying is also cleaner:
- "Show all attempts" → `SELECT * FROM attempts`
- "Show reviewed attempts" → `JOIN reviews ON attempts.id = reviews.attempt_id`
- "Show unreviewed attempts" → `LEFT JOIN reviews ... WHERE reviews.id IS NULL`

#### `extension_results` table (Stage 3 output, linked to review)

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| review_id | INTEGER FK | Links to `reviews.id` (one-to-many) |
| extension_name | TEXT | e.g., "interview", "optimize" |
| output_text | TEXT | The extension's output |
| timestamp | DATETIME | When this extension was run |

**Why a separate table instead of a JSON blob on `reviews`:** Same normalization principle as attempts/reviews. Each extension run is its own row, which means: no read-modify-write to add an extension later (just INSERT), each extension has its own timestamp, and querying is clean SQL instead of JSON parsing.

Example queries:
- "Show all interview follow-ups across all problems" → `SELECT * FROM extension_results WHERE extension_name = 'interview'`
- "Which problems have I run extensions on?" → `SELECT DISTINCT slug FROM extension_results JOIN reviews JOIN attempts`
- "Run only the optimize extension" → check if row exists for this review + extension name, skip if already run

#### `chat_messages` table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| slug | TEXT FK | Problem identifier |
| session_id | TEXT | Groups messages into sessions |
| role | TEXT | user / assistant / summary |
| message | TEXT | Message content |
| hint_tier | INTEGER NULL | If this was a hint request, which tier (1/2/3). NULL for normal chat. |
| timestamp | DATETIME | When message was sent |

**Note:** The separate `hints` table has been merged into `chat_messages`. Since `lcgrade hint` is just `lcgrade chat` with a canned prompt, hints are chat messages tagged with a `hint_tier`. This avoids a redundant table and keeps all conversational history in one place.

**Chat history lifecycle:** Chat history exists only while a problem is active. Wiped when the user solves the problem or explicitly closes/resets it. No long-term chat accumulation.

---

## Code Execution Sandbox

### Beta: subprocess with guardrails

The user's code runs in a separate Python process, never in the main lcgrade process.

**Protection layers:**

| Threat | Protection | Mechanism |
|--------|------------|-----------|
| Infinite loop | Timeout | `subprocess.run(timeout=10)` → `TimeoutExpired` → report TLE |
| Memory bomb | Memory monitoring | `psutil` monitors subprocess memory from parent, kills if over threshold (more reliable than `RLIMIT_AS` on Apple Silicon) |
| Uncaught exception | Try/catch in harness | Test harness wraps each test in try/except → report Runtime Error |
| Syntax error | Import failure | subprocess fails on import → report Compilation Error |

### Runner Abstraction

The execution layer uses a base class so adding new languages is just adding a new class. **The abstraction ships in beta with only `PythonRunner` implemented.** Future languages (Java, C++, Go, JS) just subclass `Runner`.

**Design principle: Single Responsibility.** The Runner's job is *execution* — run the code and report what happened. It does NOT judge correctness. A separate `Evaluator` handles that (see below). This means runners never need to know about validators, and validators never need to know about languages.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json

# --- Runner data: facts about what happened ---

@dataclass
class TestOutput:
    output: any             # Raw return value (serialized), None if error
    error: str | None       # Exception message if crashed
    runtime_ms: float       # Wall-clock time for this test case
    status: str             # "ok" | "error" | "TLE" | "MLE"
    captured_stdout: str = ""
    captured_stderr: str = ""

@dataclass
class RawOutput:
    stdout: str             # Raw stdout from subprocess
    stderr: str             # Raw stderr from subprocess
    exit_code: int          # Process exit code
    timed_out: bool         # True if subprocess hit timeout
    memory_exceeded: bool   # True if subprocess hit memory limit

@dataclass
class RunResult:
    outputs: list[TestOutput]   # One per test case
    compile_error: str | None = None
    harness_error: str | None = None
    status: str = "ok"          # "ok" | "compile_error" | "harness_error" | "TLE" | "MLE"

class Runner(ABC):
    """Base class for language-specific code execution.
    Responsible for: running code, enforcing resource limits, capturing output.
    NOT responsible for: judging correctness."""

    def __init__(self, timeout: int = 10, memory_limit_mb: int = 512):
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb

    # --- Abstract: vary per language, subclasses MUST implement ---

    @abstractmethod
    def validate(self, source_path: str, function_name: str, params: list[dict]) -> str | None:
        """Preflight check: parse source, verify expected function exists,
        and verify required arity. Returns None if valid, error message if not."""
        ...

    @abstractmethod
    def generate_harness(self, source_path: str, function_name: str, test_cases: list[dict]) -> str:
        """Generate a test harness script that imports user code,
        runs test cases, and writes structured results to a temp file.
        The harness captures output but does NOT compare against expected."""
        ...

    @abstractmethod
    def compile(self, source_path: str, work_dir: str) -> str | None:
        """Compile if needed (no-op for interpreted languages).
        Returns None if success, error message if compilation fails."""
        ...

    @abstractmethod
    def execute(self, harness_path: str, work_dir: str) -> RawOutput:
        """Run the harness in a subprocess with timeout and memory limits.
        Returns raw subprocess output — does NOT parse result payloads.
        Override only to change how the subprocess is launched."""
        ...

    # --- Concrete: shared logic, subclasses inherit for free ---

    def parse_output(self, raw: RawOutput) -> RunResult:
        """Parse structured harness output into RunResult.
        Handles TLE, MLE, missing/incomplete harness output, and malformed
        result files gracefully."""
        if raw.timed_out:
            return RunResult(outputs=[], status="TLE")
        if raw.memory_exceeded:
            return RunResult(outputs=[], status="MLE")
        try:
            results = self.load_results(raw)
            outputs = [TestOutput(**r) for r in results]
            return RunResult(outputs=outputs)
        except FileNotFoundError:
            return RunResult(
                outputs=[],
                harness_error="Harness did not produce a results file.",
                status="harness_error"
            )
        except json.JSONDecodeError:
            return RunResult(
                outputs=[],
                harness_error="Harness results file was not valid JSON.",
                status="harness_error"
            )

    def load_results(self, raw: RawOutput) -> list[dict]:
        """Load structured results emitted by the harness.
        Beta design: the harness writes to a temp JSON file rather than stdout."""
        ...

    def run(self, source_path: str, function_name: str, params: list[dict], test_cases: list[dict], work_dir: str) -> RunResult:
        """Full execution pipeline: validate → compile → generate harness → execute → parse.
        Concrete method — subclasses inherit the orchestration."""
        # Preflight: signature check
        sig_error = self.validate(source_path, function_name, params)
        if sig_error:
            return RunResult(outputs=[], compile_error=sig_error, status="compile_error")

        # Compile (no-op for Python)
        compile_error = self.compile(source_path, work_dir)
        if compile_error:
            return RunResult(outputs=[], compile_error=compile_error, status="compile_error")

        # Generate harness, execute, and parse output
        harness_path = self.generate_harness(source_path, function_name, test_cases)
        raw = self.execute(harness_path, work_dir)
        return self.parse_output(raw)
```

**`PythonRunner` (ships in beta):**

- `validate()` — uses Python's `ast` module to parse the file, fail fast on syntax errors, verify the expected function exists, and verify required arity. Parameter names and type hints are advisory-only in beta.
- `generate_harness()` — writes a Python script that imports the user's function, iterates over test cases, captures per-test output/errors/stdout/stderr, and writes structured results to a temp JSON file. Does NOT compare against expected values.
- `compile()` — no-op, returns None (Python is interpreted)
- `execute()` — runs via `subprocess.run()` with timeout, monitors memory via `psutil`. Returns `RawOutput` with stdout/stderr/exit code.
- `parse_output()` — inherited from base class. Loads the harness results file into `RunResult`. No override needed.

**Future runners (post-beta):**

- `JavaRunner` — `compile()` runs `javac`, `execute()` runs `java`, `parse_output()` inherited
- `CppRunner` — `compile()` runs `g++`, `execute()` runs the binary, `parse_output()` inherited
- `GoRunner` — `compile()` runs `go build`, `execute()` runs the binary, `parse_output()` inherited
- `DockerRunner` — wraps any language runner inside a Docker container for stricter isolation

To add a new language, implement the four abstract methods. `run()` orchestration and `parse_output()` are inherited for free.

### Evaluator (judges correctness)

The Evaluator takes the Runner's raw outputs and the test cases, applies the appropriate Validator per test, and produces verdicts. This is where "is the answer correct?" lives.

```python
@dataclass
class TestVerdict:
    passed: bool
    expected: any
    actual: any = None
    error: str | None = None
    runtime_ms: float = 0.0
    status: str = "ok"          # "ok" | "error" | "TLE" | "MLE"
    message: str | None = None  # Human-readable context (e.g., "expected [0,1] but got [1,0] — order doesn't matter, did you mean to use set_equality?")

class Evaluator:
    """Judges correctness by comparing Runner outputs against expected values.
    Uses Validators for comparison logic."""

    def __init__(self, validators: dict[str, Validator]):
        self.validators = validators

    def evaluate(self, run_result: RunResult, test_cases: list[dict],
                 default_validator: str = "exact_match") -> list[TestVerdict]:
        if run_result.status in {"TLE", "MLE", "compile_error", "harness_error"}:
            return [
                TestVerdict(
                    passed=False,
                    expected=case.get("expected"),
                    actual=None,
                    error=run_result.compile_error or run_result.harness_error,
                    status=run_result.status
                )
                for case in test_cases
            ]

        verdicts = []
        for output, case in zip(run_result.outputs, test_cases):
            # If the harness reported an error, propagate it
            if output.status != "ok":
                verdicts.append(TestVerdict(
                    passed=False, expected=case.get("expected"),
                    actual=None, error=output.error,
                    runtime_ms=output.runtime_ms, status=output.status
                ))
                continue

            # Look up the validator and judge
            validator_name = case.get("validator", default_validator)
            validator = self.validators[validator_name]
            passed = validator.check(case["expected"], output.output, case["input"])
            verdicts.append(TestVerdict(
                passed=passed, expected=case["expected"],
                actual=output.output, runtime_ms=output.runtime_ms
            ))
        return verdicts
```

This separation means:
- Adding a new **language** = new Runner subclass. Evaluator untouched.
- Adding a new **validator** = new Validator subclass, register it. Runner untouched.
- Better **error reporting** = the Evaluator in the parent process can produce rich diffs (e.g., "expected [0,1] but got [1,0] — did you mean to use set_equality?")
- Every language harness follows the same simple contract: "return raw output as JSON."

**Future optimization:** The Evaluator iterates sequentially. For beta this is fine (validators are microsecond equality checks). If validator count or complexity grows (CustomValidator loading slow modules, large LLM-generated test suites), consider parallelizing with `concurrent.futures.ThreadPoolExecutor`. The `Validator.check()` interface is already stateless, so parallelization is safe with no refactoring needed.

### Test Harness (Python Example)

The `PythonRunner.generate_harness()` produces something like:

```python
# Auto-generated by lcgrade — captures raw output only, does NOT judge correctness
import contextlib
import io
import json
import sys
import time

# Test cases loaded from temp file (avoids arg length limits for large test suites)
with open(sys.argv[1]) as f:
    test_cases = json.load(f)

results_path = sys.argv[2]

from solution import two_sum  # import after loading test cases to catch SyntaxError

results = []
for case in test_cases:
    start = time.perf_counter()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            output = two_sum(**case["input"])
        elapsed = (time.perf_counter() - start) * 1000
        results.append({
            "output": output,
            "runtime_ms": elapsed,
            "status": "ok",
            "captured_stdout": stdout_buffer.getvalue(),
            "captured_stderr": stderr_buffer.getvalue(),
        })
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        results.append({
            "error": f"{type(e).__name__}: {e}",
            "runtime_ms": elapsed,
            "status": "error",
            "captured_stdout": stdout_buffer.getvalue(),
            "captured_stderr": stderr_buffer.getvalue(),
        })

with open(results_path, "w") as f:
    json.dump(results, f)
```

Note: test cases are loaded from a temp file (not inlined in `sys.argv`) to avoid OS argument length limits with large test suites. The harness writes structured results to a separate temp JSON file rather than stdout so user `print()` calls do not corrupt the machine-readable result channel.

### Execution Flow

```
Runner (subprocess)              Evaluator (parent process)
─────────────────                ─────────────────────────
1. validate() — signature check
2. compile() — no-op for Python
3. generate_harness() — write script
4. execute() — run with TLE/MLE
   └→ returns RunResult          5. evaluate(run_result, test_cases)
      (raw outputs per test)        └→ returns list[TestVerdict]
                                       (pass/fail per test)
                                 6. Feed verdicts into Stage 2/3
```

1. User runs `lcgrade solve two-sum`
2. lcgrade detects language from file extension → selects appropriate `Runner`
3. `Runner.run()` orchestrates: validate → compile → generate harness → execute
4. Subprocess runs with timeout + memory monitoring via `psutil`
5. `Evaluator.evaluate()` compares raw outputs against expected values using validators
6. Feeds verdicts into Stages 2 and 3

### v1: Docker sandbox option

A future option for stricter isolation. `DockerRunner` wraps any language runner inside a container. Same `Runner` interface, different execution backend. Not needed for beta since users are running their own code on their own machine.

---

## Pipeline Orchestrator

### The Problem

Runner, Evaluator, LLMBackend, and Extensions are all well-defined components — but somebody needs to call them in the right order, handle errors between stages, and manage save-as-you-go. Without an explicit orchestrator, the CLI command handler becomes a 200-line procedural function that's hard to test and impossible to reuse across commands.

### Three-Layer Architecture

Dependencies only point downward. No layer reaches past the one below it.

```
CLI (thin)          → parses args, calls Pipeline, formats output with Rich
Pipeline (smart)    → owns ordering, error recovery, caching, save-as-you-go
Components (focused) → Runner, Evaluator, LLMBackend, Extensions — each does one job
```

- **Components** are specialists. Runner executes code. Evaluator compares outputs. Neither knows the other exists.
- **Pipeline** is the coordinator. It knows about all components, knows the order, knows what to do when one fails. But it doesn't do actual work — it delegates.
- **CLI** is the presentation layer. It calls `pipeline.solve()` and renders the result. It doesn't know about Runners or Evaluators directly.

### Pipeline Class

```python
@dataclass
class PipelineResult:
    """Complete output of a pipeline run."""
    cache_hit: bool
    preflight_errors: list[str]
    run_result: RunResult | None
    verdicts: list[TestVerdict] | None
    review_text: str | None
    review_generated: bool
    review_acknowledged: bool
    followup_completed: bool
    extensions: list[ExtensionResult]
    error: str | None = None          # Pipeline-level error message

class Pipeline:
    """Orchestrates the full solve flow.
    Owns: ordering, error recovery, save-as-you-go, cache logic.
    Does NOT own: execution, evaluation, LLM calls, output formatting."""

    def __init__(self, runner: Runner, evaluator: Evaluator,
                 llm: LLMBackend, db: Database):
        self.runner = runner
        self.evaluator = evaluator
        self.llm = llm
        self.db = db

    def solve(self, slug: str, source_path: str,
              test_mode: str = "both", force: bool = False,
              extensions: list[Extension] | None = None) -> PipelineResult:
        """Full pipeline: cache → preflight → Stage 1 → Stage 2 → Stage 3.
        Saves results to DB after each stage completes."""
        ...

    def review(self, slug: str,
               extensions: list[Extension] | None = None) -> PipelineResult:
        """Re-run Stages 2/3 against most recent saved attempt.
        Loads attempt from DB, skips Runner entirely."""
        ...
```

### CLI Commands Are Thin Wrappers

```python
@app.command()
def solve(slug: str | None = None, tests: str = "both", force: bool = False):
    slug = slug or get_active_slug()   # fall back to active problem
    pipeline = build_pipeline()         # dependency injection
    result = pipeline.solve(slug, get_source_path(slug), tests, force)
    render_solve_output(result)         # Rich formatting

@app.command()
def review(slug: str | None = None):
    slug = slug or get_active_slug()
    pipeline = build_pipeline()
    result = pipeline.review(slug)
    render_solve_output(result)
```

### Why This Matters

- **Testable**: inject `MockBackend` + `PythonRunner` with fixtures + in-memory SQLite → test full pipeline without a terminal, without Ollama, without real code execution
- **Reusable**: `lcgrade solve` and `lcgrade review` share the same Pipeline, just different entry points. No code duplication.
- **Maintainable**: adding a new stage or changing error recovery logic happens in one place (Pipeline), not across multiple CLI commands

### Active Problem

When the user runs `lcgrade start <slug>`, that problem becomes the **active problem**. All subsequent commands default to it — no need to type the slug every time.

`lcgrade start two-sum` sets the active problem. After that:
- `lcgrade solve` = `lcgrade solve two-sum`
- `lcgrade chat "I'm stuck"` = `lcgrade chat two-sum "I'm stuck"`
- `lcgrade hint` = `lcgrade hint two-sum`
- `lcgrade history` = `lcgrade history two-sum`

The slug is always optional — provide it to explicitly target a different problem. `lcgrade reset` or solving a problem successfully clears the active problem.

**Implementation:** Store `active_slug` in the SQLite metadata table. `lcgrade start` sets it, `lcgrade reset` and successful `lcgrade solve` clear it. Every command checks for it as the default when no slug is provided.

### Command Reference

`<slug>` is optional on most commands — defaults to the active problem set by `lcgrade start`. Provide it explicitly to target a different problem.

| Command | Description |
|---------|-------------|
| `lcgrade init` | First-time setup: copy problem bank, create `~/.lcgrade/`, build SQLite index, detect editor. Local LLM setup is optional and should not block the core offline workflow. |
| `lcgrade list` | Show problem categories with solve counts |
| `lcgrade list <category>` | Drill into a category, show individual problems with status |
| `lcgrade list --all` | Flat list of all problems |
| `lcgrade list --difficulty <d> --tag <t>` | Filtered listing |
| `lcgrade start <slug>` | Generate solution file with embedded docstring, open in editor, set as active problem |
| `lcgrade solve [slug]` | Run the full eval pipeline, print results (uses cache if code unchanged) |
| `lcgrade solve [slug] --force` | Bypass cache, re-run entire pipeline |
| `lcgrade solve [slug] --tests bundled\|llm\|both` | Choose test source |
| `lcgrade solve [slug] --extend interview,optimize` | Run with specific Stage 3 extensions |
| `lcgrade review [slug]` | Re-run Stages 2/3 against most recent saved attempt (no re-execution of tests) |
| `lcgrade hint [slug]` | Shortcut for `lcgrade chat [slug] "give me a hint"` |
| `lcgrade chat [slug] "<prompt>"` | Single-shot LLM question about the problem (beta) |
| `lcgrade describe [slug]` | Print problem statement to terminal |
| `lcgrade stats` | Progress dashboard |
| `lcgrade history [slug]` | Past attempts on a specific problem |
| `lcgrade reset [slug]` | Wipe workspace file, clear active problem, start fresh |
| `lcgrade random` | Start a random unsolved problem (sets it as active) |
| `lcgrade random --difficulty <d> --tag <t>` | Filtered random |
| `lcgrade prune` | Clean up old snapshots beyond the 10-attempt cap |

### Solution File Generation

When the user runs `lcgrade start two-sum`, lcgrade generates:

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    """
    Two Sum
    -------
    Difficulty: Easy | Tags: array, hash-table

    Given an array of integers nums and an integer target, return indices
    of the two numbers such that they add up to target.

    You may assume that each input would have exactly one solution, and
    you may not use the same element twice.

    You can return the answer in any order.

    Examples:
        Input: nums = [2,7,11,15], target = 9
        Output: [0,1]
        Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

        Input: nums = [3,2,4], target = 6
        Output: [1,2]

    Constraints:
        - 2 <= nums.length <= 10^4
        - -10^9 <= nums[i] <= 10^9
        - Only one valid answer exists.
    """
    pass
```

The file is saved to `~/.lcgrade/workspace/<slug>/solution.py` and opened in the user's editor.

**Editor detection:** `$EDITOR` env var → config file `~/.lcgrade/config.yaml` → fallback to printing the file path.

**Docstring is generated once on `lcgrade start`.** Never overwritten on subsequent runs — the user may add their own notes. `lcgrade reset <slug>` regenerates it.

The problem statement also remains available as standalone markdown at `~/.lcgrade/problems/<slug>/statement.md`.

### Terminal Output Design

#### `lcgrade solve` output

```
 lcgrade · Two Sum · Easy

 Provided Tests        8/8 passed  ✓
 LLM-Generated Tests   3/4 passed  ✗
   ✗ Case: nums=[], target=0 → expected [] but got error
     RuntimeError: list index out of range (line 4)

 Progress
   Solved               ✓ auto-solved from provided tests
   Review Generated     ✓
   Review Acknowledged  ·
   Follow-up Complete   ·

 Complexity
   Time:  O(n)
   Space: O(n)

 Code Quality
   ✓ No issues found

 Review
   Your solution correctly uses a hash map for O(n) lookup.
   Consider: what happens with an empty input array?
   You're not handling the edge case where nums is empty.

   Follow-up: Could you solve this with O(1) space?
```

#### `lcgrade list` output

```
 lcgrade · Blind 75

 Arrays                4/12 solved
 Linked Lists          0/6  solved
 Trees                 1/8  solved
 Dynamic Programming   0/11 solved
 ...

 Progress: 5/75 solved · 8/75 attempted
```

#### `lcgrade list arrays` output

```
 lcgrade · Blind 75 · Arrays

   ✓ Two Sum              Easy    0:42
   ✗ Best Time to Buy     Easy    2 attempts
   · Container With Water  Medium  not started
   · Product Except Self   Medium  not started
   ...
```

#### `lcgrade stats` output

```
 lcgrade · Your Progress

 Solved:     12/75  ██████░░░░░░░░░  16%
 Attempted:  18/75
 Review Generated:  14
 Review Acknowledged:  9
 Follow-up Completed:  6
 Streak:     3 days

 By Difficulty
   Easy:    8/26   ████████░░░░░░  31%
   Medium:  4/39   ███░░░░░░░░░░░  10%
   Hard:    0/10   ░░░░░░░░░░░░░░   0%

 Weakest Tags (by provided test pass rate): dynamic-programming, graph, tree
```

"Weakest Tags" = lowest provided-test pass rate grouped by tag from problem progress / attempt history.

#### `lcgrade history` output

```
 lcgrade · History · Two Sum

 #  Date           Provided Tests  Review   Follow-up  Complexity  Status
 1  Apr 10, 2:14p  6/8             ·        ·          —           attempted
 2  Apr 10, 3:01p  8/8             ✓ gen    ·          O(n²)       auto-solved
 3  Apr 11, 9:30a  8/8             ✓ ack    ✓          O(n)        completed

 Milestones:
   Solved on Apr 10, 3:01p
   Review acknowledged on Apr 11, 9:30a
   Follow-up completed on Apr 11, 9:30a
```

Shows progression across attempts and milestone completion. Attempts without reviews show `·` for review/follow-up state and `—` for complexity if no review exists yet.

#### `lcgrade chat` output (beta: single-shot)

```
$ lcgrade chat "I'm thinking about using a hash map but I'm not sure what to store"

 lcgrade · Chat · Two Sum

 Good instinct. Think about what you're searching for on each
 iteration. If you're looking at nums[i], what value would make it a
 valid pair? What if you stored that "wanted" value as the key?
```

In beta, each `lcgrade chat` call is independent — the LLM sees the problem statement, your current code, and your previous attempts, but not prior chat messages. This already provides rich context for good answers.

### Multi-Turn Chat with Context Compaction (v1 — HIGH PRIORITY)

**This is the highest-priority v1 feature.** Single-shot chat works for beta, but interactive multi-turn conversation is a significantly better experience and a strong portfolio talking point.

v1 adds `lcgrade chat` (no prompt) to enter an interactive session:

```
$ lcgrade chat

 lcgrade · Chat · Two Sum (type 'q' to exit)

 You: I'm thinking about using a hash map but I'm not sure what to store

 lcgrade: Good instinct. Think about what you're searching for on each
 iteration. If you're looking at nums[i], what value would make it a
 valid pair? What if you stored that "wanted" value as the key?

 You: oh so store the complement as the key and the index as value?

 lcgrade: Exactly. And when would you check the map — before or after
 inserting the current element? Think about why the order matters.

 You: q
```

**Context window management:**

Local models have limited context (4-8K tokens for 7B/8B models). Token budget allocation:

1. Always include: problem statement + current code (non-negotiable)
2. Include: last 2-3 previous attempt summaries
3. Fill remaining budget with chat history (most recent turns first)

**Compaction:** When chat history approaches ~70% of the available context budget:

1. Take the oldest chunk of messages
2. Send to the LLM: "Summarize this conversation. What has the user tried, what approaches were discussed, what was ruled out, what direction was suggested next?"
3. Replace the old messages with the compact summary as a single `role: "summary"` message
4. Continue the conversation with: [summary] + [recent messages] + [new user message]

The user gets conversational continuity — the LLM knows what was tried and ruled out — in a fraction of the tokens.

**Chat history lifecycle:**

- Created when user enters `lcgrade chat`
- Persists across terminal sessions (stored in SQLite by session_id)
- Deleted when the user solves the problem or runs `lcgrade reset`

**Chat guardrails:** System prompt instructs the LLM to guide the user toward the answer, not write the solution for them. The LLM should act as a tutor, not a code generator.

### Config File

Located at `~/.lcgrade/config.yaml`:

v1 may also support optional cloud-agent configuration here for users who explicitly want hosted models instead of local backends. That path is convenience-only, not core product direction. The offline/local value proposition remains primary, so MLX and other local backends take precedence over Claude/OpenAI/Gemini API-key support.

```yaml
editor: code                # Editor command ($EDITOR override)
default_test_mode: both     # bundled | llm | both
model: llama3:8b            # Ollama model name
timeout_seconds: 10         # Per-test execution timeout
max_snapshots: 10           # Max stored attempts per problem
log_level: warning          # debug | info | warning | error
```

---

## Dependencies

### `pyproject.toml` (sketch)

```toml
[project]
name = "lcgrade"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "typer>=0.9",           # CLI framework
    "rich>=13.0",           # Terminal formatting
    "psutil>=5.9",          # Memory monitoring for subprocess sandbox
    "pyyaml>=6.0",          # YAML frontmatter parsing
    "httpx>=0.25",          # HTTP client for Ollama API
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff>=0.3",            # Also a runtime dependency called via subprocess
]
mlx = [
    "mlx>=0.5",             # v1: Apple Silicon inference
    "mlx-lm>=0.3",
]

[project.scripts]
lcgrade = "lcgrade.cli:app"
```

**Key decisions:**
- `ruff` is called via `subprocess` (not imported as a library) so it's a dev dependency for development but needs to be installed on the user's machine. `lcgrade init` should check for ruff and suggest `pip install ruff` if missing. Since linting is advisory-only, missing `ruff` should never block solve execution.
- `httpx` over `requests` — async support for future use, lighter weight.
- `mlx` is an optional extra, not a core dependency — keeps the base install small and cross-platform compatible (MLX only works on Apple Silicon).

---

## Logging & Observability

When something goes wrong — LLM returns garbage, subprocess hangs, SQLite corruption — the user needs a way to see what happened. Use Python's `logging` module with configurable levels.

**CLI flags:**
- Default: only warnings and errors shown
- `--verbose` / `-v`: info-level logging (which stages ran, cache hit/miss, model used, timing per stage)
- `--debug`: full diagnostic dump (raw LLM prompts and responses, subprocess commands, SQL queries, file hashes)

**What gets logged at each level:**

| Level | Examples |
|-------|---------|
| ERROR | Ollama connection failed, subprocess crashed, SQLite write failed |
| WARNING | LLM returned malformed JSON (retrying), memory limit approaching, deprecated schema version detected |
| INFO | Cache hit/miss, stage timing ("Stage 1: 2.3s, Stage 2: 4.1s"), model loaded, tests.json hash changed |
| DEBUG | Raw LLM prompt text, raw LLM response text, subprocess command + stdout/stderr, SQL queries, full file hashes |

**Log destination:** stderr (so it doesn't interfere with `lcgrade solve` output on stdout). Optionally write to `~/.lcgrade/lcgrade.log` when `--debug` is set, so users can share log files when reporting issues.

---

## Testing Strategy

The abstractions (Runner, LLMBackend, Validator, Evaluator, Extension) make the project highly testable. This section outlines the testing approach — important both for code quality and as a portfolio talking point.

### MockBackend

A fake LLM backend that returns canned responses. Enables testing the full pipeline without Ollama running.

```python
class MockBackend(LLMBackend):
    def __init__(self, responses: dict[str, str]):
        self.responses = responses  # prompt substring → response text
        self.call_log = []          # records all calls for assertion

    def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=2048):
        self.call_log.append(prompt)
        for key, response in self.responses.items():
            if key in prompt:
                return LLMResponse(text=response, tokens_used=100, latency_ms=10)
        return LLMResponse(text="Mock response", tokens_used=50, latency_ms=5)

    def available(self):
        return True

    def model_info(self):
        return ModelInfo(name="mock-7b", context_window=4096,
                        quantization="none", backend="mock")
```

### Test Layers

| Layer | What's tested | How |
|-------|--------------|-----|
| **Unit: Validators** | `ExactMatch`, `SetEquality` with various inputs | Direct function calls, no dependencies |
| **Unit: Evaluator** | Correct verdicts from raw outputs + test cases | Inject fake `RunResult`, assert `TestVerdict` list |
| **Unit: Runner.validate()** | Signature checking catches wrong function name, missing params | Pass in known-bad source files |
| **Unit: LLMBackend.generate_json()** | JSON parsing, retry on malformed output, fence stripping | `MockBackend` with deliberately bad responses |
| **Integration: Pipeline** | Full solve flow from source file to verdicts + review | `MockBackend` + `PythonRunner` + real evaluator, temp directory with a known problem |
| **Integration: Cache** | Cache hit/miss behavior, hash comparison | Write attempt to SQLite, re-run with same/different code |
| **Integration: Graceful degradation** | Behavior when Ollama unavailable | `MockBackend` with `available()` returning False |
| **E2E: CLI** | Command output, exit codes, file creation | `typer.testing.CliRunner`, invoke commands, assert output |

### Test Fixtures

Ship a `tests/fixtures/` directory with:
- A minimal problem (two-sum with 3 bundled test cases)
- A known-correct solution
- A known-buggy solution (off-by-one)
- A TLE solution (infinite loop)
- A MLE solution (memory bomb)
- Canned LLM responses for review and test generation

This lets the full pipeline be tested deterministically without any LLM or network dependency.

---

## `lcgrade prune` Scope

`lcgrade prune` cleans up storage. It is explicit about what it deletes:

| What | Rule | Command variant |
|------|------|----------------|
| Old attempt snapshots | Delete attempts beyond the 10-per-problem cap (oldest first), along with linked reviews | `lcgrade prune` (default) |
| Chat history for solved problems | Delete chat messages for problems the user has already solved | `lcgrade prune --chats` |
| All user data | Nuclear option — wipe `~/.lcgrade/lcgrade.db` entirely, keep problem bank files | `lcgrade prune --all` |

`lcgrade prune` without flags only cleans snapshots. Destructive operations (`--all`) prompt for confirmation.

---

## Portfolio Narrative & README

### README Structure

1. **One-liner + demo GIF** — terminal recording (asciinema or vhs) showing the full flow: start → write → solve → feedback. 15 seconds.

2. **Technical highlights** — three short paragraphs, each an interview conversation starter:
   - Hybrid evaluation pipeline (deterministic tests + empirical profiling + LLM review)
   - Local inference on Apple Silicon (MLX benchmarks, quantization tradeoffs)
   - Context-aware chat with compaction (sliding window summarization)

3. **Architecture diagram** — Mermaid diagram showing the three-stage pipeline, data flow, and SQLite layer.

4. **Benchmarks** — two suites:
   - *Inference benchmarks*: MLX vs Ollama, quantization levels, tokens/sec, memory usage, first-token latency
   - *Eval accuracy benchmarks*: 20 correct + 20 flawed solutions → measure precision/recall of grader across modes (bundled-only, LLM-only, hybrid)

5. **Quick start** — `pip install lcgrade` → `lcgrade init` → `lcgrade start two-sum`

6. **Configuration & usage** — full command reference

### Blog Posts (Deferred)

Two standalone posts to be written after the project is functional:

1. **"Benchmarking Local LLM Inference on Apple Silicon"** — MLX deep dive with real numbers. Has SEO value beyond the job search.
2. **"Building a Reliable Code Grader with LLMs"** — eval methodology, why pure LLM scoring fails, how hybrid works, compaction strategy.

### Interview Narrative

> "I built a local-first CLI tool that grades LeetCode solutions using a hybrid evaluation pipeline — deterministic test execution combined with empirical complexity profiling and LLM qualitative review. The interesting engineering problems were designing a context-aware chat system with compaction for long conversations, benchmarking inference backends on Apple Silicon, and building an eval methodology that's more reliable than pure LLM scoring. Here's the data."

---

## Open Items

These are design decisions and implementation tasks that still need to be resolved:

### Prompt Engineering

| # | Item | Notes |
|---|------|-------|
| 1 | LLM test case generation prompts | What model size to target (7B? 13B?), system prompt structure, reliable JSON output from local models |
| 2 | Hint/chat system prompt | Guardrails to prevent giving full solutions, progressive guidance |
| 3 | Qualitative review prompt | How to structure the review request given test results + static analysis + code |
| 4 | Compaction prompt design | What to preserve in summary (approaches tried, user's understanding, what was ruled out, next direction) |

### Scoring & Evaluation

| # | Item | Notes |
|---|------|-------|
| 5 | Future scoring model | If a numeric score is added later, it should measure analysis depth/practice completeness rather than correctness |
| 6 | Compaction threshold | At what % of context budget to trigger summarization (~70% suggested) |
| 7 | Extensibility plugin format | How are custom community extensions defined? Prompt templates? Python plugins? Config-driven? |

### Data & Schema

| # | Item | Notes |
|---|------|-------|
| 8 | Reference solution authoring | Writing Python solutions for 60+ Blind 75 problems, running conversion against cojudge inputs |
| 9 | Problem bank distribution | Ship inside pip package vs separate `lcgrade init` download? Pip has package size limits |
| 10 | `scaling_inputs` generators | Library of input generators per problem type (random arrays, graphs, strings) for empirical profiling |

### Execution & Sandbox

| # | Item | Notes |
|---|------|-------|
| 11 | Docker sandbox option (v1) | `DockerRunner` wrapping any language runner inside a container |
| 12 | Performance benchmarking harness | Generating varying-size inputs, timing, presenting the scaling curve |
| 13 | Stronger isolation model | Current beta sandbox is subprocess guardrails for a local tool, not strict OS/container isolation |

### CLI & UX

| # | Item | Notes |
|---|------|-------|
| 13 | Editor integration | `$EDITOR` detection, config fallback, behavior when no editor is set |
| 14 | Workspace structure | Support multiple attempts as separate files? Or single solution.py snapshotted to SQLite? |
| 15 | Docstring generation template | Handling long descriptions, constraint formatting |
| 16 | Terminal width handling | Test Rich output in narrow (80 cols) vs wide terminals |
| 17 | Color/accessibility | `--no-color` flag, `NO_COLOR` env var support |
| 18 | Completion time tracking | Track time from `lcgrade start` to successful solve. Handle breaks/pauses? |
| 19 | Shell completions | Typer provides these, but need to document install for bash/zsh/fish |
| 20 | Chat context budget | Exact token allocation strategy per section, adapt to model's context window size |
| 21 | Chat session persistence | Sessions carry across terminal closes via SQLite, resume on reopen |
| 22 | `lcgrade chat` safety | System prompt guardrails to prevent LLM from giving full solutions |
| 23 | Interactive TUI browser (v1) | `textual`-based problem navigator with search, filter, arrow keys |
| 24 | `lcgrade random` weighting | Prioritize weak tags (smart coaching) vs true random? |

### Multi-Language (Runner abstraction ships in beta, additional languages post-beta)

| # | Item | Notes |
|---|------|-------|
| 25 | Additional Runner implementations | `JavaRunner`, `CppRunner`, `GoRunner` — each implements `validate()`, `generate_harness()`, `compile()`, `execute()` |
| 26 | Language-specific starter code | Each problem needs a function stub per supported language |
| 27 | Harness generator per language | Each language needs its own test harness template |

### Portfolio & Documentation

| # | Item | Notes |
|---|------|-------|
| 28 | asciinema/vhs demo recording | Script the demo flow for a clean 15-second terminal recording |
| 29 | Mermaid architecture diagram | Design the pipeline visualization for README |
| 30 | Blog post platform | Personal site for maximum portfolio signal |
| 31 | Benchmark reproducibility | Ship benchmark scripts, `lcgrade benchmark` command |
| 32 | Progress export | Should users be able to export stats as JSON/markdown for sharing? |
| 33 | Custom problem authoring UX | What does `lcgrade add` look like? Scaffold the folder structure? |

---

## Key Design Decisions (Summary)

For quick reference, these are the decisions made during the design process:

1. **All abstractions ship in beta** — Runner, LLMBackend, Validator, Evaluator, and Extension base classes are built upfront. Only the beta-required concrete classes are implemented. Adding new languages, backends, validators, or extensions later is just adding a new class — no core refactoring.
2. **Ollama first, MLX later** — `OllamaBackend` ships in beta, `MLXBackend` in v1. Both implement `LLMBackend`.
3. **Files as source of truth, SQLite as index** — Problems are markdown files (git-friendly, human-editable), SQLite indexes them for fast queries and stores user state
4. **Three-stage eval pipeline with preflight and cache** — Cache check → Preflight (syntax, signature, optional lint notes) → Stage 1: Test cases → Stage 2: LLM review → Stage 3: LLM extensibility. Each stage is a natural stopping point.
5. **Cache on (code_hash, test_mode, tests_hash)** — If code, test mode, and provided tests haven't changed, skip execution and show cached results. `--force` bypasses. `--extend` only runs new extensions without re-running Stages 1-2. Problem bank updates invalidate the cache via `tests_hash`.
6. **Preflight checks are fail-fast only where correctness matters** — syntax errors and missing/wrong-arity functions block execution; lint/style feedback is advisory
7. **Runner executes, Evaluator judges** — Single Responsibility. Runner captures raw outputs, Evaluator compares against expected values using Validators. Neither does the other's job.
8. **LLM review and extensibility are separate stages** — Stage 2 is a fixed, opinionated analysis (same for every problem). Stage 3 is optional, pluggable, user-controlled (interview questions, custom review criteria).
9. **Pipeline saves results per stage** — Stage 1 results saved immediately to `attempts` table. Stage 2 saved to `reviews` table. If a later stage fails, earlier results are preserved. `lcgrade review` re-runs Stages 2/3 without re-executing tests.
10. **Graceful degradation** — Tool always does something useful. No local LLM? Core offline solving still works. Stage 2 fails? Show test results and offer retry.
11. **LLM for complexity analysis from code structure** — AST heuristics are brittle; the LLM reads code and identifies patterns (nested loops, hash map usage, recursion with memoization). Empirical timing added in v1 to complement.
12. **Bundled test answers pre-computed** — cojudge only has inputs; we run Python reference solutions once to generate expected outputs that ship with the package
13. **LLM fallback for missing answers** — When no reference solution exists (custom problems), LLM generates expected outputs, flagged as unverified
14. **subprocess over Docker for beta** — Users run their own code locally; Docker adds friction without proportional safety benefit. This is guardrail-based execution, not strict isolation.
15. **Max 10 snapshots per problem** — Caps storage, oldest deleted first, `lcgrade prune` for manual cleanup
16. **Chat history wiped on solve/reset** — No long-term chat accumulation, keeps storage bounded
17. **Compaction over truncation** — When approaching context limit, summarize old messages instead of silently dropping them
18. **Runner abstraction in beta, Python only** — `Runner` base class ships in beta with `PythonRunner`. Adding new languages is just subclassing.
19. **Layered `lcgrade list`** — Categories first, then drill into individual problems, `--all` for flat view
20. **Single-shot chat in beta, multi-turn in v1 (HIGH PRIORITY)** — Beta ships `lcgrade chat "question"` (independent calls with problem + code + attempt context). v1 adds interactive multi-turn with session persistence and context compaction.
21. **`lcgrade random` command** — Picks unsolved problem with optional difficulty/tag filters
22. **Problem statement embedded in solution file as docstring** — User never leaves their editor to read the problem
23. **Blog posts deferred** — Write after the project is functional, not a beta or v1 blocker

---

## Implementation Guide

### Recommended Build Order

Build in this order to get a working end-to-end flow as fast as possible. Resist the urge to polish any layer before the full pipeline works.

| Step | Component | Why this order |
|------|-----------|---------------|
| 1 | SQLite schema + migrations | Foundation everything writes to. 10 minutes of work that unblocks everything. |
| 2 | Problem bank conversion (5 problems only) | Just enough to test with: two-sum, reverse-linked-list, and 3 others across categories. Not all 60+. |
| 3 | PythonRunner | Harness generator, subprocess execution, `parse_output()`. Test standalone: solution file + test cases → `RunResult`. |
| 4 | Evaluator | Takes `RunResult` + test cases → `TestVerdict` list. Unit test heavily with fixtures. |
| 5 | MockBackend + test fixtures | Define what LLM responses should look like BEFORE writing real prompts. Forces you to design the prompt/response format. Pipeline tests work without Ollama. |
| 6 | OllamaBackend | `generate()`, `available()`, `model_info()`. Test with a simple prompt to verify inference works. |
| 7 | Pipeline | Wire Runner + Evaluator + LLMBackend together. Cache check, save-as-you-go, error recovery. This is where the design comes together. |
| 8 | CLI commands | Thin wrappers: `lcgrade init`, `lcgrade start`, `lcgrade solve`. Minimal Rich formatting — ugly but functional. |
| 9 | Stage 2 prompts + Stage 3 extensions | Prompt engineering, not architecture. Iterate once the pipeline works. |
| 10 | Polish | Rich output formatting, `lcgrade list`, `lcgrade stats`, `lcgrade history`. Make it pretty. |

**The key principle: get `lcgrade solve` working end-to-end with ugly output before making anything pretty.** The Pipeline + Runner + Evaluator loop is the core. Everything else layers on top.

### What Will Be Harder Than It Looks

**Prompt engineering for test case generation.** Getting a 7B local model to reliably output valid JSON test cases is finicky. The model will sometimes output commentary before the JSON, nest it in markdown, or produce test cases with subtly wrong expected outputs. The `generate_json()` retry logic handles malformed JSON, but semantically wrong outputs (valid JSON, wrong answer) are harder to catch. Recommendation: start with `--tests bundled` as the effective default for beta. Don't let unreliable LLM test generation be the first thing users experience. Get the provided-test path rock-solid first.

**Harness transport and run-level failure handling.** Do not rely on stdout as the transport for structured harness results. User `print()` calls can corrupt stdout and make valid runs look broken. Write structured per-test results to a temp JSON file, capture user stdout/stderr separately, and let the Evaluator explicitly handle run-level failures like `TLE`, `MLE`, `compile_error`, and `harness_error`.

**The cojudge conversion script.** The doc says "one-time conversion" but it's non-trivial. cojudge problems have Java-specific input formats, their `metadata.json` schema differs from our YAML schema, and some problems have edge cases in how inputs are structured. Budget a full day, not an afternoon. Start with 5 problems, validate the format, then batch-convert the rest.

**Syntax validation before linting.** The hard preflight gate should be `ast.parse()` plus function existence/arity validation, not `ruff`. Users expect syntax errors and missing required functions to stop execution immediately. They do not expect style suggestions to block grading. Keep linting advisory and surface it only when available.

### Implementation Notes

**Write MockBackend fixtures first, real OllamaBackend second.** Define the canned LLM responses for test generation, review, and extensions before you start prompt engineering. This forces you to design the prompt/response contract, and it means Pipeline integration tests work without Ollama running. Iterate on real prompts later without breaking tests.

**Validator.check() returns bool, Evaluator generates messages.** The `TestVerdict.message` field (human-readable context like "expected [0,1] but got [1,0] — did you mean to use set_equality?") is produced by the Evaluator, not the Validator. The Validator just says pass/fail. The Evaluator interprets the result in context — it knows what the expected and actual values were, which validator was used, and can suggest alternatives. Keep this separation clean during implementation.

**Active problem state needs error handling.** If the user deletes their workspace file or the active slug points to a problem that no longer exists (deleted from problem bank), every command that relies on `get_active_slug()` needs a graceful error: "No active problem. Run `lcgrade start <slug>` to begin."
