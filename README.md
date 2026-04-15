# LocalLeetcode

`lcgrade` is a local-first CLI auto-grader for LeetCode-style interview practice. This repo is now scaffolded around the design doc with a package layout, initial CLI commands, a SQLite bootstrap path, and a sample bundled problem.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
lcgrade init
lcgrade list-problems
lcgrade solve two-sum
pytest
```
