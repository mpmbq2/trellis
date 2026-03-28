# AGENTS.md — Trellis

Guidelines for AI coding agents working in this repository.

## Project overview

Trellis standardizes **dataset I/O** for data pipelines behind a small abstract type: **`AbstractDataset`** (`load` / `save` / `exists`). **Simplicity comes first**; new dataset types should stay cheap to implement—subclass and construct concrete classes directly.

**Roadmap and intent:** [.agent_notes/GRAND_PLAN.md](.agent_notes/GRAND_PLAN.md).

- **Language:** Python 3.12+ (see `.python-version`)
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Layout:** `src/`-layout under `src/trellis/`
- **Dev tools:** `pytest`, `ruff`, `mypy` (see `pyproject.toml`)

## Build / lint / test

Use `uv run` for project commands. Makefile targets:

```sh
make fmt
make lint
make typecheck
make test
make check
```

Run `make check` (or at least lint + typecheck) before considering work complete.

## Pre-commit

`.pre-commit-config.yaml` runs formatting, lint, and related checks on commit.

## Code style (defaults)

- Prefer `from __future__ import annotations` in new modules.
- Absolute imports from the package root (e.g. `from trellis.datasets import AbstractDataset`).
- Modern annotations (`str | None`, `list[str]`, etc.).
- Abstract method bodies use `...`, not `pass` or bare `raise NotImplementedError`.
- Let **Ruff** own formatting; do not fight the formatter.

## Architecture

- **`AbstractDataset`** is the foundation: `src/trellis/datasets/abstract.py`.
- Concrete classes subclass it and are instantiated normally (e.g. ``CSVDataset(location=...)``). A string-keyed registry on the ABC is intentionally **not** part of Trellis—use plain imports or your own catalog/config layer if you need name→class mapping.
- Do **not** add optional “kitchen sink” APIs to the ABC for hypothetical features; keep shared code in functions or separate modules unless multiple subclasses need the same protocol.

## Testing

- Tests under `tests/`, `test_<module>.py`, functions `test_<behavior>`.
- Use `uv run pytest`; `uv run pytest -x` is useful while iterating.
