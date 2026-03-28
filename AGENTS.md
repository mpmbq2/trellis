# AGENTS.md -- Trellis

Guidelines for AI coding agents working in this repository.

## Project Overview

Trellis is a lightweight, pure-Python data catalog and I/O adapter library for
data pipelines. It provides a standardized `AbstractDataset` ABC with a
registry/factory pattern. Concrete subclasses (Parquet, CSV, JSON, DuckDB, SQL)
implement `.load()`, `.save()`, and `.exists()`.

- **Language:** Python 3.12+ (pinned via `.python-version`)
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build backend:** Hatchling
- **Project layout:** `src`-layout (`src/trellis/`)
- **Runtime deps:** `pyarrow`, `fsspec`
- **Dev deps:** `pytest`, `ruff`, `mypy`

## Build / Lint / Test Commands

All commands use `uv run` (never raw `python` or `pip`). Makefile targets:

```sh
make fmt          # Format code and auto-fix lint issues
make lint         # Lint only (no auto-fix)
make typecheck    # Run mypy on src/
make test         # Run full test suite
make check        # Run all of the above in sequence
```

Underlying commands (if you need to run them directly):

```sh
uv run ruff format .          # Format
uv run ruff check --fix .     # Lint + auto-fix
uv run ruff check .           # Lint only
uv run mypy src/              # Type checking
uv run pytest                 # All tests
uv run pytest tests/test_foo.py              # Single test file
uv run pytest tests/test_foo.py::test_bar    # Single test function
uv run pytest -k "test_bar"                  # Tests matching pattern
```

**Always run `make check` (or at minimum `make lint` and `make typecheck`)
before considering your work done.**

## Pre-commit Hooks

A `.pre-commit-config.yaml` is configured. Hooks run on every commit:

- Trailing whitespace removal, EOF fixer, YAML/TOML checks, merge-conflict
  detection, large-file guard
- **Ruff** lint (`--fix`) and format
- **mypy** type checking

If a commit is rejected by pre-commit, fix the issues and commit again.

## Code Style

### Imports

- Always add `from __future__ import annotations` as the first import in every
  module (PEP 563 deferred annotations).
- Group imports in this order, separated by blank lines:
  1. `__future__`
  2. Standard library
  3. Third-party packages
  4. Local/project imports
- Use absolute imports from the package root: `from trellis.datasets import AbstractDataset`
  (not relative imports like `from .datasets import ...`).

### Type Annotations

- Use modern Python 3.12 syntax everywhere:
  - `str | None` (not `Optional[str]`)
  - `list[str]`, `dict[str, Any]` (lowercase builtins, not `List`, `Dict`)
  - `type[Foo]` (not `Type[Foo]`)
- Use `ClassVar` from `typing` for class-level attributes.
- Annotate all function signatures including return types.
- Use `Any` sparingly; prefer concrete types when possible.

### Formatting

- Ruff handles all formatting -- do not manually adjust style.
- 88-character line length (Ruff default).
- 4-space indentation, no tabs.
- Trailing commas in multi-line structures.
- Single blank line between methods; two blank lines between top-level
  definitions.

### Naming Conventions

- `snake_case` for functions, methods, variables, and modules.
- `PascalCase` for classes.
- `_leading_underscore` for private/internal attributes and methods.
- `UPPER_SNAKE_CASE` for module-level constants.
- Dataset subclass names end with `Dataset` (e.g., `ParquetDataset`).

### Docstrings

- Google-style docstrings on all public classes, methods, and functions.
- Include `Args:`, `Returns:`, and `Raises:` sections where applicable.
- Class docstrings describe purpose; method docstrings describe behavior.

Example:
```python
def create(cls, name: str, **kwargs: Any) -> AbstractDataset:
    """Instantiate a registered dataset subclass by *name*.

    Args:
        name: The registered name of the dataset type (e.g. ``"parquet"``).
        **kwargs: Arguments forwarded to the subclass constructor.

    Returns:
        An instance of the registered subclass.

    Raises:
        KeyError: If *name* has not been registered.
    """
```

### Abstract Methods

- Use `...` (Ellipsis) as the body for abstract methods, not `pass` or
  `raise NotImplementedError`.

### Error Handling

- Raise specific built-in exceptions (`ValueError`, `KeyError`, `TypeError`)
  with descriptive messages.
- Include actionable context in error messages (e.g., list available options
  when a lookup fails).
- Do not use bare `except:` or `except Exception:` without re-raising.

### Code Organization

- Use comment-block section dividers to group related methods within a class:
  ```python
  # ------------------------------------------------------------------
  # Section name
  # ------------------------------------------------------------------
  ```
- Keep modules focused: one primary class or concern per module.
- Re-export public API through `__init__.py` files using `__all__`.

### f-strings

- Prefer f-strings for string formatting.
- Use `!r` for repr-formatting in f-strings (e.g., `f"path={self._path!r}"`).

## Testing

- Tests live in the `tests/` directory, mirroring `src/trellis/` structure.
- Test files are named `test_<module>.py`.
- Test functions are named `test_<behavior>`.
- Use plain `pytest` conventions (no unittest classes unless needed).
- Run `uv run pytest -x` to stop on first failure during development.

## Architecture Notes

- The `AbstractDataset` ABC at `src/trellis/datasets/abstract.py` is the
  foundation. All dataset types inherit from it and must implement `load()`,
  `save()`, and `exists()`.
- Subclasses register themselves via the `@AbstractDataset.register("name")`
  decorator and can be instantiated via `AbstractDataset.create("name", ...)`.
- See `.agent_notes/GRAND_PLAN.md` for the full roadmap and architectural
  decisions.
