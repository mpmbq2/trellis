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
make fmt       # Format code with ruff
make lint      # Lint with ruff
make typecheck # Type check with mypy
make test      # Run all tests
make check     # Run fmt + lint + typecheck + test
```

### Running single tests

```sh
# Single test file
uv run pytest tests/test_abstract_dataset.py

# Single test function
uv run pytest tests/test_abstract_dataset.py::test_repr_includes_class_and_location

# Single test by name pattern
uv run pytest -k "test_repr"
```

### Pre-commit

`.pre-commit-config.yaml` runs formatting, lint, and type checks on commit. Install with:

```sh
uv run pre-commit install
```

## Code style

### Imports

- Always use `from __future__ import annotations` in new modules
- Use absolute imports from the package root:
  ```python
  from trellis.datasets import AbstractDataset
  from trellis.datasets.csv import CSVDataset
  ```
- Avoid relative imports like `from .abstract import ...`

### Type annotations

- Modern union syntax: `str | None`, `list[str]`, `dict[str, int]`
- Return type required on all public functions/methods
- Use `Any` sparingly; prefer specific types
- Type ignore comments only when necessary (`# type: ignore[abstract]`)

### Naming conventions

- **Classes:** `PascalCase` (e.g., `CSVDataset`, `AbstractDataset`)
- **Functions/methods:** `snake_case` (e.g., `load`, `save`, `exists`)
- **Variables:** `snake_case` (e.g., `location`, `data_path`)
- **Constants:** `SCREAMING_SNAKE_CASE` if module-level and truly constant
- **Private:** Leading underscore for internal APIs (`_internal_method`)

### Docstrings

Follow Google-style docstrings:

```python
def load(self, *, backend: Literal["polars", "pandas"] = "polars") -> Any:
    """Load data from the CSV location.

    Args:
        backend: Which library to use ("polars" or "pandas").

    Returns:
        polars DataFrame or pandas DataFrame.
    """
```

### Abstract methods

Use `...` (ellipsis) for abstract method bodies, not `pass` or bare `raise NotImplementedError`:

```python
@abstractmethod
def load(self) -> Any:
    """Load data from the dataset location."""
    ...
```

### Error handling

- Raise specific exceptions with clear messages:
  ```python
  raise TypeError(f"Unsupported data type: {type(data)}")
  ```
- Let exceptions propagate; avoid bare `except:` clauses
- Use `pytest.raises` in tests for expected exceptions

### Formatting

- Let **Ruff** own formatting; do not fight the formatter
- Line length: default Ruff settings (88 chars)
- No trailing whitespace
- End files with a newline

## Architecture

- **`AbstractDataset`** is the foundation: `src/trellis/datasets/abstract.py`
- Concrete classes subclass it and are instantiated directly:
  ```python
  ds = CSVDataset(location="/path/to/data.csv")
  ```
- A string-keyed registry on the ABC is intentionally **not** part of Trellis—use plain imports or your own catalog/config layer if you need name→class mapping
- Do **not** add optional "kitchen sink" APIs to the ABC for hypothetical features; keep shared code in functions or separate modules unless multiple subclasses need the same protocol

## Testing

- Tests under `tests/`
- Test file naming: `test_<module>.py`
- Test function naming: `test_<behavior>`
- Use `pytest` assertions (not `assertEqual`)
- Stub classes for testing abstract classes:

```python
class _StubDataset(AbstractDataset):
    def load(self) -> str:
        return "ok"

    def save(self, data: object) -> None:
        return None

    def exists(self) -> bool:
        return True
```

## File organization

```
src/trellis/
  __init__.py       # Package init (may export public API)
  datasets/
    __init__.py     # Dataset exports
    abstract.py     # AbstractDataset
    csv.py          # CSVDataset
    parquet.py      # ParquetDataset (example)
tests/
  test_abstract_dataset.py
  test_csv_dataset.py
```

## Dataframe / Ibis operations

When operating on a single dataframe or table in a linear, step-by-step fashion, prefer **method chains** for clarity. Chains keep the logic readable by making the transformation sequence explicit—no need to track intermediate variables or state.

```python
# Prefer this
result = (
    df.filter(pl.col("status") == "active")
    .with_columns(pl.col("value").cast(pl.Float64))
    .group_by("category")
    .agg(pl.col("value").mean().alias("avg_value"))
    .sort("avg_value", descending=True)
)

# Over this
df = df.filter(pl.col("status") == "active")
df = df.with_columns(pl.col("value").cast(pl.Float64))
df = df.group_by("category").agg(pl.col("value").mean().alias("avg_value"))
result = df.sort("avg_value", descending=True)
```

This is not a hard requirement—use your judgment—but chains shine when you're doing a sequence of independent transformations on a single object.

## Before submitting

1. Run `make check` (or at minimum `make lint` and `make typecheck`)
2. Ensure all tests pass
3. Check that pre-commit hooks pass if installed
