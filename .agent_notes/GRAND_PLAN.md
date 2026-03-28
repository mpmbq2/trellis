# Trellis — direction

This file is the **source of truth** for what Trellis is trying to be. Keep it short and accurate so agents (and humans) do not invent scope.

## Purpose

Trellis is a **small** Python library that standardizes **dataset I/O** behind a common shape so data pipelines can swap storage/format without rewriting pipeline code.

**Priority: simplicity.** Adding a new dataset type should be easy—minimal abstract API, no ceremony, no required framework.

## Core abstraction: `AbstractDataset`

The base type lives at `src/trellis/datasets/abstract.py`.

- **Contract:** `load()`, `save(data)`, `exists()` only. The base class is **format-agnostic**: it does not assume tabular data, a particular engine, or specific types for `load`/`save`.
- **Identity:** an optional string `location` (path, URI, or whatever that dataset type uses). Subclasses add their own constructor parameters (e.g. SQL connection settings) as needed.
- **No registry on the ABC:** construct concrete types directly (`CSVDataset(...)`, etc.). If you need string names or YAML-driven setup, that belongs in a separate layer (e.g. a catalog), not baked into the base class.

**Not on the abstract base:** SQL querying helpers, DataFrame conversion tables, metadata/versioning hooks, or “incremental” protocols. If a concrete class needs extra methods (`query`, partitions, append mode), it adds them; duck typing is fine. Shared helpers belong in **plain functions or small modules**, not on the ABC, unless several implementations truly share a stable protocol (decide that when the second implementation exists).

## Implementation order (near term)

1. **Solidify `AbstractDataset`** — types, docstrings, `...` abstract bodies, tests.
2. **`CSVDataset`** — first real I/O; establish patterns (e.g. PyArrow + fsspec).
3. **`ParquetDataset`**
4. **`SQLDataset`** (or similar for relational tables)

Revisit design after that (JSON lines, DuckDB, catalogs, multi-file bundles, incremental loads, etc.). Do **not** treat a long wish list as committed work.

## Future possibilities (not scheduled)

Ideas that may or may not land: multi-file or composite datasets, append/incremental semantics, a named catalog over datasets, YAML config, lineage. Capture them in design discussions or issues when relevant—avoid baking them into the base class prematurely.

## Dependencies

See `pyproject.toml`. Keep the install set justified by what the code actually uses.
