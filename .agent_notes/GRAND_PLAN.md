# Trellis — direction

This file is the **source of truth** for what Trellis is trying to be. Keep it short and accurate so agents (and humans) do not invent scope.

## Purpose

Trellis is a **small** Python library that standardizes **dataset I/O** behind a common shape so data pipelines can swap storage/format without rewriting pipeline code. It is Kedro's `AbstractDataset` concept lifted out as a standalone library for use from **non-Kedro pipeline frameworks** (Metaflow, Prefect, Dagster, Airflow, plain Python).

**Priority: simplicity, and extension cost near zero.** Adding a new dataset type must be cheap enough that there is no excuse not to do it. Trellis is *not* a pipeline framework, *not* a catalog, *not* a runner.

## Core abstraction: `AbstractDataset`

The base type lives at `src/trellis/datasets/abstract.py`.

- **Contract:** `load()`, `save(data)`, `exists()`, and `describe()` only. The base class is **format-agnostic**: it does not assume tabular data, a particular engine, or specific types for `load`/`save`.
- **Identity:** an optional string `location` (path, URI, or whatever that dataset type uses). Subclasses add their own constructor parameters as needed.
- **No registry on the ABC:** construct concrete types directly (`CSVDataset(...)`, etc.). If you need string names or YAML-driven setup, that is your concern, not Trellis's.

**Not on the abstract base:** versioning, credentials handling, template-method `_load`/`_save` split, lineage, metadata hooks, "incremental" protocols. If a concrete class needs extra methods (`query`, partitions, append mode), it adds them; duck typing is fine. Shared helpers belong in **plain functions or small modules**.

### `describe()` contract

`describe()` returns a JSON-serializable `dict[str, Any]` with at least `type` and `location` keys. Subclasses add their own fields. Rules:

- **No I/O.** `describe()` is a cheap, pure-config inspector — never queries the underlying store.
- **`__repr__` is derived from `describe()`** (single source of truth; subclasses do not override `__repr__`).
- **Subclasses redact their own secrets** (e.g., a `SQLDataset` location with embedded credentials is redacted in `describe()`).

### Signature conventions (loose by intent, uniform by convention)

The ABC mandates only `load(self) -> Any` and `save(self, data) -> None`. Concrete classes may add kwargs; document and follow these conventions where applicable:

- `load(*, backend: Literal[...] = ...)` for selecting in-memory representation (polars/pandas/ibis/etc.).
- `save(data, *, if_exists: Literal["fail", "replace", "append"] = ...)` for write modes.
- Format-specific kwargs (`compression`, `partition_cols`, …) follow the underlying library's spelling.

Swap-without-rewrite holds when datasets share backends and data shapes; otherwise expect to touch the call site.

## `AbstractDatasource`: read-only protocol

`AbstractDatasource` is a `Protocol` (or tiny ABC) exposing `load()` + `exists()` only. `AbstractDataset` extends it by adding `save()`. Every `Dataset` is therefore also a `Datasource` — there is no parallel concrete hierarchy. Code that wants to express "read-only" types it as `AbstractDatasource`. Standalone read-only types (e.g., HTTP-backed sources with no save story) subclass `AbstractDatasource` directly.

## `_FileDataset`: shared scaffolding

`_FileDataset(AbstractDataset)` is internal scaffolding for fsspec-backed file formats. It owns:

- `location` parsing and `storage_options` plumbing,
- a default `exists()` via fsspec stat,
- `_open_read()` / `_open_write()` helpers.

Subclasses (CSV, Parquet, JSON, Text, Pickle, …) implement the file-level read/write only. `_FileDataset` is **implementation scaffolding, not part of the public contract** — `AbstractDataset` stays minimal.

## Public API and lazy imports

`trellis/__init__.py` uses PEP 562 `__getattr__` to expose dataset classes lazily, plus a `TYPE_CHECKING` block so type checkers see the names. Adding a dataset is a one-line entry in the lazy-export table.

Heavy dependencies (`polars`, `pandas`, `ibis`, `duckdb`, future `torch`, `Pillow`, …) are:

1. **Imported lazily** inside the dataset module (not at top of file).
2. **Declared as extras** in `pyproject.toml` (`pip install trellis[polars]`, `trellis[sql]`, `trellis[all]`, …).

Missing extras raise a clear `ImportError` at use time with a `pip install trellis[<extra>]` hint.

## Errors

A small exception hierarchy in `src/trellis/exceptions.py`:

- `TrellisError(Exception)`
- `DatasetNotFoundError(TrellisError, FileNotFoundError)` — missing data
- `DatasetLoadError(TrellisError)` — failures during load
- `DatasetSaveError(TrellisError)` — failures during save

Datasets wrap underlying-library errors at the public boundary (`raise DatasetLoadError(...) from e`) so the original traceback is preserved under `__cause__`. Pipeline code can `except TrellisError` for portable handling, or keep `except FileNotFoundError` for missing-data cases.

## Sync only

The contract is sync. Async users wrap calls themselves: `await asyncio.to_thread(ds.load)`. Trellis does not ship async wrappers, dual `aload`/`asave`, or an async core. This may be revisited only if a dataset appears whose underlying client is *natively* async and benefits from concurrent I/O — and even then, async is added as an opt-in subclass method, not promoted to the ABC.

## No catalog

Trellis intentionally does not ship a catalog, registry, or config-driven construction. Users who want a name→dataset mapping use a plain Python dict, their pipeline framework's resource system (Prefect blocks, Dagster resources, Metaflow `IncludeFile`), or a separate library. This is a deliberate non-feature: every step from "tiny catalog" to "YAML-driven config" is how Kedro grew, and Trellis differentiates by stopping before that slope.

## Testing

`trellis.testing` is **shipped** (importable, not tests-only) and exposes a `DatasetContract` mixin. Each dataset's test class subclasses `DatasetContract`, supplies `make_dataset(tmp_path)` and `sample_data()`, and gets a fixed set of contract tests for free (`exists` before/after save, round-trip, `describe()` includes required keys, repr non-empty). Dataset-specific tests are added on top. Third-party dataset packages can use the same machinery.

## Current status

**Datasets** (read/write):

| Dataset | Backend | Notes |
|---------|---------|-------|
| `CSVDataset` | polars, pandas | fsspec for remote storage |
| `ParquetDataset` | polars, pandas | fsspec for remote storage |
| `SQLDataset` | ibis, polars, pandas | DuckDB, PostgreSQL, SQLite, etc. via ibis |

**Datasources** (read-only):

| Datasource | Backend | Notes |
|------------|---------|-------|
| `SQLDatasource` | ibis, polars, pandas | Will collapse: `SQLDataset` itself satisfies the `AbstractDatasource` protocol after Q4 refactor |

## Implementation order

Done:

1. Solidify `AbstractDataset` — types, docstrings, `...` abstract bodies, tests. ✅
2. `CSVDataset` — first real I/O; establish fsspec patterns. ✅
3. `ParquetDataset` ✅
4. `SQLDataset` ✅
5. `AbstractDatasource` / `SQLDatasource` ✅ (will be refactored — see below)

Next, in roughly this order:

6. **Add `describe()` to `AbstractDataset`**, derive `__repr__` from it, update existing concrete classes (including secret redaction in `SQLDataset`).
7. **Introduce `TrellisError` hierarchy** and wrap errors at boundaries in current concrete classes.
8. **Collapse `Datasource`** to a `Protocol` and remove the parallel `SQLDatasource` concrete class.
9. **Restructure dependencies** — move `polars`/`pandas`/`ibis`/`duckdb` to extras; add lazy imports inside each dataset module.
10. **Add PEP 562 lazy exports** in `trellis/__init__.py` with `TYPE_CHECKING` block.
11. **Extract `_FileDataset`** from CSV/Parquet; migrate both onto it.
12. **Ship `trellis.testing.DatasetContract`** and migrate existing test files onto it.
13. **New file-based datasets, on demand:** `JSONDataset`, `JSONLinesDataset`, `TextDataset`, `YAMLDataset`, `PickleDataset`. Add only when needed.
14. **Framework integration docs** (Metaflow first, then Prefect/Dagster) as short README sections or `docs/` pages — no sphinx/mkdocs site yet.

Do **not** treat unscheduled wishlist items as committed work.

## Audience and maturity

Trellis is a niche public library, pre-1.0, published to PyPI. Breaking changes are allowed during 0.x; record them in `CHANGELOG.md`. Docs target users on Metaflow / Prefect / Dagster / plain Python — that's the differentiation, so it earns the page count. No sphinx site, no governance, no plugin ecosystem until there are real external users beyond the author.

## Future possibilities (not scheduled)

Ideas that may or may not land: append/incremental semantics, partitioned datasets, multi-file bundles, image/audio/model-artifact datasets, async-native datasets where the underlying client is natively async. Capture them in design discussions or issues when relevant — avoid baking them into the base class prematurely.

## Dependencies

Core install must stay tiny. Heavy/format-specific deps are extras; see `pyproject.toml`. Keep the install set justified by what core code actually uses.
