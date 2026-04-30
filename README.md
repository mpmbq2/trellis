# Trellis

A small, framework-agnostic dataset I/O abstraction for Python data pipelines.

Trellis takes the idea behind Kedro's `AbstractDataset` and ships it as a standalone library, so pipelines built on Metaflow, Prefect, Dagster, Airflow, or plain Python can swap storage and formats without rewriting pipeline code. It does **one** thing — `load` / `save` / `exists` behind a common shape — and leaves orchestration, catalogs, config, and runners to your framework of choice.

> **Status:** pre-1.0. APIs may change.

## Why

Most pipeline frameworks have strong opinions about scheduling, state, and DAGs, and weak opinions about how you read external data and write final outputs. Pipelines end up with bespoke "load this CSV, write that Parquet" boilerplate per project. Trellis is the boring shared layer for that boilerplate, no framework lock-in.

## What it is not

- Not a pipeline framework.
- Not a catalog or config system. Use a plain `dict`, your framework's resource system, or a separate library if you need name → dataset.
- Not a Kedro replacement. If you want versioning, lineage, hooks, and YAML-driven config, use Kedro.

## Install

```sh
pip install trellis
```

Heavy dependencies are opt-in via extras (planned: `polars`, `pandas`, `sql`, `all`). Until those land, the current dependency set installs everything.

## Quick example

```python
from trellis import CSVDataset, ParquetDataset, SQLDataset

# Read a CSV from local disk or any fsspec-compatible URI.
raw = CSVDataset(location="s3://bucket/raw/users.csv")
df = raw.load(backend="polars")

# Write Parquet.
clean = ParquetDataset(location="s3://bucket/clean/users.parquet")
clean.save(df)

# Read/write a SQL table via ibis (DuckDB, Postgres, SQLite, ...).
table = SQLDataset(location="duckdb:///warehouse.db", table_name="users")
table.save(df, if_exists="replace")
ibis_expr = table.load(backend="ibis")
```

The same three calls — `load()`, `save()`, `exists()` — work across every dataset type.

## Currently shipped

| Dataset | Backends | Notes |
|---|---|---|
| `CSVDataset` | polars, pandas | local + fsspec remote |
| `ParquetDataset` | polars, pandas | local + fsspec remote |
| `SQLDataset` | ibis, polars, pandas | DuckDB, Postgres, SQLite, etc. via ibis |
| `SQLDatasource` | ibis, polars, pandas | read-only counterpart |

New dataset types are deliberately cheap to add — that is the project's main design constraint.

## Project direction

The single source of truth for scope, design decisions, and roadmap is [.agent_notes/GRAND_PLAN.md](.agent_notes/GRAND_PLAN.md). Contributor and AI-agent guidelines live in [AGENTS.md](AGENTS.md).
