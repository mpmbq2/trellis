# Trellis: Big-Picture Plan

## The Goal
Trellis is a lightweight, pure-Python data catalog and I/O utility designed for data pipeline development. Inspired by the Kedro data catalog, Trellis provides standard data I/O objects (`Dataset` subclasses) without requiring the installation of an entire orchestration framework.

## Core Architecture

### 1. `Dataset` (Abstract Base Class)
An abstract base class (ABC) defining the minimal contract for all data I/O objects. The base class is intentionally **format-agnostic and type-agnostic** -- it makes no assumptions about whether the underlying data is tabular, binary, text, or anything else. Each concrete subclass owns its data format, its I/O engine, and the types it accepts and returns.

- **Abstract Methods:** `.load()`, `.save()`, `.exists()`.
- **Common Constructor Args:** `path` (str), `versioned` (bool), `**kwargs` for format-specific options.
- **No shared I/O logic on the base class.** The base class does not provide conversion utilities, DataFrame handling, or any format-specific behavior. This keeps the door open for non-tabular dataset types (e.g., `PickleDataset`, `ImageDataset`, `TextDataset`) without fighting the base class.

#### 1a. Concrete Subclasses
Each file/storage format gets its own `Dataset` subclass responsible for its own I/O engine:

| Class | Format | Engine | Notes |
|---|---|---|---|
| `ParquetDataset` | `.parquet` | PyArrow + fsspec | Default/primary format. Columnar, partition-aware. |
| `CSVDataset` | `.csv` | PyArrow + fsspec | Handles encoding, delimiters, quoting options. |
| `JSONDataset` | `.json` / `.jsonl` | PyArrow + fsspec | Supports both single-object and newline-delimited JSON. |
| `DuckDBDataset` | DuckDB table | DuckDB | Reads/writes tables within a DuckDB database file. |
| `SQLDataset` | SQL table | SQLAlchemy / ibis | Reads/writes tables in external SQL databases. |

Each subclass fully owns its `.load()` and `.save()` behavior, including what types it accepts and returns. Tabular subclasses may use shared utility functions (see §1c) for DataFrame conversion, but this is opt-in, not inherited.

#### 1b. `.query()` Method (Per-Subclass, Optional)
`.query(sql)` is **not** part of the abstract base class contract. Concrete subclasses implement it where it makes sense and omit it where it doesn't. This follows standard Python duck typing -- no mixin or formal interface is needed.

- **File-based datasets** (`ParquetDataset`, `CSVDataset`, `JSONDataset`): Implement `.query(sql)` using an ephemeral DuckDB connection that registers the file and executes the query. Shared DuckDB setup logic lives in a standalone utility function (not a mixin) to avoid duplication.
- **`DuckDBDataset`**: Implements `.query(sql)` natively through its own DuckDB connection.
- **`SQLDataset`**: May implement `.query(sql)` through its SQL engine, or omit it if out of scope initially.

#### 1c. DataFrame Conversion Utilities (Standalone)
Tabular dataset subclasses need to convert between PyArrow Tables and other DataFrame types (pandas, polars, ibis). Rather than baking this into the base `Dataset` class (which would wrongly assume all datasets are tabular), these are **standalone utility functions** in a shared module (e.g., `trellis.conversion`):

```python
def convert_output(table: pa.Table, as_type: str) -> Any:
    """Convert a PyArrow Table to the requested DataFrame type."""
    # Handles: "pyarrow" (no-op), "pandas", "polars", "ibis"

def normalize_input(data: Any) -> pa.Table:
    """Convert any supported DataFrame type to a PyArrow Table."""
```

Tabular subclasses (`ParquetDataset`, `CSVDataset`, `JSONDataset`, etc.) call these functions in their own `.load()` and `.save()` implementations. Non-tabular subclasses simply ignore them. This keeps the base class clean while still avoiding code duplication across tabular formats.

#### 1d. Factory / Registry Pattern
To support config-driven instantiation (needed for YAML catalog support and clean APIs), `Dataset` provides a class-level registry:

```python
# Registration (happens automatically via subclass __init_subclass__ or a decorator)
Dataset.register("parquet", ParquetDataset)
Dataset.register("csv", CSVDataset)

# Instantiation from config
ds = Dataset.create("parquet", path="s3://bucket/data.parquet")
# Equivalent to: ParquetDataset(path="s3://bucket/data.parquet")
```

This allows the `Catalog` to instantiate datasets from YAML config without hardcoding format-to-class mappings, and gives users a single entry point when they don't want to import specific subclasses.

### 2. `Catalog` (The Registry)
A pure-Python registry used to organize, retrieve, and enumerate datasets.
- Usage: `catalog.add("name", ParquetDataset(...))` or any `Dataset` subclass.
- Retrieval: `catalog.name` or `catalog["name"]`
- **Type:** The catalog stores `Dict[str, Dataset]` where `Dataset` is the abstract base type. Any concrete subclass is accepted. This means code that operates on catalog entries can program against the `Dataset` interface without knowing the specific format.

### 3. Metadata & Versioning (Hash-based)
Instead of timestamped folders, datasets will write a `metadata.json` (or `.yaml`) sidecar file alongside the data.
- This metadata will track properties similar to `python-pins`: file hash (e.g., SHA-256), schema information, row counts, creation time, and user-defined custom metadata.
- Hash checking will allow us to quickly determine if a dataset has fundamentally changed.

### 4. Incremental Execution Support
Datasets natively support partitioning or accept a `mode="append" | "overwrite"` argument upon saving.
- `.load()` will accept filters/partition keys so pipelines can load just the new subset of data to run an incremental pipeline step without rebuilding the entire dataframe.

### 5. Dependencies (Batteries Included)
Trellis defaults to installing the main modern data stack tools: `pandas`, `polars`, `pyarrow`, `ibis-framework`, `duckdb`, and `fsspec`. This makes the default `pip install trellis` extremely capable out-of-the-box.
- In the future, we'll support optional minimal installs.

---

## Roadmap / Phases

### ✅ Phase 1: Skeleton and Setup (Completed)
- Initialize the `uv` project.
- Add core dependencies (`pandas`, `polars`, `pyarrow`, `duckdb`, `fsspec`, `ibis-framework`).
- Define the barebones `Dataset` and `Catalog` interfaces in `src/trellis/`.
- Set up tests with `pytest` and pre-commit hooks (`ruff`, `trailing-whitespace`, etc.).

### 🔄 Phase 1.5: Architecture Refactor (Next)
Restructure the skeleton to reflect the abstract base class design before implementing any real I/O logic:
1.  **Refactor `Dataset` to ABC**: Convert `Dataset` into a minimal abstract base class with abstract methods for `load`, `save`, `exists`. Add the `create()` factory and subclass registry. No I/O or conversion logic on the base class.
2.  **Create concrete subclass stubs**: Add `ParquetDataset`, `CSVDataset`, `JSONDataset`, `DuckDBDataset`, `SQLDataset` as stub subclasses with `NotImplementedError` in their methods. File-based subclasses include a stub `.query()` method.
3.  **Update `Catalog` type hints**: Ensure `Catalog` references the abstract `Dataset` type and works with any subclass.
4.  **Update tests**: Adjust the existing skeleton test and add new tests verifying the ABC contract, factory pattern, and subclass registration.
5.  **Update `__init__.py` exports**: Export all public classes.

### 🚧 Phase 2: Core Logic Implementation
This phase brings the skeleton to life. It involves:
1.  **DataFrame Conversion Utilities**: Implement `convert_output` and `normalize_input` as standalone functions in `trellis.conversion`.
2.  **The I/O Engine (per subclass)**: Implement `ParquetDataset` first (PyArrow + fsspec), then `CSVDataset`, `JSONDataset`, `DuckDBDataset`, and `SQLDataset`. Tabular subclasses use the conversion utilities from step 1.
3.  **`.query()` implementations**: Wire up ephemeral DuckDB connections for file-based datasets (shared utility function), native querying for `DuckDBDataset`.
4.  **Metadata, Hashing, and Versioning**: Establishing how Trellis calculates hashes, tracks row counts, and registers versions via a `metadata.json` sidecar.

### 🔮 Phase 3 & Beyond (Future Scope)
- **Data Validation**: Enforcing schema validation on save/load using `pandera` or `pydantic`.
- **YAML Configuration**: Adding `.from_yaml()` and `.to_yaml()` methods to the `Catalog` to support non-Python users (e.g., R developers). The `Dataset.create()` factory enables this cleanly.
- **Lineage Metadata**: Letting datasets track their origin (without becoming a full orchestrator).
- **Pipeline Abstraction**: A dedicated design session to figure out what a lightweight "Pipeline" abstraction looks like natively within Trellis.
