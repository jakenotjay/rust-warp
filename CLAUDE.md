# rust-warp

High-performance raster reprojection engine in Rust with xarray/dask integration. A GDAL-free alternative for chunked raster reprojection.

## Architecture

**Mixed Rust + Python project** using PyO3 and maturin.

- `src/` — Rust source (compiled into `rust_warp._rust` Python extension)
- `python/rust_warp/` — Python source (high-level API, dask/xarray integration)
- `tests/` — Python test suite
- `benches/` — Rust benchmarks (Criterion)

### Rust modules

| Module | Purpose |
|--------|---------|
| `src/lib.rs` | PyO3 module entry point |
| `src/error.rs` | Error types (`WarpError`, `ProjError`, `PlanError`) |
| `src/affine.rs` | 6-parameter affine geotransform |
| `src/proj/` | Map projection math (forward/inverse transforms) |
| `src/resample/` | Resampling kernels (nearest, bilinear, cubic, lanczos) |
| `src/warp/` | Inverse-mapping warp engine |
| `src/chunk/` | Chunk planner for dask tile mapping |
| `src/py/` | PyO3 binding layer |

### Python modules

| Module | Purpose |
|--------|---------|
| `python/rust_warp/__init__.py` | Public API |
| `python/rust_warp/_rust.pyi` | Type stubs for the Rust extension |
| `python/rust_warp/geobox.py` | GeoBox class (CRS + affine + shape) |
| `python/rust_warp/reproject.py` | High-level `reproject()` dispatcher |
| `python/rust_warp/dask_graph.py` | Dask graph builder for lazy reprojection |
| `python/rust_warp/xarray_accessor.py` | `.warp` xarray accessor |

## Build & Development Commands

```bash
# First-time setup (creates venv, installs deps, compiles Rust)
uv sync --all-extras

# After editing Rust code — rebuild extension
uv run maturin develop --release   # optimised
uv run maturin develop             # debug (faster compile)

# Run Rust tests
cargo test

# Run Python tests
uv run pytest tests/ -x -v

# Run Rust benchmarks
cargo bench

# Lint everything
cargo clippy --all-targets -- -D warnings
cargo fmt --check
uv run ruff check python/ tests/

# Build a release wheel
uv run maturin build --release
```

## Implementation Phases

The full plan is in `rust-warp-implementation-plan.md`. Summary:

1. **Phase 1** — Projection math (affine, TM, Mercator, Lambert, Albers, etc.)
2. **Phase 2** — Warp engine + resampling kernels
3. **Phase 3** — Python bindings via PyO3
4. **Phase 4** — Chunk planner + Dask integration
5. **Phase 5** — Xarray accessor + high-level API
6. **Phase 6** — SIMD optimisation + Rayon parallelism
7. **Phase 7** — Production hardening

## Coding Conventions

### Rust
- Every `.rs` file must have a `#[cfg(test)] mod tests` block
- Zero warnings under `cargo clippy -- -D warnings`
- Use `thiserror` for error types
- Generic over element type using `T: Copy + NumCast + PartialOrd + Default + Send`
- Release GIL for all computation via `py.allow_threads()`

### Python
- Ruff for linting (line-length 100, target py313)
- Type stubs in `_rust.pyi` for all Rust-exposed functions
- Comparison tests against pyproj/GDAL/odc-geo for correctness

### Testing
- Rust: `cargo test` (unit tests inline in each module)
- Python: `uv run pytest tests/ -x -v`
- Correctness: pixel-by-pixel comparison against GDAL output
- Performance: pytest-benchmark + Criterion

## Key Dependencies

### Rust
- `pyo3` — Python bindings
- `numpy` (rust-numpy) — PyO3 ↔ ndarray bridge
- `ndarray` — N-dimensional arrays
- `rayon` — Data parallelism
- `proj4rs` — Pure-Rust PROJ.4 implementation
- `thiserror` — Error types

### Python
- `numpy` — Required
- `xarray`, `dask`, `pyproj` — Optional (xarray extra)
- `rasterio`, `odc-geo`, `rioxarray` — Dev dependencies (comparison benchmarks)
