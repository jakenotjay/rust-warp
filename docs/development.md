# Development Guide

## Prerequisites

- **Rust** >= 1.75 (install via [rustup](https://rustup.rs/))
- **Python** >= 3.12 (managed via [uv](https://docs.astral.sh/uv/))
- **uv** >= 0.5 (Python package manager)
- **maturin** >= 1.7 (Rust-Python build backend, installed automatically by uv)

### macOS (Apple Silicon)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup component add clippy rustfmt

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Optional: GDAL/PROJ for comparison benchmarks (not needed for rust-warp itself)
brew install gdal proj
```

## First-Time Setup

```bash
git clone https://github.com/jww/rust-warp.git
cd rust-warp

# Create venv, install all deps (Python + Rust compilation)
uv sync --all-extras
```

This single command:
1. Creates a Python virtual environment
2. Installs all Python dependencies (numpy, xarray, dask, pyproj, rasterio, odc-geo, etc.)
3. Compiles the Rust extension via maturin
4. Installs the compiled extension into the venv

## Build Commands

```bash
# After editing Rust code — rebuild extension (optimized)
uv run maturin develop --release

# Debug build (faster compile, slower runtime — good for iteration)
uv run maturin develop

# Build a release wheel
uv run maturin build --release
```

## Testing

### Rust tests

```bash
cargo test
```

Every `.rs` file has inline `#[cfg(test)] mod tests` blocks. Tests cover:
- Projection forward/inverse roundtrips
- Projection accuracy against reference values
- Resampling kernel mathematical properties
- Warp engine correctness
- Chunk planner coverage and bounds

### Python tests

```bash
# Run all tests (excluding stress tests)
uv run pytest tests/ -x -v

# Run specific test file
uv run pytest tests/test_warp.py -x -v

# Run correctness tests
uv run pytest tests/test_correctness/ -x -v

# Run integration tests
uv run pytest tests/test_integration/ -x -v

# Run stress tests (large rasters, slow)
uv run pytest tests/ -m stress -v

# Run epoch compatibility tests
uv run pytest tests/test_epoch_compat/ -x -v
```

### Test structure

```
tests/
├── conftest.py                  Shared fixtures (gdal_reproject, compare_arrays, etc.)
├── test_warp.py                 Core warp tests (multi-dtype, error handling)
├── test_projections.py          Projection accuracy vs pyproj
├── test_transform.py            transform_points tests
├── test_chunk.py                Chunk planner tests
├── test_dask.py                 Dask integration tests
├── test_xarray.py               xarray accessor tests
├── test_benchmark.py            Performance benchmarks
├── test_dask_graph_bench.py     Dask graph build benchmarks
├── test_correctness/
│   ├── test_vs_gdal.py          Pixel-by-pixel vs GDAL (120+ parametrized cases)
│   ├── test_edge_cases.py       NaN, nodata, upscale, downscale
│   ├── test_kernel_isolation.py Kernel-only tests via reproject_with_grid
│   └── test_projection_accuracy.py
├── test_integration/
│   ├── test_large_scale.py      4096x4096 stress tests
│   └── test_multi_zone.py       Multi-UTM-zone merge
└── test_epoch_compat/
    ├── test_geobox_compat.py    GeoBox vs odc-geo compatibility
    └── test_aef_workflow.py     AEF-style multi-band workflow
```

## Benchmarks

### Rust benchmarks (Criterion)

```bash
cargo bench
```

Results are saved to `target/criterion/` with HTML reports. Benchmarks cover:
- Per-kernel warp at various sizes (256-4096)
- Bilinear scaling from 256 to 4096
- Thread scaling (1, 2, 4, 8 threads)
- Projection throughput (1M points)
- LinearApprox scanline performance

### Python benchmarks (pytest-benchmark)

```bash
uv run pytest tests/test_benchmark.py --benchmark-enable -v
```

Compares rust-warp vs rasterio across all kernels and sizes.

## Linting

```bash
# Rust
cargo clippy --all-targets -- -D warnings   # Zero warnings policy
cargo fmt --check                             # Formatting

# Python
uv run ruff check python/ tests/
```

## Coding Conventions

### Rust

- Every `.rs` file must have a `#[cfg(test)] mod tests` block
- Zero warnings under `cargo clippy -- -D warnings`
- Use `thiserror` for error types
- Generic over element type: `T: Copy + NumCast + PartialOrd + Default + Send + Sync`
- Release GIL for all computation via `py.allow_threads()`
- Use `?` for error propagation — no `.unwrap()` in production code

### Python

- Ruff for linting (line-length 100, target py313)
- Type stubs in `_rust.pyi` for all Rust-exposed functions
- Optional dependencies imported inside functions (dask, xarray, pyproj)
- Comparison tests against pyproj/GDAL/odc-geo for correctness

## CI/CD

### CI (`.github/workflows/ci.yml`)

Runs on every push and PR:
- `cargo fmt --check`
- `cargo clippy -- -D warnings`
- `cargo test`
- `uv sync --all-extras`
- `uv run maturin develop`
- `uv run pytest tests/ -x -v`
- `uv run ruff check python/ tests/`

Matrix: ubuntu-latest + macos-latest, Python 3.13, latest stable Rust.

### Release (`.github/workflows/release.yml`)

Runs on tag push (`v*`):
- Builds wheels for manylinux x86_64/aarch64 and macOS ARM/x86_64
- Publishes to PyPI with trusted publishing

## Project Architecture

See [architecture.md](architecture.md) for detailed documentation of the Rust and Python module structure, the warp pipeline, and how the layers interact.
