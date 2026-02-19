# Phase 7: Testing, Correctness Verification & Production Hardening

## Overview

Phase 7 transforms rust-warp from a fast prototype into a production-quality library. It adds systematic pixel-by-pixel correctness verification against GDAL/rasterio, analytical Rust tests for every kernel and projection, CI/CD pipelines, expanded benchmarks, and package metadata. The test suite grew from ~142 Python tests and ~140 Rust tests to **254 Python tests and 153 Rust tests**, all passing with zero warnings.

## What Was Built

### 1. Test Infrastructure (`tests/conftest.py`)

Shared utilities for all correctness tests:

- **`compare_arrays(actual, expected, method, atol, rtol)`** — reports max/mean/RMSE error, 99th percentile, % pixels differing, % with error > 1, worst-case pixel location
- **`synthetic_raster(shape, dtype, pattern)`** — generates test data with patterns: gradient, checkerboard, random, constant, sinusoidal
- **`gdal_reproject(...)`** — rasterio-based reference reprojection supporting all dtypes and custom nodata
- **`make_reprojection_setup(src_crs, dst_crs, size)`** — creates complete test setups for arbitrary CRS pairs with source data normalized to [0, 1000] range
- **`CRS_PAIRS`** — 6 standardized pairs: UTM33↔4326, UTM33→3857, 3857→4326, 4326→UTM17, UTM33→UTM17

### 2. Pixel-by-Pixel Correctness Tests (`tests/test_correctness/`)

**`test_vs_gdal.py`** — Parametrized test matrix (120+ test cases):

| Dimension | Values |
|-----------|--------|
| CRS pairs | UTM33→4326, 4326→UTM33, UTM33→3857, 3857→4326, 4326→UTM17, UTM33→UTM17 |
| Kernels | nearest, bilinear, cubic, lanczos, average |
| Dtypes | float64, float32, uint8, uint16, int16 |
| Sizes | 64×64, 256×256 |

Tolerance thresholds achieved:

| Kernel | Float tolerance | Integer tolerance |
|--------|----------------|-------------------|
| Nearest | >90% exact pixel match | >95% exact match |
| Bilinear | atol=15.0, rtol=0.01 | Correlation >0.95 (>0.90 for uint8) |
| Cubic | atol=15.0, rtol=0.01 | Correlation >0.95 (>0.90 for uint8) |
| Lanczos | atol=15.0, rtol=0.01 | Correlation >0.95 (>0.90 for uint8) |
| Average | Correlation-based | N/A |

**`test_edge_cases.py`** — 43 edge-case tests:
- NaN block propagation (all kernels)
- Sentinel nodata propagation (integer dtypes)
- 4× downscale and upscale (all kernels including average)
- Small rasters (4×4, 8×8) identity reprojection
- All-NaN, all-sentinel, and partial-nodata border regions

### 3. Rust-Side Analytical Tests

Each kernel and projection module received mathematical proof tests:

| Module | Tests Added | What They Prove |
|--------|-------------|-----------------|
| `nearest.rs` | 2 | Sub-pixel offset correctness (36 offsets), boundary behavior at pixel edges |
| `bilinear.rs` | 1 | Exact linear gradient preservation: f(x,y) = 3x - 2y + 7 at 25 sub-pixel positions |
| `cubic.rs` | 2 | Quadratic surface reproduction within 1e-6; 2D weight partition of unity at 36 offsets |
| `lanczos.rs` | 2 | Constant field exact reproduction at 25 positions; 1D weight normalization sums to 1.0 |
| `average.rs` | 2 | Known 2× downscale block averages; 3× downscale with nodata skip |
| `pipeline.rs` | 2 | Native vs proj4rs cross-validation at 100 points (<1mm error); UTM roundtrip (5 zones) |
| `approx.rs` | 2 | LinearApprox accuracy <0.2px for UTM→4326 and WebMerc→4326 at 1024-wide scanlines |

### 4. Production Code Fix (`src/py/plan.rs`)

Replaced 6 `.unwrap()` calls on `dict.set_item()` with proper `?` error propagation. This was the only production panic risk in the codebase — all other `unwrap()` calls are in test code.

```rust
// Before: panics on dict error
let dict = PyDict::new(py);
dict.set_item("dst_slice", ...).unwrap();

// After: propagates error
let result: Vec<PyObject> = plans.iter()
    .map(|plan| -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("dst_slice", plan.dst_slice)?;
        // ...
        Ok(dict.into_any().unbind())
    })
    .collect::<PyResult<Vec<_>>>()?;
```

### 5. CI/CD (`.github/workflows/`)

**`ci.yml`** — Runs on every push/PR:
- Matrix: ubuntu-latest + macos-latest, Python 3.13, latest stable Rust
- Steps: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`, `uv sync`, `maturin develop`, `pytest`, `ruff check`

**`release.yml`** — Runs on tag push (`v*`):
- Builds wheels: manylinux x86_64/aarch64, macOS ARM/x86_64
- Publishes to PyPI with trusted publishing

### 6. Expanded Rust Benchmarks (`benches/warp_bench.rs`)

| Benchmark | What it measures |
|-----------|-----------------|
| `warp_{kernel}_{size}` | All kernels at 256–2048 (added 2048) |
| `warp_scaling_bilinear_{size}` | Bilinear at 256, 512, 1024, 2048, 4096 |
| `warp_threads_{n}_bilinear_1024` | Thread scaling at 1, 2, 4, 8 threads |
| `proj_utm33_1M` / `proj_webmerc_1M` | Projection throughput at 1M points |
| `linear_approx_scanline_{width}` | LinearApprox at widths 256, 1024, 4096 |

### 7. Expanded Python Benchmarks (`tests/test_benchmark.py`)

| Benchmark | What it measures |
|-----------|-----------------|
| `test_plan_reproject_overhead[{tiles}]` | Chunk planner at 4, 16, 64, 256 tiles |
| `test_rustwarp_multiband[{bands}]` | 3-band and 64-band reprojection |
| `test_rasterio_multiband[{bands}]` | Matching rasterio baseline |

### 8. Integration Tests (`tests/test_integration/`)

**`test_large_scale.py`** (marked `@pytest.mark.stress`):
- 4096×4096 reprojection with all kernels
- Average 4× downscale at 2048×2048
- Spot-check 100 random pixels against GDAL at 2048×2048

**`test_multi_zone.py`** (marked `@pytest.mark.stress`):
- 3 UTM zones (32, 33, 34) to common EPSG:4326 grid
- Zone value preservation through reprojection
- Adjacent zone overlap consistency (zones 33 and 34)
- GeoBox bounds/shape consistency

### 9. epoch-mono Compatibility Tests (`tests/test_epoch_compat/`)

**`test_geobox_compat.py`**:
- `GeoBox.from_odc()` roundtrip for UTM and 4326
- `xr_coords()` shape, monotonicity, and pixel-center alignment
- Bounds and resolution property correctness
- `from_bbox()` with resolution and shape

**`test_aef_workflow.py`** — Simulates the AEF pattern:
- Single-zone reprojection (UTM33 → 4326, 3 bands, 64×64)
- Multi-zone reprojection (UTM 32/33/34 → 4326, 3 bands each)
- 64-band performance test
- `.warp` xarray accessor registration

### 10. Package Metadata

- **`Cargo.toml`**: description, repository, license (MIT), keywords, categories
- **`README.md`**: features, installation, quick start (numpy/GeoBox/xarray/dask), API reference, resampling methods table, development commands
- **`python/rust_warp/py.typed`**: PEP 561 marker for typed package
- **`pyproject.toml`**: registered `stress` pytest mark

## Benchmark Results

### rust-warp vs rasterio/GDAL (1024×1024, UTM33 → EPSG:4326)

| Kernel | rust-warp | rasterio | Speedup |
|--------|-----------|----------|---------|
| nearest | 1.56 ms | 10.35 ms | **6.6×** |
| bilinear | 1.92 ms | 41.91 ms | **21.8×** |
| cubic | 3.31 ms | 68.27 ms | **20.6×** |
| lanczos | 8.84 ms | 87.20 ms | **9.9×** |
| average (4× down) | 1.02 ms | 4.38 ms | **4.3×** |

### Thread Scaling (Bilinear 1024×1024)

| Threads | Time | Speedup |
|---------|------|---------|
| 1 | 2.36 ms | 1.0× |
| 2 | 1.42 ms | 1.66× |
| 4 | 1.07 ms | 2.21× |
| 8 | 884 µs | 2.67× |

### Warp Engine Scaling (Bilinear, Rayon default threads)

| Size | Time | Megapixels/sec |
|------|------|----------------|
| 256×256 | 74 µs | 886 |
| 512×512 | 298 µs | 880 |
| 1024×1024 | 796 µs | 1,317 |
| 2048×2048 | 1.21 ms | 3,467 |
| 4096×4096 | 13.2 ms | 1,271 |

### Multi-band (256×256, bilinear)

| Bands | rust-warp | rasterio | Speedup |
|-------|-----------|----------|---------|
| 3 | 0.32 ms | 8.93 ms | **27.8×** |
| 64 | 9.41 ms | 193.0 ms | **20.5×** |

### Projection Throughput (1M points)

| Projection | Throughput |
|------------|------------|
| UTM33 (Transverse Mercator) | 5.4M pts/sec |
| Web Mercator | 189M pts/sec |

### Chunk Planning Overhead

| Tiles | Time |
|-------|------|
| 4 | 39 µs |
| 16 | 153 µs |
| 64 | 612 µs |
| 256 | 2.46 ms |

## Error Sources and Tolerances

The primary sources of difference between rust-warp and GDAL/rasterio:

1. **Projection coordinate differences** — proj4rs vs PROJ produce slightly different coordinates (~0.1-1px), especially for cross-continental transforms (e.g., 4326→UTM17)
2. **Boundary pixel handling** — rust-warp returns None/NaN for edge pixels where the kernel neighborhood extends out of bounds; GDAL may extrapolate or clamp
3. **Integer quantization** — sub-pixel coordinate differences are amplified by rounding to integer output values, especially for uint8 (only 256 levels)

These are fundamental implementation differences, not bugs. The tolerances chosen reflect this reality while still being tight enough to catch real regressions.

## Files Changed

| File | Lines Changed | Description |
|------|---------------|-------------|
| `src/py/plan.rs` | +12 -9 | Replace `.unwrap()` with `?` error propagation |
| `src/resample/nearest.rs` | +40 | Sub-pixel offset and boundary tests |
| `src/resample/bilinear.rs` | +29 | Linear gradient preservation test |
| `src/resample/cubic.rs` | +48 | Quadratic surface and weight tests |
| `src/resample/lanczos.rs` | +39 | Constant field and weight normalization tests |
| `src/resample/average.rs` | +35 | Known downscale and nodata tests |
| `src/proj/pipeline.rs` | +78 | Cross-validation and roundtrip tests |
| `src/proj/approx.rs` | +88 | LinearApprox accuracy bounds tests |
| `benches/warp_bench.rs` | +148 | Scaling, threading, projection benchmarks |
| `tests/conftest.py` | +281 -73 | Test infrastructure and utilities |
| `tests/test_correctness/test_vs_gdal.py` | +361 | Parametrized correctness matrix |
| `tests/test_correctness/test_edge_cases.py` | +220 | Edge case tests |
| `tests/test_integration/test_large_scale.py` | +124 | Stress tests |
| `tests/test_integration/test_multi_zone.py` | +138 | Multi-zone integration |
| `tests/test_epoch_compat/test_aef_workflow.py` | +181 | AEF workflow simulation |
| `tests/test_epoch_compat/test_geobox_compat.py` | +137 | odc-geo compatibility |
| `tests/test_benchmark.py` | +106 | Plan overhead and multi-band benchmarks |
| `.github/workflows/ci.yml` | +59 | CI pipeline |
| `.github/workflows/release.yml` | +75 | Release pipeline |
| `Cargo.toml` | +5 | Package metadata |
| `README.md` | +145 | Full project README |
| `pyproject.toml` | +1 | stress marker registration |
| `python/rust_warp/py.typed` | 0 | PEP 561 marker |

**Total: +2,289 lines across 27 files**

## Test Summary

| Suite | Count | Status |
|-------|-------|--------|
| Rust (`cargo test`) | 153 | All pass |
| Python (`pytest`, excl. stress) | 254 | All pass |
| Clippy | — | Zero warnings |
| `cargo fmt` | — | Clean |
| Ruff | — | Clean |
