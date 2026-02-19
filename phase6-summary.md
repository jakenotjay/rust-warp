# Phase 6: SIMD Optimisation + Rayon Parallelism

## Overview

Phase 6 addresses the two root causes of rust-warp's performance gap versus GDAL/rasterio: sequential row processing and expensive transcendental math in resampling kernels. After these changes, rust-warp is **faster than rasterio across all kernels** on the 1024×1024 UTM→4326 benchmark.

## What Was Built

### 1. Rayon Parallel Row Loop (`src/warp/engine.rs`)

Replaced the sequential `for row in 0..dst_rows` loop with Rayon parallel iteration over output rows. Each row is independent — it reads from a shared `&ArrayView2<T>` and writes to a disjoint output row slice.

- `dst.axis_iter_mut(Axis(0)).into_par_iter().enumerate()` distributes rows across the thread pool
- Thread-local coordinate buffers (`src_cols_buf`, `src_rows_buf`) allocated once per row
- Added `Sync` bound to generic `T` (propagated to `reproject_typed` in `src/py/reproject.rs`)
- `Pipeline` is already `Send + Sync` — the `Projection` trait requires it, and `proj4rs::Proj` contains only plain data

### 2. Fast Lanczos Weight Function (`src/resample/lanczos.rs`)

Replaced transcendental `sin()` calls in the Lanczos-3 kernel with a polynomial approximation:

- **`sin_kernel(x)`** — degree-11 Taylor polynomial for `sin(x)` on `[0, π/2]`, max absolute error ~6e-8
- **`lanczos_weight(t)`** — computes `L(t) = 3·sin(πt)·sin(πt/3) / (π²t²)` using range reduction and `sin_kernel`
  - `sin(πt)` reduced via `floor(t)` sign flip and half-period symmetry
  - `sin(πt/3)` always in first half-period since `t/3 ∈ (0, 1)` for `t ∈ (0, 3)`
  - Combined formula avoids two separate divisions
- Original `sinc()` and `lanczos_weight_exact()` retained under `#[cfg(test)]` for validation
- Validated at 10k uniformly-spaced points: max error vs exact < 1e-6

### 3. Cubic Weight Precomputation (`src/resample/cubic.rs`)

Precomputed 1D weight arrays before the 2D convolution loop:

```rust
let wx: [f64; 4] = std::array::from_fn(|k| cubic_weight(dx - (k as f64 - 1.0)));
let wy: [f64; 4] = std::array::from_fn(|k| cubic_weight(dy - (k as f64 - 1.0)));
```

Reduces `cubic_weight()` calls from 16 (in the 4×4 loop) to 8 (4 for x + 4 for y). The inner loop now multiplies precomputed weights, which LLVM can auto-vectorize.

### 4. Average Kernel Row-Overlap Precomputation (`src/resample/average.rs`)

Precomputed y-overlap weights into a `Vec<f64>` before the inner x loop:

```rust
let oy_weights: Vec<f64> = (y_min..y_max)
    .map(|iy| { /* overlap computation */ })
    .collect();
```

Eliminates redundant `max()`/`min()` calls per inner-loop iteration.

### 5. `.cargo/config.toml`

Enables `target-cpu=native` for both x86_64 and aarch64, allowing LLVM to emit AVX2/NEON instructions in dev benchmarks. Not used for distributed wheels (maturin builds don't read `.cargo/config.toml` by default).

## Performance Results (1024×1024, UTM→4326)

| Kernel | Before | After | rasterio | Speedup | vs rasterio |
|--------|--------|-------|----------|---------|-------------|
| nearest | 96 ms | 1.7 ms | 10.7 ms | 56x | **6.3x faster** |
| bilinear | 178 ms | 1.7 ms | 42.5 ms | 105x | **25x faster** |
| cubic | 740 ms | 3.4 ms | 70.6 ms | 218x | **21x faster** |
| lanczos | 1972 ms | 8.2 ms | 84.7 ms | 240x | **10x faster** |
| average | 78 ms | 1.0 ms | 4.1 ms | 78x | **4x faster** |

The gains come from:
- **Rayon** — scales across all available cores (~10x on the test machine)
- **`target-cpu=native`** — enables NEON/AVX2 auto-vectorization (~2-3x for compute-bound kernels)
- **Polynomial sin** — eliminates transcendental function calls in Lanczos (~3x for that kernel)
- **Weight precomputation** — reduces redundant work in cubic and average kernels

## Files Changed

| File | Lines | Description |
|------|-------|-------------|
| `src/warp/engine.rs` | ~50 | Rayon parallel row loop, `Sync` bound, 2 new tests |
| `src/resample/lanczos.rs` | ~60 | `sin_kernel`, fast `lanczos_weight`, validation test |
| `src/resample/cubic.rs` | ~8 | Precomputed `wx`/`wy` arrays |
| `src/resample/average.rs` | ~8 | Precomputed `oy_weights` vector |
| `src/py/reproject.rs` | ~1 | Added `Sync` bound |
| `.cargo/config.toml` | 5 | `target-cpu=native` |

## Tests

- 140 Rust tests pass (`cargo test`)
- 142 Python tests pass (`uv run pytest tests/ -x -v`)
- Clippy clean (`cargo clippy --all-targets -- -D warnings`)

### New Tests

| Test | File | Purpose |
|------|------|---------|
| `test_pipeline_is_send_sync` | `engine.rs` | Static assertion that `Pipeline` is `Send + Sync` |
| `test_parallel_matches_sequential` | `engine.rs` | 128×128 identity reprojection matches for nearest + bilinear |
| `test_fast_lanczos_matches_exact` | `lanczos.rs` | Polynomial vs sinc at 10k points, error < 1e-6 |
