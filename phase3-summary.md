# Phase 3: Complete Python Bindings via PyO3

## Overview

Phase 3 completes the Python binding layer for rust-warp. The core `reproject_array()` binding previously only handled f64 arrays. This phase adds multi-dtype dispatch (f32, f64, u8, u16, i16), a `transform_points()` function for batch CRS coordinate transformation, and a `plan_reproject()` stub for Phase 4's chunk planner.

## What Was Built

### 1. Multi-dtype `reproject_array` (`src/py/reproject.rs`)

Replaced the f64-only implementation with a runtime dtype-dispatching version:

- Accepts `PyUntypedArray` and inspects dtype at runtime via `is_equiv_to()`
- Dispatches to a generic `reproject_typed::<T>()` helper for each supported type
- Supported dtypes: `float32`, `float64`, `uint8`, `uint16`, `int16`
- Output dtype always matches input dtype
- Nodata handling per dtype:
  - Float types: default fill is `NaN`
  - Integer types: default fill is `0` (`T::default()`); explicit nodata cast via `NumCast`
- Returns `PyObject` (since output dtype varies)
- GIL released during computation via `py.allow_threads()`

### 2. `transform_points` (`src/py/transform.rs`)

New binding for batch CRS coordinate transformation:

```python
x_out, y_out = transform_points(x, y, src_crs, dst_crs)
```

- Accepts 1D numpy arrays of x/y coordinates
- Forward transform (src→dst) achieved by swapping CRS args to Pipeline: `Pipeline::new(dst_crs, src_crs)` then calling `transform_inv_batch`
- Validates matching array lengths
- GIL released during the batch transform

### 3. `plan_reproject` stub (`src/py/plan.rs`)

Stub binding that establishes the function signature for Phase 4:

```python
tasks = plan_reproject(src_crs, src_transform, src_shape,
                       dst_crs, dst_transform, dst_shape,
                       dst_chunks=(256, 256), resampling="bilinear")
```

Currently returns an empty list. Phase 4 will implement the real chunk planner that maps destination chunks to source ROIs.

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `src/py/reproject.rs` | Modified | Multi-dtype dispatch via `PyUntypedArray` + `reproject_typed::<T>()` |
| `src/py/transform.rs` | Created | `transform_points` batch coordinate transformation |
| `src/py/plan.rs` | Created | `plan_reproject` stub for Phase 4 |
| `src/py/mod.rs` | Modified | Registered `transform` and `plan` modules |
| `python/rust_warp/__init__.py` | Modified | Exports `transform_points` and `plan_reproject` |
| `python/rust_warp/_rust.pyi` | Modified | Type stubs for all new/updated functions |
| `tests/test_warp.py` | Modified | Added `TestMultiDtype` (8 tests), `TestPlanReproject` (2 tests), unsupported dtype test |
| `tests/test_transform.py` | Created | Round-trip, pyproj comparison, and edge case tests (8 tests) |

## Test Results

- **128 Rust unit tests** — all pass (`cargo test`)
- **34 Python tests** — all pass (`uv run pytest tests/test_warp.py tests/test_transform.py -x -v`)
- **0 clippy warnings** (`cargo clippy --all-targets -- -D warnings`)
- **Formatting clean** (`cargo fmt --check`)

### New Python Tests

| Test Class | Count | Description |
|------------|-------|-------------|
| `TestMultiDtype` | 8 | Identity reprojection per dtype, f32/f64 comparison, integer nodata, float32 NaN fill |
| `TestPlanReproject` | 2 | Stub callable, returns empty list with/without chunks |
| `TestErrorHandling` | 1 | Unsupported dtype raises error |
| `TestTransformRoundTrip` | 2 | 4326↔UTM33 and 4326↔3857 round-trips |
| `TestTransformVsPyproj` | 1 | Comparison against pyproj (1m tolerance) |
| `TestTransformEdgeCases` | 5 | Empty arrays, single point, mismatched lengths, invalid CRS, same-CRS identity |

## Design Decisions

1. **Runtime dtype dispatch over compile-time generics** — Python arrays don't carry type info at the Rust function boundary. Using `PyUntypedArray` + `is_equiv_to` is the idiomatic rust-numpy pattern.

2. **CRS swap for forward transform** — Pipeline only exposes `transform_inv` (dst→src). Rather than adding new Pipeline methods, `transform_points` creates `Pipeline::new(dst, src)` and calls `transform_inv_batch` to achieve src→dst. Simpler, no Pipeline changes needed.

3. **NaN vs 0 fill for integer types** — Float types naturally use NaN for missing data. Integer types have no NaN equivalent, so they default to 0 (`T::default()`). Users can provide explicit nodata sentinels (e.g., 255 for u8, -9999 for i16).
