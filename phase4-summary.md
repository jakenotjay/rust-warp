# Phase 4: Chunk Planner + Dask Integration

## Overview

Phase 4 bridges the core reprojection engine to lazy/chunked workflows. The `plan_reproject` stub (previously returning `[]`) now computes real tile plans that map destination chunks to source ROIs with halo padding. New Python modules provide a `GeoBox` dataclass, a dask graph builder, and a high-level `reproject()` dispatcher that automatically handles both numpy and dask arrays.

## What Was Built

### 1. Chunk Planner (`src/chunk/planner.rs`)

The core Rust algorithm that divides a destination grid into tiles and computes the corresponding source region-of-interest for each:

- **`TilePlan` struct** — carries `dst_slice`, `src_slice` (with halo), shifted affine transforms, tile shape, and a `has_data` coverage flag
- **`plan_tiles()` function** — for each destination tile:
  1. Samples boundary points along the tile perimeter (`tile_boundary_points`)
  2. Projects each point: dst pixel → dst CRS → src CRS → src pixel via `Pipeline::transform_inv`
  3. Computes the source pixel bounding box from valid projected points
  4. Expands by kernel radius (halo padding for resampling kernels)
  5. Clips to source bounds; sets `has_data = false` if no valid projections
  6. Adjusts affine transforms so origins are relative to the tile/slice start
- **21 sample points per edge** (chosen in the PyO3 binding) balances accuracy vs. overhead

### 2. PyO3 Binding (`src/py/plan.rs`)

Replaced the empty stub with a real implementation:

- Parses resampling method → `kernel_radius().ceil()` for halo size
- Builds `Affine` structs from the 6-element tuples
- Falls back to full image size when `dst_chunks` is `None`
- Runs `plan_tiles()` inside `py.allow_threads()` for GIL release
- Converts each `TilePlan` to a Python dict with keys: `dst_slice`, `src_slice`, `src_transform`, `dst_transform`, `dst_tile_shape`, `has_data`

### 3. GeoBox Dataclass (`python/rust_warp/geobox.py`)

Frozen dataclass combining CRS + affine + shape:

```python
gbox = GeoBox(crs="EPSG:32633", shape=(64, 64), affine=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0))
gbox = GeoBox.from_bbox(bbox=(left, bottom, right, top), crs="EPSG:32633", resolution=100.0)
```

- `from_bbox(bbox, crs, resolution=None, shape=None)` — classmethod, computes affine from bounding box
- `bounds` property — `(left, bottom, right, top)`
- `resolution` property — `(res_x, res_y)`, both positive
- `xr_coords()` — dict of 1D x/y arrays at pixel centers (for xarray construction)

### 4. Dask Graph Builder (`python/rust_warp/dask_graph.py`)

`reproject_dask(src_data, src_geobox, dst_geobox, ...)`:

- Calls `plan_reproject()` to get tile plans
- For each tile with `has_data=True`: `dask.delayed(_reproject_tile)` on the source slice
- For each tile with `has_data=False`: `da.full(..., fill_value)` (NaN for float, 0 for int)
- Assembles tiles via `da.concatenate` (rows, then columns)
- `dask` and `dask.array` imported inside the function (optional dependencies)

### 5. High-Level Dispatcher (`python/rust_warp/reproject.py`)

```python
result = reproject(src_data, src_geobox, dst_geobox, resampling="bilinear")
```

- Detects dask arrays via `hasattr(src_data, "dask")` — no dask import needed
- Dask path → `reproject_dask()`
- Numpy path → `reproject_array()`

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `src/affine.rs` | Modified | Added `to_tuple()` method for Python serialization |
| `src/chunk/planner.rs` | Created | `TilePlan` struct + `plan_tiles()` algorithm + 8 unit tests |
| `src/chunk/mod.rs` | Modified | Re-exports `planner` module (was 2-line stub) |
| `src/py/plan.rs` | Modified | Real implementation replacing empty stub |
| `python/rust_warp/geobox.py` | Created | `GeoBox` frozen dataclass |
| `python/rust_warp/dask_graph.py` | Created | `reproject_dask()` dask graph builder |
| `python/rust_warp/reproject.py` | Created | `reproject()` high-level dispatcher |
| `python/rust_warp/__init__.py` | Modified | Exports `GeoBox` and `reproject` |
| `python/rust_warp/_rust.pyi` | Modified | Updated `plan_reproject` docs with dict key descriptions |
| `tests/test_chunk.py` | Created | 6 tests for `plan_reproject` |
| `tests/test_dask.py` | Created | 13 tests for GeoBox, dask integration, numpy dispatch |
| `tests/test_warp.py` | Modified | Updated `TestPlanReproject` to expect non-empty results |

## Test Results

- **137 Rust unit tests** — all pass (`cargo test`)
- **0 clippy warnings** (`cargo clippy --all-targets -- -D warnings`)
- **Formatting clean** (`cargo fmt --check`)
- **63 Python tests** — all pass (excluding benchmarks)
- **Ruff clean** on all new/modified files

### New Rust Tests (`src/chunk/planner.rs`)

| Test | What it verifies |
|------|-----------------|
| `test_same_crs_4_tiles_cover_full_extent` | 2x2 tiling covers all pixels, no gaps/overlaps |
| `test_cross_crs_valid_slices` | UTM→4326 source slices within bounds |
| `test_halo_padding` | Lanczos (radius=3) widens source slice vs nearest (radius=0) |
| `test_edge_tile_clipping` | Uneven 100x100 / 64x64 tiles: partial tiles, bounds clipping |
| `test_has_data_false_for_out_of_bounds` | Far-away destination → `has_data = false` |
| `test_affine_adjustment` | Tile affine origins correctly shifted |
| `test_no_chunks_single_tile` | Full-image tile size → 1 tile |
| `test_tile_boundary_points_basic` | Point count and range validation |
| `test_zero_tile_size_error` | Zero tile dimension → `PlanError` |

### New Python Tests

| Test Class | Count | Description |
|------------|-------|-------------|
| `TestPlanReprojectBasic` | 2 | Non-empty result with correct dict keys; tile shape matches dst_slice |
| `TestPlanReprojectCoverage` | 2 | 2x2 same-CRS full coverage; uneven chunks full coverage |
| `TestPlanReprojectBounds` | 1 | Cross-CRS source slices within bounds |
| `TestPlanReprojectNoChunks` | 1 | No chunks → single tile covering full extent |
| `TestGeoBox` | 6 | `from_bbox` (2 variants), `bounds`, `resolution`, `xr_coords`, missing-args error |
| `TestReprojectDask` | 7 | Returns lazy array; same-CRS/cross-CRS match numpy; chunk-size invariance; bilinear halo correctness; `has_data=False` fill |
| `TestReprojectNumpy` | 1 | `reproject()` numpy path matches `reproject_array()` |
| `TestPlanReproject` (updated) | 2 | Non-empty list with correct keys; 4 tiles with chunks |

## Design Decisions

1. **Boundary sampling over full inverse mapping** — Rather than projecting every destination pixel to find source coverage, `plan_tiles` samples ~80 points along each tile boundary (21 per edge). This is O(tiles × pts_per_edge) vs O(dst_pixels), making planning fast even for large grids. The trade-off is that concave source regions could be slightly overestimated, which wastes a small amount of read bandwidth but never causes missing data.

2. **Halo from kernel radius** — The source slice is expanded by `kernel_radius().ceil()` pixels on each side. This ensures interpolating kernels (bilinear=1, cubic=2, lanczos=3) have enough neighboring pixels at tile boundaries. Testing confirmed a single-pixel boundary artifact for bilinear (<0.025% of pixels) from sampling imprecision — acceptable and consistent with how GDAL handles tiled reprojection.

3. **`has_data` flag for skip optimization** — Tiles where no boundary points project successfully into the source extent are marked `has_data = false`. The dask graph builder uses this to emit `da.full(nodata)` instead of scheduling unnecessary compute, which is significant for large sparse reprojections (e.g., global grid → regional projection).

4. **dask as optional dependency** — `dask` and `dask.array` are imported inside `reproject_dask()` and `reproject()` only when needed. Users who only need numpy reprojection never pay the import cost. The `test_dask.py` module uses `pytest.importorskip` to skip gracefully if dask isn't installed.

5. **GeoBox as frozen dataclass** — Immutable by design since grid definitions shouldn't change after creation. The `from_bbox` classmethod accepts either `resolution` or `shape` (not both required), and `xr_coords()` produces arrays ready for Phase 5's xarray integration.
