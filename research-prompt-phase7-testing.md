# Research Prompt: Testing, Correctness Verification, Monitoring & Integration Framework for rust-warp

## Context

rust-warp is a high-performance raster reprojection engine in Rust with PyO3 bindings and xarray/dask integration. Phases 1–6 are complete: projection math, warp engine, resampling kernels (nearest, bilinear, cubic, lanczos, average), Python bindings, chunk planner, dask graph builder, xarray `.warp` accessor, Rayon parallelism, and kernel optimisations. It is now **faster than GDAL/rasterio across all kernels** at 1024×1024.

The codebase lives at `/Users/jww/projects/rust-warp`. The CLAUDE.md at the repo root describes the full architecture and build commands.

The goal of this research is to design and specify a comprehensive testing, correctness verification, monitoring, and integration framework that will bring rust-warp to production quality.

---

## 1. Correctness Verification Against GDAL and Best-in-Class

**This is the highest priority.** We must prove pixel-level correctness against GDAL (the gold standard) and odc-geo/rasterio.

### 1.1 GDAL Reference Test Methodology

GDAL's warp autotest suite is at `/Users/jww/projects/gdal/autotest/alg/warp.py` (2314 lines, 98 test functions). Study it in detail. Key patterns to replicate:

- **Pixel-by-pixel comparison** using `gdaltest.compare_ds()` which computes max absolute difference across all pixels. Their standard tolerance is `maxdiff <= 1` for integer types.
- **Checksum regression** via `band.Checksum()` for quick pass/fail.
- **Reference rasters**: `autotest/gcore/data/byte.tif` (21×21 byte), `autotest/gcore/data/utmsmall.tif` (85×85 byte NAD27 UTM Zone 11N). These are MIT-licensed.
- **VRT-based reference outputs** for each kernel: `utmsmall_near.vrt`, `utmsmall_blinear.vrt`, `utmsmall_cubic.vrt`, `utmsmall_lanczos.vrt`, etc. in `autotest/alg/data/`.
- **CRS pairs tested**: EPSG:4326 ↔ EPSG:32611, EPSG:26711, EPSG:32632, EPSG:3857.
- **Data type coverage**: Byte, UInt16, Int16, UInt32, Int32, Float32, Float64.
- **Downsampling tests**: `test_warp_lanczos_downsize_50_75` (50:75 ratio), `test_warp_average_oversampling`.
- **Nodata**: `test_reproject_3()` (source nodata propagation with bilinear), `UNIFIED_SRC_NODATA`, percentage thresholds.
- **Edge cases**: Empty RGBA bands (`test_warp_12/13/14`), rotated affines via GCPs, multi-band nodata, alpha blending.

**Research tasks:**
- Read `/Users/jww/projects/gdal/autotest/alg/warp.py` thoroughly. Document every test function, what CRS pair it uses, what kernel, what data type, what tolerance.
- Read `/Users/jww/projects/gdal/autotest/utilities/test_gdalwarp_lib.py` (4864 lines, 178 functions) for the gdalwarp library tests.
- Read `/Users/jww/projects/gdal/autotest/pymod/gdaltest.py` for the `compare_ds()` implementation (around line 1230).
- Identify which GDAL test rasters in `autotest/gcore/data/` and `autotest/alg/data/` we should copy or regenerate.
- Design a test harness that: (a) generates a reference output using GDAL's Python bindings (`gdal.Warp()`), (b) generates rust-warp output, (c) compares pixel-by-pixel.

### 1.2 odc-geo Reference Test Methodology

odc-geo's test suite is at `/Users/jww/projects/odc-geo/tests/`. Study these files:

- `/Users/jww/projects/odc-geo/tests/test_warp.py` (71 lines) — Parametrized over CRS (EPSG:4326, 3577, 3857), resampling (nearest, bilinear, average, sum), and country geometries (AUS, NZL). Tests NaN handling: sets rows/columns to NaN and verifies preservation.
- `/Users/jww/projects/odc-geo/tests/test_xr_interop.py` (825 lines) — `test_xr_reproject()` parametrized over chunks (None, (-1,-1), (4,4)) and time dimensions.
- `/Users/jww/projects/odc-geo/tests/conftest.py` — Fixtures: `ocean_raster` (128×256 from rasterized ocean geometry), `country_raster_f32` (random values inside country polygon).
- `/Users/jww/projects/odc-geo/odc/geo/testutils.py` (243 lines) — `mkA()` (affine from rot/scale/shear/translation), `gen_test_image_xy()` (encodes x,y coords in pixel values), `approx_equal_geobox()` (tolerance-aware GeoBox comparison).
- Test data: `tests/data/au-3577.tif`, `au-3577-rotated.tif`, `au-gcp.tif`.

**Key observation**: odc-geo does NOT compare against GDAL pixel-by-pixel. They test self-consistency (same result with/without dask, NaN preservation). This is a gap we should fill.

**Research tasks:**
- Read the odc-geo test utilities and document reusable patterns.
- Design tests that compare rust-warp output against odc-geo's `xr_reproject()` output for the same inputs.
- Document the `rio_reproject()` and `rio_warp_affine()` functions in odc-geo's warp module (`/Users/jww/projects/odc-geo/odc/geo/warp.py`) — these are the functions that call rasterio internally.

### 1.3 License Analysis

- **GDAL**: MIT license (`/Users/jww/projects/gdal/LICENSE.TXT`). Test code and test data are MIT. We can freely copy/adapt test rasters and test patterns.
- **odc-geo**: Apache 2.0 (`/Users/jww/projects/odc-geo/LICENSE`). Test code and test utilities are Apache 2.0. We can replicate patterns and even copy test utilities with attribution.
- **rasterio**: BSD 3-clause. Test code reusable with attribution.

**Research task:** Confirm these licenses by reading the actual files. Document attribution requirements for any code/data we copy.

### 1.4 Correctness Test Matrix to Design

Design the full test matrix. For each cell, the test should:
1. Generate or load a source raster
2. Reproject with GDAL (`gdal.Warp()` or `rasterio.warp.reproject()`)
3. Reproject with rust-warp (`rust_warp.reproject_array()`)
4. Compare pixel-by-pixel, report max absolute error

| Dimension | Values |
|-----------|--------|
| **CRS pairs** | UTM33→4326, 4326→UTM33, UTM33→3857, 3857→4326, 4326→UTM17, UTM33→UTM17 |
| **Kernels** | nearest, bilinear, cubic, lanczos, average |
| **Dtypes** | uint8, uint16, int16, float32, float64 |
| **Sizes** | 64×64, 256×256, 1024×1024 |
| **Nodata** | None, NaN (float), sentinel value (integer) |
| **Edge cases** | Antimeridian crossing, polar regions, rotated affine, downscale 4x, upscale 4x |

Expected tolerances:
- Nearest: exact match (maxdiff = 0) for integer types
- Bilinear/cubic/lanczos: maxdiff ≤ 1 for integer types, relative error < 1e-4 for float types
- Average: maxdiff ≤ 1 for integer, relative error < 1e-3 for float

---

## 2. Rust-Side Correctness Tests

Design additional Rust unit tests (`#[cfg(test)]`) that verify kernel correctness without Python:

### 2.1 Resampling Kernel Reference Tests

For each kernel, create tests with known analytical solutions:
- **Nearest**: Integer grid, sub-pixel offsets → verify exact pixel selection.
- **Bilinear**: Linear gradient → verify exact interpolation (bilinear preserves linear functions).
- **Cubic**: Quadratic surface → verify exact interpolation (cubic preserves up to degree 3).
- **Lanczos**: Constant field → verify exact reproduction. Sinusoidal field → verify Nyquist properties.
- **Average**: Uniform field with known downscale ratio → verify exact average.

### 2.2 Projection Accuracy Tests

Compare rust-warp's native projections against proj4rs fallback at 1000+ random points:
- For each supported EPSG code, forward+inverse roundtrip error < 1mm.
- Cross-validate: native UTM vs proj4rs UTM, native WebMerc vs proj4rs WebMerc.

### 2.3 LinearApprox Accuracy Tests

Verify that LinearApprox output matches exact per-pixel projection within the tolerance (0.125 pixels) for:
- UTM→4326 (moderate curvature)
- WebMerc→4326 (high curvature near poles)
- 4326→Sinusoidal (non-conformal)

---

## 3. Monitoring & Benchmarking Framework

### 3.1 Rust Benchmarks (Criterion)

The existing `benches/warp_bench.rs` needs expansion. Design benchmarks that measure:

| Benchmark | What it measures |
|-----------|-----------------|
| `projection_throughput` | Points/sec for each native projection (forward + inverse) |
| `linear_approx_scanline` | Time per scanline for LinearApprox at various widths (256, 1024, 4096) |
| `resample_kernel_<method>` | Time per pixel for each kernel in isolation (no projection) |
| `warp_<method>_<size>` | End-to-end warp for each kernel at 256², 512², 1024², 2048², 4096² |
| `warp_scaling` | Same kernel at increasing sizes to measure Rayon scaling |
| `warp_thread_scaling` | Same workload with 1, 2, 4, 8, N threads (via `rayon::ThreadPoolBuilder`) |

Read the existing benchmark at `/Users/jww/projects/rust-warp/benches/warp_bench.rs` and design expansions.

### 3.2 Python Benchmarks (pytest-benchmark)

The existing `tests/test_benchmark.py` compares rust-warp vs rasterio. Design expansions:

| Benchmark | What it measures |
|-----------|-----------------|
| `reproject_array` vs `rasterio.warp.reproject` | Per-kernel, per-size comparison (already exists, verify completeness) |
| `reproject` (high-level) vs `odc.geo.xr.xr_reproject` | xarray-level comparison with chunked data |
| `plan_reproject` | Chunk planning overhead at various tile counts |
| `dask_graph_build` | Graph construction time for chunked reprojection |
| `dask_compute` | End-to-end dask compute vs odc-geo for chunked data |
| `multi_band` | 3-band and 64-band reprojection (AEF use case) |

### 3.3 Profiling Integration

Design a profiling harness that can:
- Break down warp time into: projection (LinearApprox), resampling (kernel), memory allocation, Rayon overhead.
- Use `std::time::Instant` instrumentation in the Rust code with a `#[cfg(feature = "profile")]` gate.
- Output a flamegraph-compatible trace or structured JSON.

### 3.4 Continuous Benchmarking

Design a CI benchmark that:
- Runs on every PR against main
- Compares against stored baseline numbers
- Flags regressions > 10%
- Uses `criterion`'s built-in comparison (`cargo bench -- --baseline main`)

---

## 4. Large-Scale & Cloud-Native Tests

### 4.1 VirtualiZarr Integration Tests

The epoch-mono codebase at `/Users/jww/epoch/epoch-mono` uses VirtualiZarr extensively for lazy COG access. The AEF loader (`packages/aef-loader/epoch/aef_loader/reader.py`) creates virtual zarr stores from COGs using `virtual-tiff`, then reprojects with `odc.geo.xr.xr_reproject()`.

**Research tasks:**
- Read `/Users/jww/epoch/epoch-mono/packages/aef-loader/epoch/aef_loader/reader.py` — understand the `VirtualTiffReader` class, how it creates DataTree by UTM zone, how it assigns CRS via `odc.geo.xr.assign_crs()`.
- Read `/Users/jww/epoch/epoch-mono/packages/aef-loader/epoch/aef_loader/utils.py` — study `reproject_datatree()` which calls `xr_reproject()` per zone.
- Design tests that replicate this workflow with rust-warp:
  1. Open a COG virtually (via virtualizarr or fsspec)
  2. Create source/destination GeoBox
  3. Reproject lazily with rust-warp's dask integration
  4. Verify output matches odc-geo's output

### 4.2 Network/Remote Data Tests

Design integration tests (marked `@pytest.mark.slow` or `@pytest.mark.network`) that:
- Open a remote COG from a public S3/GCS bucket (e.g., Sentinel-2 COGs on AWS, or AEF tiles on Source Cooperative)
- Reproject a subset (e.g., 1024×1024 window) using rust-warp
- Compare against rasterio's output for the same window
- Measure end-to-end latency (network + compute)

### 4.3 Large Raster Stress Tests

Design tests (marked `@pytest.mark.stress`) that:
- Generate 8192×8192 and 16384×16384 synthetic rasters
- Reproject with all kernels
- Verify memory usage stays bounded (no full-array materialization for dask paths)
- Verify correctness on spot-check pixels (not full pixel-by-pixel at this scale)
- Measure peak RSS and wall-clock time

### 4.4 Multi-Zone Reprojection Test

Replicate the AEF use case from epoch-mono:
- Create synthetic tiles in 3 different UTM zones (e.g., UTM32, UTM33, UTM34)
- Each tile is 512×512, int8 with 64 bands
- Reproject all tiles to a common EPSG:4326 GeoBox
- Combine with `combine_first()`
- Verify seamless zone boundaries (no gaps, no duplicate coverage)

---

## 5. epoch-mono Integration Analysis

### 5.1 Current odc-geo Usage in epoch-mono

The following functions in epoch-mono call odc-geo for reprojection. For each, document the exact call signature, inputs, and expected outputs so we can design drop-in replacement tests:

| File | Function | odc-geo call |
|------|----------|-------------|
| `packages/aef-loader/epoch/aef_loader/utils.py` | `reproject_datatree()` | `xr_reproject(zone_ds, target_geobox, resampling="nearest")` |
| `packages/aef-loader/epoch/aef_loader/utils.py` | `clip_to_geometry()` | `rasterio.features.geometry_mask()` + CRS transform |
| `packages/epoch-utils/epoch/epoch_utils/xr/zonal.py` | zonal stats | `spatial_dims()`, GeoBox operations |
| `packages/epoch-utils/epoch/epoch_utils/icechunk.py` | chunk grid setup | `GeoBox`, `xr_zeros()` |
| `apps/sco2api/.../inference_to_icechunk.py` | inference pipeline | `xr_reproject()` for covariate reprojection |

**Research tasks:**
- Read each file listed above.
- Document the exact GeoBox construction patterns (from_bbox, resolution, CRS).
- Document the data shapes and dtypes (AEF: 64 bands, int8, 8192×8192 per tile).
- Design a compatibility test: run the same reprojection with odc-geo and rust-warp, compare outputs.

### 5.2 GeoBox Compatibility

rust-warp has its own `GeoBox` class (`python/rust_warp/geobox.py`). Compare it against `odc.geo.GeoBox`:

| Feature | odc-geo GeoBox | rust-warp GeoBox | Gap? |
|---------|---------------|-----------------|------|
| `from_bbox()` | Yes | Yes | Check API parity |
| `from_geopolygon()` | Yes | ? | Likely missing |
| `zoom_to()` / `zoom_out()` | Yes | ? | Likely missing |
| `__getitem__` (slicing) | Yes | ? | Likely missing |
| `xr_coords()` | Yes | Yes | Check output format |
| `crs` property | CRS object | string | Different types |
| `affine` property | Affine object | tuple | Different types |

**Research task:** Read both GeoBox implementations and document the full API gap.

### 5.3 Drop-in Replacement Path

Design the migration path for epoch-mono to use rust-warp instead of odc-geo for reprojection:
1. What Python API changes are needed?
2. Can rust-warp accept an `odc.geo.GeoBox` directly? (The existing code in `python/rust_warp/geobox.py` has `from_odc()` and `as_geobox()` — verify these work.)
3. What's the expected speedup for the AEF workflow (64-band int8, 8192×8192, multi-zone merge)?

---

## 6. Test Infrastructure Design

### 6.1 Test Directory Structure

Propose the directory layout for the expanded test suite:

```
tests/
├── conftest.py                    # Shared fixtures, GDAL/rasterio helpers
├── test_warp.py                   # Existing basic tests (keep)
├── test_benchmark.py              # Existing benchmarks (keep)
├── test_correctness/
│   ├── conftest.py                # Reference raster generation fixtures
│   ├── test_vs_gdal.py            # Pixel-by-pixel vs GDAL for all kernels/dtypes/CRS
│   ├── test_vs_rasterio.py        # Pixel-by-pixel vs rasterio.warp.reproject
│   ├── test_vs_odc_geo.py         # Comparison vs odc-geo xr_reproject
│   └── test_edge_cases.py         # Antimeridian, polar, rotated, large scale
├── test_monitoring/
│   ├── test_profiling.py          # Per-phase timing breakdown
│   └── test_memory.py             # Peak RSS, allocation tracking
├── test_integration/
│   ├── test_virtualizarr.py       # Virtual COG → reproject workflow
│   ├── test_large_scale.py        # 8k×8k and 16k×16k stress tests
│   ├── test_multi_zone.py         # Multi-UTM-zone merge (AEF pattern)
│   └── test_network.py            # Remote COG reprojection (marked slow)
├── test_epoch_compat/
│   ├── test_geobox_compat.py      # rust-warp GeoBox vs odc-geo GeoBox
│   └── test_aef_workflow.py       # End-to-end AEF reproject pipeline
├── data/
│   ├── byte.tif                   # From GDAL (MIT) — 21×21 byte
│   ├── utmsmall.tif               # From GDAL (MIT) — 85×85 NAD27 UTM11
│   └── au-3577.tif                # From odc-geo (Apache 2.0) — Australian
└── reference/
    └── generate_reference.py      # Script to generate GDAL reference outputs
```

### 6.2 Fixture Design

Design pytest fixtures that:
- `gdal_reproject(src, src_crs, src_transform, dst_crs, dst_transform, dst_shape, resampling)` — wraps `gdal.Warp()` to produce reference output
- `rasterio_reproject(...)` — wraps `rasterio.warp.reproject()` to produce reference output
- `synthetic_raster(shape, dtype, pattern)` — generates test rasters with known patterns (gradient, checkerboard, random, constant)
- `compare_arrays(actual, expected, method)` — reports maxdiff, RMSE, % pixels differing, and asserts within tolerance

### 6.3 Reporting

Design a test report format that for each correctness test outputs:
- Max absolute error
- Mean absolute error
- RMSE
- % of pixels with error > 0
- % of pixels with error > 1
- Worst-case pixel location and values (actual vs expected)

---

## 7. Deliverables

Produce a detailed implementation plan (not code) with:

1. **Correctness test specifications** — exact test functions, inputs, expected tolerances, comparison methodology for every cell in the test matrix.
2. **Rust test specifications** — new `#[cfg(test)]` tests for each module, with analytical reference values.
3. **Monitoring framework design** — Criterion benchmark expansion, pytest-benchmark expansion, profiling instrumentation, CI integration.
4. **Large-scale test specifications** — virtualizarr workflow test, network test, stress test, multi-zone test.
5. **epoch-mono integration spec** — GeoBox API gap analysis, migration path, compatibility test design.
6. **Infrastructure** — directory layout, fixture design, reference data management, CI configuration.
7. **Priority ordering** — what to build first for maximum confidence in correctness.

Focus on specificity: name exact files, functions, CRS codes, raster sizes, tolerances, and comparison methods. Reference actual code in the GDAL, odc-geo, and epoch-mono codebases by file path.
