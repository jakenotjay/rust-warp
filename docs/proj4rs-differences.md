# proj4rs vs PROJ: Known Differences

rust-warp uses **proj4rs** — a pure-Rust port of PROJ.4 — for coordinate transformations, rather than the C PROJ library used by GDAL, rasterio, pyproj, and odc-geo. This document describes the known accuracy differences and their impact.

## Background

[PROJ](https://proj.org/) is the canonical C library for cartographic projections, maintained by OSGeo. It has evolved significantly beyond the original PROJ.4 series, with modern features like time-dependent transformations, datum grid shifts, and WKT2 CRS parsing.

[proj4rs](https://github.com/3liz/proj4rs) is a Rust port of the older PROJ.4 codebase. It implements the core projection math but does not include all of PROJ's modern features.

rust-warp also includes **native Rust projection implementations** for the most common CRSes (all UTM zones, Web Mercator, Equirectangular, Lambert, Albers, Stereographic, Sinusoidal). These match PROJ's output to sub-millimetre accuracy. The proj4rs fallback is used only for CRSes not covered by native implementations.

## Quantified Accuracy

The diagnostic test suite (`test_accuracy_diagnostic.py`) decomposes error into projection and kernel components across CRS pairs, kernels, and scale factors. All measurements use a synthetic gradient raster on a 0–1000 value scale.

### Projection Coordinate Error (pixels)

Measured as Euclidean distance between rust-warp and pyproj source pixel coordinates:

| CRS pair | 64×64 max | 64×64 mean | 256×256 max | Dominant axis |
|----------|-----------|------------|-------------|---------------|
| UTM33 → 4326 | 0.013 | 0.009 | 0.055 | row |
| 4326 → UTM33 | 0.038 | 0.023 | 0.045 | row |
| UTM33 → 3857 | 0.013 | 0.008 | 0.054 | row |
| 3857 → 4326 | **0.000** | **0.000** | **0.000** | exact |
| 4326 → UTM17 | 0.114 | 0.071 | 0.136 | col + row |
| UTM33 → UTM17 | 0.006 | 0.004 | 0.092 | row |

Key observations:
- **3857 → 4326 is exact** — the Mercator inverse matches PROJ perfectly
- **4326 → UTM17 is worst** (~0.14px at 256×256) — the cross-zone Transverse Mercator jump stresses the projection math the most
- **Error is overwhelmingly in the row (Y) axis** — column error is typically 100× smaller, pointing to a systematic offset in the Transverse Mercator northing calculation
- All pairs stay below the 0.15px threshold

### Error Decomposition (value units, 0–1000 scale)

Breaking total error into projection vs kernel components (pyproj coords + rust-warp kernels isolates kernel-only error):

| CRS pair | Kernel | Total max | Kernel-only max | Projection-only max |
|----------|--------|-----------|-----------------|---------------------|
| UTM33→4326 | bilinear | 2.27 | 0.80 | 2.23 |
| UTM33→4326 | cubic | 0.89 | 0.82 | 0.61 |
| UTM33→4326 | lanczos | 2.01 | 1.42 | 1.47 |
| 4326→UTM33 | bilinear | 4.57 | 0.39 | 4.50 |
| 4326→UTM33 | cubic | 1.67 | 0.37 | 1.67 |
| 4326→UTM33 | lanczos | 3.60 | 0.83 | 3.59 |
| UTM33→3857 | bilinear | 2.12 | 0.66 | 2.09 |
| UTM33→3857 | cubic | 0.72 | 0.67 | 0.44 |
| UTM33→3857 | lanczos | 1.49 | 1.14 | 1.03 |

**Projection error dominates.** Kernel-only error (with identical coordinates) is 2–10× smaller than total error. The kernel implementations are correct; the accuracy gap is almost entirely from proj4rs vs PROJ coordinate differences.

### Error by Scale Factor (same CRS, no projection involved)

| Scale | bilinear max | lanczos max |
|-------|-------------|-------------|
| 4× up | 0.000 | 0.000 |
| 2× up | 0.000 | 0.000 |
| 1× identity | 0.000 | 0.000 |
| 0.5× down | 13.93 | 8.74 |
| 0.25× down | 25.54 | 4.30 |

Upscaling and identity are **perfectly exact** (0.000 error). Downscaling with interpolating kernels diverges from GDAL at image borders — these kernels are not designed for downsampling. Use `average` for downscaling.

### Non-Square Pixel Error (UTM33 → 4326)

| Aspect ratio | bilinear max | lanczos max |
|-------------|-------------|-------------|
| 2:1 (wide) | 9.19 | 5.77 |
| 1:2 (tall) | 0.05 | 0.06 |
| 3:1 (wide) | 30.21 | 18.75 |
| 1:3 (tall) | 0.17 | 0.11 |

Wide pixels amplify the row-axis projection error through the wider pixel footprint. Tall pixels have near-zero error.

### Edge vs Interior Error (UTM33 → 4326, 64×64)

| Kernel | Interior max | Edge max | Interior mean | Edge mean |
|--------|-------------|----------|---------------|-----------|
| bilinear | 0.99 | 2.27 | 0.53 | 0.60 |
| cubic | 0.32 | 0.89 | 0.07 | 0.13 |
| lanczos | 0.57 | 2.01 | 0.23 | 0.41 |

Edge pixels show ~2× higher error due to kernel boundary handling differences.

### Sources of Inaccuracy (ranked)

1. **Transverse Mercator northing offset** — the row-axis coordinate error in proj4rs is the dominant source. Fixing this would close most of the gap with GDAL.
2. **Interpolating kernels used for downsampling** — bilinear/lanczos produce large border errors when downscaling (max 25 value units at 4× downscale). Use `average` for downsampling.
3. **Wide non-square pixels** — amplify projection error through the wider pixel footprint. 3:1 aspect ratio hits 30 value units max error.
4. **Kernel boundary handling** — small differences (~1.4 value units max) in edge pixel treatment. This is the noise floor.

## Accuracy at Large Scales

At larger scales (8K+ pixels, cross-continental transforms), accumulated floating-point differences become visible:

### 16384x16384 nearest-neighbor (int8, UTM to EPSG:3857)

**rust-warp vs odc-geo (which uses PROJ):**
- 99.9987% exact pixel match
- ~1,100 off-by-1 pixels per band
- ~1,500 off-by->1 pixels per band (max diff 37-58)

**rust-warp vs rasterio/GDAL (same transform):**
- 99.9939% exact pixel match
- ~5,400 off-by-1 pixels
- ~7,400 off-by->1 pixels (max diff 52)

### Why the differences occur

1. **Sub-pixel coordinate divergence** — proj4rs and PROJ compute slightly different source pixel coordinates for the same destination pixel. For interpolating kernels this causes small weight differences; for nearest-neighbor, it causes different pixel selection at boundaries where the coordinate is close to 0.5.

2. **Floating-point accumulation** — the linear approximation module interpolates between exactly-projected control points. Small differences in the control point values accumulate across long scanlines (16K+ pixels wide).

3. **Boundary pixel handling** — rust-warp returns NaN/nodata for edge pixels where the kernel neighbourhood extends outside source bounds. GDAL may extrapolate or clamp in some configurations.

4. **Integer quantisation** — for integer dtypes (especially uint8 with only 256 levels), sub-pixel coordinate differences are amplified by rounding. A 0.001-pixel source coordinate difference near a sharp edge can flip the output value entirely.

## Which CRSes are affected

| Category | CRSes | Expected Accuracy |
|----------|-------|-------------------|
| Native implementations | All UTM zones (326xx/327xx), EPSG:3857, EPSG:4326, Lambert, Albers, Stereographic, Sinusoidal | Sub-millimetre match with PROJ |
| proj4rs with common datums | Most EPSG codes with WGS84 or NAD83 datum | <1mm for moderate extents |
| proj4rs with datum shifts | Codes requiring grid-based datum transforms | May differ significantly — proj4rs does not support NTv2 grid shifts |
| WKT-only CRSes | CRSes defined only by WKT (no EPSG code or PROJ string) | Requires pyproj to extract EPSG code; falls back to WKT parsing which proj4rs may not support |

## Practical Impact

For most scientific and geospatial workflows, the differences are **not meaningful**:

- Temperature, precipitation, NDVI, and other continuous fields: bilinear/cubic interpolation smooths out sub-pixel coordinate differences. RMSE between rust-warp and GDAL output is typically <0.01% of the data range.
- Classification/categorical data with nearest-neighbor: the ~0.001-0.01% of boundary pixels that differ are at sharp class boundaries where the "correct" answer depends on the exact sub-pixel source coordinate.
- The differences are comparable to those between GDAL and odc-geo themselves.

## Mitigations

1. **Use `reproject_with_grid` + `transform_grid` for testing** — these separate the projection step from the resampling step, making it easy to isolate whether a difference comes from the projection or the kernel.

2. **Prefer EPSG codes over PROJ strings** — EPSG codes route to native implementations more reliably than PROJ strings.

3. **For CRSes requiring datum grid shifts**, consider pre-transforming coordinates with pyproj and using `reproject_with_grid` with the pre-computed grids.

## Future Work

- Expanding native projection coverage to reduce reliance on proj4rs
- Potentially integrating PROJ directly via FFI for users who need exact GDAL compatibility (at the cost of a C dependency)

## PROJ C FFI (Future Option)

The [`proj`](https://crates.io/crates/proj) crate provides safe Rust bindings to the C PROJ library (libproj). This could be integrated as a third pipeline variant alongside native and proj4rs.

### Design

- New `Pipeline::ProjC` variant, feature-gated behind a `proj-sys` Cargo feature
- Requires libproj installed on the system (via conda, apt, brew, or vcpkg)
- Would be enabled with `cargo build --features proj-sys`

### Performance

FFI overhead is ~50-100ns per call, which is negligible when combined with rust-warp's `LinearApprox` module. The linear approximation only makes 5-15 exact projection calls per scanline (at control points), then interpolates between them. The FFI cost is amortised across thousands of pixels per scanline.

### Use Cases

- **NTv2 grid shifts** — datum transforms requiring grid files (e.g. NAD27→NAD83)
- **WKT2 parsing** — CRS definitions only available as WKT2 strings
- **Time-dependent transforms** — plate motion models, epoch-based transforms
- **Exact GDAL compatibility** — bit-for-bit match with GDAL/rasterio output
