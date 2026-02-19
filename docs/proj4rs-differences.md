# proj4rs vs PROJ: Known Differences

rust-warp uses **proj4rs** — a pure-Rust port of PROJ.4 — for coordinate transformations, rather than the C PROJ library used by GDAL, rasterio, pyproj, and odc-geo. This document describes the known accuracy differences and their impact.

## Background

[PROJ](https://proj.org/) is the canonical C library for cartographic projections, maintained by OSGeo. It has evolved significantly beyond the original PROJ.4 series, with modern features like time-dependent transformations, datum grid shifts, and WKT2 CRS parsing.

[proj4rs](https://github.com/3liz/proj4rs) is a Rust port of the older PROJ.4 codebase. It implements the core projection math but does not include all of PROJ's modern features.

rust-warp also includes **native Rust projection implementations** for the most common CRSes (all UTM zones, Web Mercator, Equirectangular, Lambert, Albers, Stereographic, Sinusoidal). These match PROJ's output to sub-millimetre accuracy. The proj4rs fallback is used only for CRSes not covered by native implementations.

## Accuracy at Small Scales

At typical working sizes (up to ~4096x4096 pixels), the coordinate differences between proj4rs and PROJ are negligible:

- **Native projections** match pyproj/PROJ to **<1mm at the equator** for all supported CRSes
- **proj4rs fallback** matches PROJ to **<1mm** for most common transforms
- **Pixel-by-pixel GDAL comparison tests** pass at 64x64 and 256x256 with exact match (`atol=0`) for nearest-neighbor resampling

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

3. **Boundary pixel handling** — rust-warp returns NaN/nodata for edge pixels where the kernel neighborhood extends outside source bounds. GDAL may extrapolate or clamp in some configurations.

4. **Integer quantization** — for integer dtypes (especially uint8 with only 256 levels), sub-pixel coordinate differences are amplified by rounding. A 0.001-pixel source coordinate difference near a sharp edge can flip the output value entirely.

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
