# Architecture

rust-warp is a mixed Rust + Python project. The Rust core handles all computation-intensive work (projection math, resampling, the warp loop), while Python provides the high-level API, dask integration, and xarray accessor.

## Project Layout

```
rust-warp/
├── src/                     Rust source (compiled to rust_warp._rust)
│   ├── lib.rs               PyO3 module entry point
│   ├── error.rs             Error types (WarpError, ProjError, PlanError)
│   ├── affine.rs            6-parameter affine geotransform
│   ├── proj/                Projection math
│   │   ├── mod.rs           Projection trait + CRS dispatch
│   │   ├── crs.rs           CRS parsing and endpoint resolution
│   │   ├── ellipsoid.rs     Ellipsoid constants (WGS84, GRS80, etc.)
│   │   ├── common.rs        Shared helpers (tsfn, msfn, qsfn, etc.)
│   │   ├── pipeline.rs      CRS_A -> geodetic -> CRS_B chain
│   │   ├── approx.rs        Linear approximation for fast transforms
│   │   ├── transverse_mercator.rs
│   │   ├── mercator.rs
│   │   ├── equirectangular.rs
│   │   ├── lambert_conformal.rs
│   │   ├── albers_equal_area.rs
│   │   ├── stereographic.rs
│   │   └── sinusoidal.rs
│   ├── resample/            Resampling kernels
│   │   ├── mod.rs           ResamplingMethod enum + dispatch
│   │   ├── nearest.rs       Nearest neighbour
│   │   ├── bilinear.rs      2x2 bilinear interpolation
│   │   ├── cubic.rs         4x4 cubic convolution (Keys, a=-0.5)
│   │   ├── lanczos.rs       6x6 Lanczos-3 with polynomial sin approx
│   │   └── average.rs       Area-weighted average for downsampling
│   ├── warp/                Warp engine
│   │   ├── mod.rs
│   │   └── engine.rs        Inverse-mapping warp loop with Rayon
│   ├── chunk/               Chunk planner
│   │   ├── mod.rs
│   │   └── planner.rs       Destination tile -> source ROI mapping
│   └── py/                  PyO3 binding layer
│       ├── mod.rs
│       ├── reproject.rs     reproject_array + reproject_with_grid bindings
│       ├── transform.rs     transform_points + transform_grid bindings
│       └── plan.rs          plan_reproject binding
├── python/rust_warp/        Python source
│   ├── __init__.py          Public API exports
│   ├── _rust.pyi            Type stubs for the Rust extension
│   ├── geobox.py            GeoBox dataclass
│   ├── reproject.py         High-level reproject() dispatcher
│   ├── dask_graph.py        Dask HighLevelGraph builder
│   └── xarray_accessor.py   .warp xarray accessor
├── tests/                   Python test suite
├── benches/                 Rust benchmarks (Criterion)
└── scripts/                 Utility scripts
```

## The Warp Pipeline

The core algorithm is an **inverse-mapping warp**. For each pixel in the output (destination) grid:

```
destination pixel (row, col)
    │
    ▼
destination affine transform ──► projected coordinates (easting, northing)
    │
    ▼
projection pipeline inverse ──► geographic coordinates (lon, lat)
    │
    ▼
source projection forward ──► source projected coordinates
    │
    ▼
source affine inverse ──► source pixel coordinates (fractional row, col)
    │
    ▼
resampling kernel ──► sample source array at sub-pixel location
    │
    ▼
write to output pixel
```

This is implemented in `src/warp/engine.rs`. The loop runs row-by-row with Rayon parallelism, where each row is independent (reads from shared source, writes to disjoint output row).

### Linear Approximation

Computing exact projections for every pixel is expensive. The `LinearApprox` module (`src/proj/approx.rs`) optimizes this:

1. For each output row, compute exact source coordinates at 3 points (start, middle, end)
2. Linearly interpolate between them for all pixels in between
3. If interpolation error exceeds 0.125 pixels, recursively subdivide (max depth 20)

This reduces projection calls from `width` per row to typically 5-15, with more subdivisions only where curvature demands it (e.g. near the poles).

## Projection Pipeline

The `Pipeline` (`src/proj/pipeline.rs`) routes between two backends:

```
enum Pipeline {
    Native { src: CrsEndpoint, dst: CrsEndpoint },
    Proj4rs(Box<CrsTransform>),
}
```

- **Native path** — used when both source and destination CRSes are among the natively-implemented projections (UTM, Web Mercator, Equirectangular, etc.). These are hand-written Rust implementations matching PROJ's output to sub-millimetre accuracy.
- **proj4rs fallback** — a pure-Rust port of PROJ.4, used for any CRS not covered by native implementations. See [proj4rs-differences.md](proj4rs-differences.md) for accuracy details.

CRS strings are parsed in `src/proj/crs.rs`. The parser recognizes EPSG codes (`"EPSG:32633"`) and PROJ strings (`"+proj=utm +zone=33 +datum=WGS84"`).

## Resampling Kernels

Each kernel in `src/resample/` implements the same interface: given source array, fractional source coordinates, and optional nodata value, return the interpolated value.

| Kernel | Neighborhood | Key Optimizations |
|--------|-------------|-------------------|
| nearest | 1x1 | Simple floor/round |
| bilinear | 2x2 | Standard 4-point weighted average |
| cubic | 4x4 | Precomputed 1D weight arrays (8 calls instead of 16) |
| lanczos | 6x6 | Polynomial sin approximation (degree-11 Taylor, ~6e-8 max error) |
| average | Variable | Precomputed y-overlap weights, area-weighted sums |

All kernels are generic over element type `T: Copy + NumCast + PartialOrd + Default + Send + Sync`.

### Nodata Handling

- **Float types** — NaN is the default nodata sentinel. If any pixel in the kernel neighborhood is NaN, the output is NaN.
- **Integer types** — no NaN equivalent; defaults to 0. Users can provide explicit nodata sentinels (e.g., 255 for uint8, -9999 for int16). Nodata values are excluded from interpolation.

## Chunk Planning

The chunk planner (`src/chunk/planner.rs`) bridges the warp engine to dask's tiled execution model:

1. **Tile the destination grid** into chunks of the requested size
2. For each destination tile:
   a. Sample 21 points per edge along the tile boundary (84 points total)
   b. Inverse-project each point to source pixel coordinates
   c. Compute the bounding box of valid projected points
   d. Expand by kernel radius (halo padding: 0 for nearest, 1 for bilinear, 2 for cubic, 3 for lanczos)
   e. Clip to source array bounds
   f. Record the mapping as a `TilePlan`
3. Mark tiles with no valid projections as `has_data = false` (skip optimization)

## Dask Integration

The dask graph builder (`python/rust_warp/dask_graph.py`) uses `HighLevelGraph.from_collections` to avoid the graph-duplication problem of `dask.delayed`:

```
plan_reproject()  ──►  list of TilePlans
    │
    ▼
For each tile:
    - Map source ROI to dask block indices
    - Create task: (worker_fn, plan_dict, *source_block_keys)
    │
    ▼
HighLevelGraph.from_collections(name, task_dict, dependencies=(src_array,))
    │
    ▼
da.Array(graph, name, chunks, dtype)
```

The worker function (`_reproject_from_blocks`) assembles overlapping source blocks into a contiguous array, then calls `reproject_array` on it. Static arguments (CRS strings, resampling method) are frozen via `functools.partial`.

This approach references source blocks by key rather than embedding their task subgraphs, which is what makes graph construction O(tiles) instead of O(tiles * source_graph_size).

## xarray Accessor

The `.warp` accessor (`python/rust_warp/xarray_accessor.py`) is registered on both `DataArray` and `Dataset`:

- **`da.warp.reproject(dst_crs, resolution=..., shape=...)`** — projects the source boundary to compute a destination GeoBox, then reprojects
- **`da.warp.reproject_match(other)`** — reprojects to match another dataset's grid (accepts rust-warp GeoBox, odc-geo GeoBox, or xarray object)
- **`ds.warp.reproject(...)`** — reprojects all spatial variables, passes non-spatial variables through

For N-D arrays (e.g. `(band, y, x)` or `(time, y, x)`), the accessor iterates over non-spatial dimensions and reprojects each 2D slice independently.

## GIL Release Strategy

All Rust computation releases the Python GIL via `py.allow_threads()`. This means:

- Multiple Python threads can reproject concurrently without blocking each other
- Dask's threaded scheduler works efficiently (no GIL contention)
- The main bottleneck for parallel Python code is memory bandwidth, not the GIL

## Error Handling

Rust errors are defined in `src/error.rs` using `thiserror`:

- **`WarpError`** — warp engine failures (out-of-bounds, allocation)
- **`ProjError`** — projection failures (invalid coordinates, unsupported CRS)
- **`PlanError`** — chunk planner failures (invalid tile sizes, empty grids)

These are converted to Python `RuntimeError` or `ValueError` at the PyO3 boundary.
