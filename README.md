# rust-warp

High-performance raster reprojection engine in Rust with xarray/dask integration. A GDAL-free alternative for chunked raster reprojection.

rust-warp reprojects geospatial raster data between coordinate reference systems using an inverse-mapping warp engine written in Rust, exposed to Python via PyO3. It integrates natively with numpy, dask, and xarray, and is designed for scientific workflows that need fast, lazy, chunked reprojection without a GDAL dependency.

## Features

- **Pure-Rust projection math** — native implementations of Transverse Mercator (UTM), Web Mercator, Lambert Conformal Conic, Albers Equal Area, Stereographic, Sinusoidal, and Equirectangular, with proj4rs fallback for other CRSes
- **Five resampling kernels** — nearest, bilinear, cubic (Keys), Lanczos-3, and area-weighted average
- **Multi-dtype support** — float32, float64, int8, uint8, uint16, int16
- **Rayon parallelism** — scanline-level threading within each chunk for intra-chunk parallelism
- **Linear approximation** — GDAL-style scanline subdivision for fast coordinate transforms (3 exact projections per row, rest interpolated)
- **Dask integration** — HighLevelGraph-based lazy chunked reprojection with zero graph duplication
- **xarray `.warp` accessor** — ergonomic reprojection for DataArrays and Datasets
- **GIL-free** — all computation releases the Python GIL, enabling true multi-threaded Python workloads
- **Zero GDAL dependency** — no C library dependencies at runtime

## Performance

rust-warp matches or exceeds GDAL/rasterio performance across all resampling kernels, with significant advantages for interpolating methods:

| Kernel | rust-warp | rasterio/GDAL | Speedup |
|--------|-----------|---------------|---------|
| nearest | 1.6 ms | 10.4 ms | **6.6x** |
| bilinear | 1.9 ms | 41.9 ms | **22x** |
| cubic | 3.3 ms | 68.3 ms | **21x** |
| lanczos | 8.8 ms | 87.2 ms | **10x** |
| average | 1.0 ms | 4.4 ms | **4.3x** |

*Benchmark: 1024x1024 float32, UTM zone 33N to EPSG:4326, Apple Silicon.*

For dask graph construction, rust-warp builds HighLevelGraphs **11,000x faster** than the naive `dask.delayed` approach (0.024s vs 272s for a 16384x16384 source).

See [docs/performance.md](docs/performance.md) for full benchmark details.

## Installation

```bash
pip install rust-warp
```

For xarray/dask support:

```bash
pip install "rust-warp[xarray]"
```

For development:

```bash
git clone https://github.com/jww/rust-warp.git
cd rust-warp
uv sync --all-extras
uv run maturin develop --release
```

## Quick Start

### NumPy array reprojection

```python
import numpy as np
from rust_warp import reproject_array

src = np.random.rand(512, 512).astype(np.float32)

result = reproject_array(
    src,
    src_crs="EPSG:32633",
    src_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
    dst_crs="EPSG:4326",
    dst_transform=(0.001, 0.0, 15.0, 0.0, -0.001, 59.5),
    dst_shape=(512, 512),
    resampling="bilinear",
)
```

### High-level API with GeoBox

```python
from rust_warp import reproject, GeoBox

src_geobox = GeoBox.from_bbox(
    bbox=(500000, 6550000, 520000, 6600000),
    crs="EPSG:32633",
    resolution=100.0,
)
dst_geobox = GeoBox.from_bbox(
    bbox=(14.5, 59.0, 15.5, 60.0),
    crs="EPSG:4326",
    resolution=0.001,
)

result = reproject(src, src_geobox, dst_geobox, resampling="cubic")
```

### xarray accessor

```python
import xarray as xr
import rust_warp  # registers .warp accessor

ds = xr.open_dataset("my_data.nc")
reprojected = ds["temperature"].warp.reproject("EPSG:4326", resolution=0.01)

# Match another dataset's grid
reprojected = ds["temperature"].warp.reproject_match(reference_ds)
```

### Dask integration

```python
import dask.array as da
from rust_warp import reproject, GeoBox

# Automatically uses HighLevelGraph builder for lazy reprojection
src_dask = da.from_array(src, chunks=(256, 256))
result = reproject(src_dask, src_geobox, dst_geobox, dst_chunks=(256, 256))
# result is a lazy dask array — nothing computed yet
result.compute()
```

## API Reference

### Core functions

#### `reproject_array(src, src_crs, src_transform, dst_crs, dst_transform, dst_shape, resampling="nearest", nodata=None)`

Low-level reprojection of a 2D numpy array. Output dtype matches input. Supports float32, float64, int8, uint8, uint16, int16.

#### `reproject(src_data, src_geobox, dst_geobox, resampling="bilinear", nodata=None, dst_chunks=None)`

High-level dispatcher that accepts numpy or dask arrays. Automatically routes dask arrays through the HighLevelGraph builder for lazy, chunked reprojection.

#### `transform_points(x, y, src_crs, dst_crs)`

Batch coordinate transformation. Takes 1D arrays of x/y coordinates and transforms between CRSes.

#### `transform_grid(src_crs, src_transform, dst_crs, dst_transform, dst_shape)`

Compute source pixel coordinate grids for a reprojection. Returns `(src_col_grid, src_row_grid)` as 2D float64 arrays.

#### `reproject_with_grid(src, src_col_grid, src_row_grid, resampling="nearest", nodata=None)`

Reproject using pre-computed coordinate grids. Useful for testing resampling kernels in isolation.

#### `plan_reproject(src_crs, src_transform, src_shape, dst_crs, dst_transform, dst_shape, dst_chunks=None, resampling="bilinear")`

Chunk planner for dask integration. Returns a list of tile plan dicts mapping destination tiles to source ROIs with halo padding.

### GeoBox

```python
from rust_warp import GeoBox

# From bounding box + resolution
gbox = GeoBox.from_bbox(bbox=(left, bottom, right, top), crs="EPSG:32633", resolution=100.0)

# From bounding box + shape
gbox = GeoBox.from_bbox(bbox=(...), crs="EPSG:4326", shape=(1000, 1000))

# From xarray object
gbox = GeoBox.from_xarray(da)

# From odc-geo GeoBox
gbox = GeoBox.from_odc(odc_geobox)

# Properties
gbox.bounds      # (left, bottom, right, top)
gbox.resolution  # (res_x, res_y), both positive
gbox.shape       # (rows, cols)
gbox.crs         # CRS string
gbox.affine      # (a, b, c, d, e, f) geotransform

# Coordinate arrays for xarray
gbox.xr_coords()  # {"x": array, "y": array}
```

### xarray `.warp` accessor

Registered on both `DataArray` and `Dataset` when `rust_warp` is imported.

```python
da.warp.reproject("EPSG:4326", resolution=0.01)
da.warp.reproject_match(other_dataset)
da.warp.geobox   # GeoBox from coordinates
da.warp.crs      # CRS string or None
```

## Resampling Methods

| Method | Kernel Size | Best For |
|--------|-------------|----------|
| `nearest` | 1x1 | Categorical data, masks, integer class labels |
| `bilinear` | 2x2 | General purpose, smooth continuous data |
| `cubic` | 4x4 | High-quality interpolation, photography |
| `lanczos` | 6x6 | Maximum quality, sharp features |
| `average` | Variable | Downsampling, area aggregation |

## Projection Support

rust-warp includes native Rust implementations for the most common projections:

| Projection | EPSG Examples | Implementation |
|------------|---------------|----------------|
| Transverse Mercator (UTM) | 326xx, 327xx | Kruger n-series, 6th order |
| Web Mercator | 3857 | Native |
| Equirectangular | 4326 (as projection) | Native |
| Lambert Conformal Conic | 2154, State Plane | Native (1SP and 2SP) |
| Albers Equal Area | 5070 | Native |
| Polar/Oblique Stereographic | 3031, 3413 | Native |
| Sinusoidal | MODIS grids | Native |

For CRSes not covered by native implementations, rust-warp falls back to **proj4rs**, a pure-Rust port of PROJ.4. See [docs/architecture.md](docs/architecture.md) for details on the projection pipeline and [docs/proj4rs-differences.md](docs/proj4rs-differences.md) for known differences between proj4rs and PROJ.

## Known Limitations

- **proj4rs vs PROJ** — rust-warp uses proj4rs (a Rust port of PROJ.4) rather than the C PROJ library. This produces slightly different coordinate values at sub-millimetre precision for some projections. At large scales (16K+ pixels), accumulated floating-point differences can cause ~0.001-0.01% of pixels to differ by 1+ values compared to GDAL output. See [docs/proj4rs-differences.md](docs/proj4rs-differences.md).
- **CRS parsing** — CRS input must be an EPSG code (e.g. `"EPSG:32633"`) or a PROJ string (e.g. `"+proj=utm +zone=33 +datum=WGS84"`). WKT strings are converted to EPSG codes via pyproj when available.
- **2D arrays only** — the Rust warp engine operates on 2D arrays. N-D arrays are handled by iterating over non-spatial dimensions in the Python layer.

## Documentation

- [Architecture](docs/architecture.md) — how the Rust and Python layers fit together
- [API Reference](docs/api-reference.md) — detailed function signatures and usage
- [Performance](docs/performance.md) — benchmark results and optimization details
- [proj4rs Differences](docs/proj4rs-differences.md) — known accuracy differences vs PROJ/GDAL
- [Development](docs/development.md) — building, testing, and contributing

## Development

```bash
# Build and test
uv sync --all-extras
uv run maturin develop --release
cargo test
uv run pytest tests/ -x -v

# Benchmarks
cargo bench
uv run pytest tests/test_benchmark.py --benchmark-enable

# Lint
cargo clippy --all-targets -- -D warnings
cargo fmt --check
uv run ruff check python/ tests/
```

## License

MIT
