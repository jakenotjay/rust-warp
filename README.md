# rust-warp

High-performance raster reprojection engine in Rust with xarray/dask integration. A GDAL-free alternative for chunked raster reprojection.

## Features

- Pure-Rust projection math (UTM, Web Mercator, and more) with proj4rs fallback
- Resampling kernels: nearest, bilinear, cubic (Keys), Lanczos, area-weighted average
- Multi-dtype support: float32, float64, uint8, uint16, int16
- Rayon parallelism with scanline-level threading
- Linear approximation for fast coordinate transforms (GDAL-style subdivision)
- Dask integration for lazy, chunked reprojection
- xarray `.geoflux` accessor for ergonomic geospatial workflows
- Zero GDAL dependency at runtime

## Installation

```bash
pip install rust-warp
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

src = np.random.rand(512, 512).astype(np.float64)

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
import rust_warp  # registers .geoflux accessor

ds = xr.open_dataset("my_data.nc")
reprojected = ds["temperature"].geoflux.reproject("EPSG:4326", resolution=0.01)
```

### Dask integration

```python
import dask.array as da
from rust_warp import reproject, GeoBox

# Automatically uses dask graph builder for lazy reprojection
src_dask = da.from_array(src, chunks=(256, 256))
result = reproject(src_dask, src_geobox, dst_geobox, dst_chunks=(256, 256))
```

## API Reference

### `reproject_array(src, src_crs, src_transform, dst_crs, dst_transform, dst_shape, resampling="bilinear", nodata=None)`

Low-level reprojection of a 2D numpy array.

### `reproject(src_data, src_geobox, dst_geobox, resampling="bilinear", nodata=None, dst_chunks=None)`

High-level dispatcher supporting numpy and dask arrays.

### `GeoBox(crs, shape, affine)`

Georeferenced grid definition combining CRS, affine transform, and shape.

### `plan_reproject(...)`

Chunk planner for dask integration. Returns tile plans for parallel execution.

## Resampling Methods

| Method | Kernel Size | Best For |
|--------|------------|----------|
| `nearest` | 1x1 | Categorical data, masks |
| `bilinear` | 2x2 | General purpose, smooth data |
| `cubic` | 4x4 | High-quality interpolation |
| `lanczos` | 6x6 | Maximum quality, sharp features |
| `average` | Variable | Downsampling, area aggregation |

## Performance

rust-warp matches or exceeds GDAL/rasterio performance for reprojection at all tested sizes, with the advantage of being a pure-Rust implementation with no C library dependencies.

## Development

```bash
# Run Rust tests
cargo test

# Run Python tests
uv run pytest tests/ -x -v

# Run benchmarks
cargo bench
uv run pytest tests/test_benchmark.py --benchmark-enable

# Lint
cargo clippy --all-targets -- -D warnings
cargo fmt --check
uv run ruff check python/ tests/
```

## License

MIT
