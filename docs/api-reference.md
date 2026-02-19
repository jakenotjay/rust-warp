# API Reference

## Low-Level Functions

These are exposed directly from the Rust extension (`rust_warp._rust`).

### `reproject_array`

```python
rust_warp.reproject_array(
    src: NDArray,
    src_crs: str,
    src_transform: tuple[float, float, float, float, float, float],
    dst_crs: str,
    dst_transform: tuple[float, float, float, float, float, float],
    dst_shape: tuple[int, int],
    resampling: str = "nearest",
    nodata: float | None = None,
) -> NDArray
```

Reproject a 2D numpy array from one CRS to another.

**Parameters:**

- **src** — Input 2D numpy array. Supported dtypes: `float32`, `float64`, `int8`, `uint8`, `uint16`, `int16`.
- **src_crs** — Source CRS as an EPSG code (`"EPSG:32633"`) or PROJ string (`"+proj=utm +zone=33 +datum=WGS84"`).
- **src_transform** — Source affine geotransform as a 6-element tuple `(a, b, c, d, e, f)` where `x = a*col + b*row + c` and `y = d*col + e*row + f`. For a north-up raster: `(pixel_width, 0, x_origin, 0, -pixel_height, y_origin)`.
- **dst_crs** — Destination CRS string.
- **dst_transform** — Destination affine geotransform.
- **dst_shape** — Output shape as `(rows, cols)`.
- **resampling** — One of `"nearest"`, `"bilinear"`, `"cubic"`, `"lanczos"`, or `"average"`.
- **nodata** — Optional nodata value. For float types, pixels outside the source extent are filled with NaN by default. For integer types, the default fill is 0.

**Returns:** Reprojected 2D array with the same dtype as input.

**Example:**

```python
import numpy as np
from rust_warp import reproject_array

src = np.random.rand(256, 256).astype(np.float32)
result = reproject_array(
    src,
    src_crs="EPSG:32633",
    src_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
    dst_crs="EPSG:4326",
    dst_transform=(0.001, 0.0, 15.0, 0.0, -0.001, 59.5),
    dst_shape=(256, 256),
    resampling="bilinear",
)
```

---

### `transform_points`

```python
rust_warp.transform_points(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    src_crs: str,
    dst_crs: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Transform arrays of coordinates between CRSes.

**Parameters:**

- **x** — 1D array of x coordinates (longitude or easting).
- **y** — 1D array of y coordinates (latitude or northing).
- **src_crs** — Source CRS string.
- **dst_crs** — Destination CRS string.

**Returns:** Tuple of `(x_out, y_out)` arrays in the destination CRS.

**Example:**

```python
import numpy as np
from rust_warp import transform_points

lons = np.array([15.0, 16.0, 17.0])
lats = np.array([52.0, 53.0, 54.0])
eastings, northings = transform_points(lons, lats, "EPSG:4326", "EPSG:32633")
```

---

### `transform_grid`

```python
rust_warp.transform_grid(
    src_crs: str,
    src_transform: tuple[float, float, float, float, float, float],
    dst_crs: str,
    dst_transform: tuple[float, float, float, float, float, float],
    dst_shape: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Compute source pixel coordinate grids for a reprojection. For each destination pixel, returns the corresponding source pixel coordinates using the same transform chain as `reproject_array`.

**Parameters:**

- **src_crs** — Source CRS string.
- **src_transform** — Source affine geotransform.
- **dst_crs** — Destination CRS string.
- **dst_transform** — Destination affine geotransform.
- **dst_shape** — Output shape as `(rows, cols)`.

**Returns:** Tuple of `(src_col_grid, src_row_grid)` — two 2D float64 arrays of shape `dst_shape`.

---

### `reproject_with_grid`

```python
rust_warp.reproject_with_grid(
    src: NDArray,
    src_col_grid: NDArray[np.float64],
    src_row_grid: NDArray[np.float64],
    resampling: str = "nearest",
    nodata: float | None = None,
) -> NDArray
```

Reproject using pre-computed source pixel coordinate grids. Bypasses projection entirely — tests only the resampling kernel. Useful for isolating kernel behavior in tests.

**Parameters:**

- **src** — Input 2D numpy array.
- **src_col_grid** — 2D float64 array of source column coordinates.
- **src_row_grid** — 2D float64 array of source row coordinates.
- **resampling** — Resampling method name.
- **nodata** — Optional nodata value.

**Returns:** Resampled 2D array with same dtype as input.

---

### `plan_reproject`

```python
rust_warp.plan_reproject(
    src_crs: str,
    src_transform: tuple[float, float, float, float, float, float],
    src_shape: tuple[int, int],
    dst_crs: str,
    dst_transform: tuple[float, float, float, float, float, float],
    dst_shape: tuple[int, int],
    dst_chunks: tuple[int, int] | None = None,
    resampling: str = "bilinear",
) -> list[dict]
```

Plan chunk-level reprojection tasks. Divides the destination grid into tiles and computes the corresponding source ROI (with halo padding) for each tile.

**Parameters:**

- **src_crs**, **dst_crs** — CRS strings.
- **src_transform**, **dst_transform** — Affine geotransforms.
- **src_shape**, **dst_shape** — Raster shapes as `(rows, cols)`.
- **dst_chunks** — Chunk size for destination tiles. If None, uses the full destination shape.
- **resampling** — Resampling method (determines halo size: 0 for nearest, 1 for bilinear, 2 for cubic, 3 for lanczos).

**Returns:** List of tile plan dicts with keys:

| Key | Type | Description |
|-----|------|-------------|
| `dst_slice` | `(int, int, int, int)` | `(row_start, row_end, col_start, col_end)` in the destination grid |
| `src_slice` | `(int, int, int, int)` | `(row_start, row_end, col_start, col_end)` in the source grid (with halo) |
| `src_transform` | `(float, ...)` | Affine transform shifted to `src_slice` origin |
| `dst_transform` | `(float, ...)` | Affine transform shifted to `dst_slice` origin |
| `dst_tile_shape` | `(int, int)` | `(rows, cols)` of this destination tile |
| `has_data` | `bool` | Whether the source extent covers this tile |

---

## High-Level Functions

### `reproject`

```python
rust_warp.reproject(
    src_data,
    src_geobox: GeoBox,
    dst_geobox: GeoBox,
    resampling: str = "bilinear",
    nodata: float | None = None,
    dst_chunks: tuple[int, int] | None = None,
)
```

High-level reproject dispatcher. Detects dask arrays automatically and routes to the appropriate backend.

**Parameters:**

- **src_data** — 2D numpy array or dask array.
- **src_geobox** — Source `GeoBox`.
- **dst_geobox** — Destination `GeoBox`.
- **resampling** — Resampling method.
- **nodata** — Optional nodata value.
- **dst_chunks** — Destination chunk size for dask path. Ignored for numpy input.

**Returns:** Reprojected array (numpy or dask, matching input type).

---

## GeoBox

```python
from rust_warp import GeoBox
```

A frozen dataclass combining CRS, affine transform, and grid shape to fully describe a georeferenced pixel grid.

### Constructors

#### `GeoBox(crs, shape, affine)`

Direct construction.

- **crs** (`str`) — CRS string (e.g., `"EPSG:32633"`).
- **shape** (`tuple[int, int]`) — Grid shape as `(rows, cols)`.
- **affine** (`tuple[float, ...]`) — 6-element affine transform `(a, b, c, d, e, f)`.

#### `GeoBox.from_bbox(bbox, crs, resolution=None, shape=None)`

Create from bounding box. Must provide either `resolution` or `shape`.

- **bbox** — `(left, bottom, right, top)` in CRS units.
- **resolution** — Pixel size as scalar or `(res_x, res_y)`.
- **shape** — Grid shape as `(rows, cols)`.

#### `GeoBox.from_xarray(obj)`

Extract from xarray DataArray or Dataset. Reads CRS from `spatial_ref` coordinate or `attrs["crs"]`, and infers affine from coordinate arrays.

#### `GeoBox.from_odc(gbox)`

Create from an `odc.geo.geobox.GeoBox` instance.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `crs` | `str` | CRS string |
| `shape` | `tuple[int, int]` | `(rows, cols)` |
| `affine` | `tuple[float, ...]` | 6-element geotransform |
| `bounds` | `tuple[float, float, float, float]` | `(left, bottom, right, top)` |
| `resolution` | `tuple[float, float]` | `(res_x, res_y)`, both positive |

### Methods

#### `xr_coords()`

Returns `{"x": NDArray, "y": NDArray}` — 1D arrays of pixel-center coordinates suitable for xarray construction.

---

## xarray `.warp` Accessor

Registered automatically when `rust_warp` is imported. Available on both `DataArray` and `Dataset`.

### DataArray

#### `da.warp.reproject(dst_crs, resolution=None, shape=None, resampling="bilinear", nodata=None, dst_chunks=None)`

Reproject to a new CRS. If neither `resolution` nor `shape` is given, auto-computes to preserve total pixel count.

#### `da.warp.reproject_match(other, resampling="bilinear", nodata=None, dst_chunks=None)`

Reproject to match another grid. Accepts a rust-warp GeoBox, odc-geo GeoBox, or xarray DataArray/Dataset.

#### `da.warp.geobox`

Property returning the `GeoBox` derived from coordinates and CRS metadata.

#### `da.warp.crs`

Property returning the CRS string, or `None` if not set.

### Dataset

The Dataset accessor works identically but reprojects all spatial variables. Non-spatial variables are passed through unchanged.
