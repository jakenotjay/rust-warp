from typing import Any

import numpy as np
from numpy.typing import NDArray

def hello() -> str: ...

def reproject_array(
    src: NDArray[Any],
    src_crs: str,
    src_transform: tuple[float, float, float, float, float, float],
    dst_crs: str,
    dst_transform: tuple[float, float, float, float, float, float],
    dst_shape: tuple[int, int],
    resampling: str = "nearest",
    nodata: float | None = None,
) -> NDArray[Any]:
    """Reproject a 2D array from one CRS to another.

    Supports multiple dtypes: float32, float64, uint8, uint16, int16.
    Output dtype matches input dtype.

    Args:
        src: Input 2D numpy array.
        src_crs: Source CRS string (e.g. "EPSG:32633").
        src_transform: Source affine transform as 6-element tuple.
        dst_crs: Destination CRS string.
        dst_transform: Destination affine transform as 6-element tuple.
        dst_shape: Output shape as (rows, cols) tuple.
        resampling: Resampling method — one of "nearest", "bilinear",
            "cubic", "lanczos", or "average".
        nodata: Optional nodata value.

    Returns:
        Reprojected 2D array with same dtype as input.
    """
    ...

def transform_points(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    src_crs: str,
    dst_crs: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transform arrays of coordinates from one CRS to another.

    Args:
        x: 1D array of x coordinates (longitude or easting).
        y: 1D array of y coordinates (latitude or northing).
        src_crs: Source CRS string (e.g. "EPSG:4326").
        dst_crs: Destination CRS string (e.g. "EPSG:32633").

    Returns:
        Tuple of (x_out, y_out) arrays in the destination CRS.
    """
    ...

def plan_reproject(
    src_crs: str,
    src_transform: tuple[float, float, float, float, float, float],
    src_shape: tuple[int, int],
    dst_crs: str,
    dst_transform: tuple[float, float, float, float, float, float],
    dst_shape: tuple[int, int],
    dst_chunks: tuple[int, int] | None = None,
    resampling: str = "bilinear",
) -> list[dict]:
    """Plan chunk-level reprojection tasks for a raster dataset.

    Divides the destination grid into tiles and computes the corresponding
    source ROI (with halo padding) for each tile.

    Args:
        src_crs: Source CRS string.
        src_transform: Source affine transform as 6-element tuple.
        src_shape: Source raster shape as (rows, cols).
        dst_crs: Destination CRS string.
        dst_transform: Destination affine transform as 6-element tuple.
        dst_shape: Destination raster shape as (rows, cols).
        dst_chunks: Optional chunk size as (rows, cols).
        resampling: Resampling method name. Defaults to "bilinear".

    Returns:
        List of tile plan dicts, each with keys:
        - dst_slice: (row_start, row_end, col_start, col_end)
        - src_slice: (row_start, row_end, col_start, col_end)
        - src_transform: (a, b, c, d, e, f) shifted to src_slice origin
        - dst_transform: (a, b, c, d, e, f) shifted to dst_slice origin
        - dst_tile_shape: (rows, cols)
        - has_data: bool
    """
    ...
