"""High-level reproject() dispatcher for numpy and dask arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rust_warp._rust import reproject_array

if TYPE_CHECKING:
    from rust_warp.geobox import GeoBox


def reproject(
    src_data,
    src_geobox: GeoBox,
    dst_geobox: GeoBox,
    resampling: str = "bilinear",
    nodata: float | None = None,
    dst_chunks: tuple[int, int] | None = None,
):
    """Reproject a 2D array from one CRS/grid to another.

    Automatically dispatches to dask-backed reprojection if the input is a
    dask array, otherwise uses the direct numpy path.

    Args:
        src_data: Source 2D array (numpy or dask).
        src_geobox: Source GeoBox.
        dst_geobox: Destination GeoBox.
        resampling: Resampling method name.
        nodata: Optional nodata value.
        dst_chunks: Destination chunk size for dask path. Ignored for numpy.

    Returns:
        Reprojected 2D array (same type as input: numpy or dask).
    """
    if hasattr(src_data, "dask"):
        from rust_warp.dask_graph import reproject_dask

        return reproject_dask(
            src_data,
            src_geobox,
            dst_geobox,
            resampling=resampling,
            dst_chunks=dst_chunks,
            nodata=nodata,
        )

    src_arr = np.ascontiguousarray(src_data)
    return reproject_array(
        src_arr,
        src_geobox.crs,
        src_geobox.affine,
        dst_geobox.crs,
        dst_geobox.affine,
        dst_geobox.shape,
        resampling=resampling,
        nodata=nodata,
    )
