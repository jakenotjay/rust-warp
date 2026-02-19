"""Dask graph builder for chunked reprojection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rust_warp._rust import plan_reproject, reproject_array

if TYPE_CHECKING:
    from rust_warp.geobox import GeoBox


def _reproject_tile(
    src_chunk: np.ndarray,
    src_crs: str,
    src_transform: tuple[float, ...],
    dst_crs: str,
    dst_transform: tuple[float, ...],
    dst_tile_shape: tuple[int, int],
    resampling: str,
    nodata: float | None,
) -> np.ndarray:
    """Worker function: reproject a single tile."""
    src_arr = np.ascontiguousarray(src_chunk)
    return reproject_array(
        src_arr,
        src_crs,
        src_transform,
        dst_crs,
        dst_transform,
        dst_tile_shape,
        resampling=resampling,
        nodata=nodata,
    )


def reproject_dask(
    src_data,
    src_geobox: GeoBox,
    dst_geobox: GeoBox,
    resampling: str = "bilinear",
    dst_chunks: tuple[int, int] | None = None,
    nodata: float | None = None,
):
    """Reproject a dask array using the chunk planner.

    Args:
        src_data: Source dask array (2D).
        src_geobox: Source GeoBox.
        dst_geobox: Destination GeoBox.
        resampling: Resampling method name.
        dst_chunks: Destination chunk size as (rows, cols). If None, uses
            source chunk sizes or full image.
        nodata: Optional nodata value.

    Returns:
        Reprojected dask array.
    """
    import dask
    import dask.array as da

    if dst_chunks is None:
        # Default to source chunk sizes if available
        if hasattr(src_data, "chunks") and src_data.chunks:
            dst_chunks = (src_data.chunks[0][0], src_data.chunks[1][0])
        else:
            dst_chunks = dst_geobox.shape

    plans = plan_reproject(
        src_crs=src_geobox.crs,
        src_transform=src_geobox.affine,
        src_shape=src_geobox.shape,
        dst_crs=dst_geobox.crs,
        dst_transform=dst_geobox.affine,
        dst_shape=dst_geobox.shape,
        dst_chunks=dst_chunks,
        resampling=resampling,
    )

    dtype = src_data.dtype
    if nodata is not None:
        fill_value = nodata
    elif np.issubdtype(dtype, np.floating):
        fill_value = np.nan
    else:
        fill_value = 0

    # Group plans by row for assembly
    row_groups: dict[int, list[dict]] = {}
    for plan in plans:
        r0 = plan["dst_slice"][0]
        row_groups.setdefault(r0, []).append(plan)

    # Sort row groups and tiles within each row
    row_keys = sorted(row_groups.keys())
    tile_rows = []

    for r0 in row_keys:
        tiles = sorted(row_groups[r0], key=lambda p: p["dst_slice"][2])
        row_tiles = []

        for plan in tiles:
            tile_shape = plan["dst_tile_shape"]

            if not plan["has_data"]:
                tile = da.full(tile_shape, fill_value, dtype=dtype)
            else:
                sr0, sr1, sc0, sc1 = plan["src_slice"]
                src_chunk = src_data[sr0:sr1, sc0:sc1]
                tile = dask.delayed(_reproject_tile)(
                    src_chunk,
                    src_geobox.crs,
                    plan["src_transform"],
                    dst_geobox.crs,
                    plan["dst_transform"],
                    tile_shape,
                    resampling,
                    nodata,
                )
                tile = da.from_delayed(tile, shape=tile_shape, dtype=dtype)

            row_tiles.append(tile)

        if len(row_tiles) == 1:
            tile_rows.append(row_tiles[0])
        else:
            tile_rows.append(da.concatenate(row_tiles, axis=1))

    if len(tile_rows) == 1:
        return tile_rows[0]
    return da.concatenate(tile_rows, axis=0)
