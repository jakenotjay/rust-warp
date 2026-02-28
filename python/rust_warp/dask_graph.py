"""Dask graph builder for chunked reprojection.

Uses HighLevelGraph to reference source blocks by key, avoiding the
graph-duplication problem of dask.delayed.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from rust_warp._backend import _compute_chunk_sizes, _derive_batch_size  # noqa: F401
from rust_warp._rust import plan_reproject, reproject_array

if TYPE_CHECKING:
    from rust_warp.geobox import GeoBox


def _slice_to_block_indices(sr0, sr1, sc0, sc1, row_bounds, col_bounds):
    """Return list of (block_row, block_col) that overlap the pixel-space slice."""
    row_blocks = [
        i for i in range(len(row_bounds) - 1) if row_bounds[i] < sr1 and row_bounds[i + 1] > sr0
    ]
    col_blocks = [
        j for j in range(len(col_bounds) - 1) if col_bounds[j] < sc1 and col_bounds[j + 1] > sc0
    ]
    return [(i, j) for i in row_blocks for j in col_blocks]


def _reproject_from_blocks(
    plan,
    *blocks,
    row_bounds,
    col_bounds,
    src_crs,
    dst_crs,
    resampling,
    nodata,
):
    """Worker: assemble overlapping source blocks and reproject one tile."""
    block_indices = plan["_block_indices"]
    sr0, sr1, sc0, sc1 = plan["src_slice"]
    # assembled is fully tiled by block_indices — every pixel in [sr0:sr1, sc0:sc1]
    # belongs to exactly one source chunk, so all elements are overwritten below.
    assembled = np.empty((sr1 - sr0, sc1 - sc0), dtype=blocks[0].dtype)
    for block, (bi, bj) in zip(blocks, block_indices, strict=True):
        ar0 = max(row_bounds[bi], sr0) - sr0
        ar1 = min(row_bounds[bi + 1], sr1) - sr0
        ac0 = max(col_bounds[bj], sc0) - sc0
        ac1 = min(col_bounds[bj + 1], sc1) - sc0
        sbr0 = max(sr0 - row_bounds[bi], 0)
        sbc0 = max(sc0 - col_bounds[bj], 0)
        assembled[ar0:ar1, ac0:ac1] = block[sbr0 : sbr0 + (ar1 - ar0), sbc0 : sbc0 + (ac1 - ac0)]

    return reproject_array(
        np.ascontiguousarray(assembled),
        src_crs,
        plan["src_transform"],
        dst_crs,
        plan["dst_transform"],
        plan["dst_tile_shape"],
        resampling=resampling,
        nodata=nodata,
    )


def _reproject_from_blocks_3d(
    plan,
    *blocks,
    n_slices,
    row_bounds,
    col_bounds,
    src_crs,
    dst_crs,
    resampling,
    nodata,
):
    """Worker: assemble overlapping 3D source blocks and reproject all slices.

    Each block has shape (n_slices, h, w). Assembles a (n_slices, src_h, src_w)
    patch and reprojects each slice independently, returning (n_slices, dst_h, dst_w).
    """
    block_indices = plan["_block_indices"]
    sr0, sr1, sc0, sc1 = plan["src_slice"]
    # assembled is fully tiled by block_indices — every pixel in [sr0:sr1, sc0:sc1]
    # belongs to exactly one source chunk, so all elements are overwritten below.
    assembled = np.empty((n_slices, sr1 - sr0, sc1 - sc0), dtype=blocks[0].dtype)
    for block, (bi, bj) in zip(blocks, block_indices, strict=True):
        ar0 = max(row_bounds[bi], sr0) - sr0
        ar1 = min(row_bounds[bi + 1], sr1) - sr0
        ac0 = max(col_bounds[bj], sc0) - sc0
        ac1 = min(col_bounds[bj + 1], sc1) - sc0
        sbr0 = max(sr0 - row_bounds[bi], 0)
        sbc0 = max(sc0 - col_bounds[bj], 0)
        assembled[:, ar0:ar1, ac0:ac1] = block[
            :, sbr0 : sbr0 + (ar1 - ar0), sbc0 : sbc0 + (ac1 - ac0)
        ]

    dst_shape = plan["dst_tile_shape"]
    # result is fully written by the loop below — every slice i is assigned.
    result = np.empty((n_slices, *dst_shape), dtype=blocks[0].dtype)
    for i in range(n_slices):
        result[i] = reproject_array(
            np.ascontiguousarray(assembled[i]),
            src_crs,
            plan["src_transform"],
            dst_crs,
            plan["dst_transform"],
            dst_shape,
            resampling=resampling,
            nodata=nodata,
        )
    return result


def reproject_dask(
    src_data,
    src_geobox: GeoBox,
    dst_geobox: GeoBox,
    resampling: str = "bilinear",
    dst_chunks: tuple[int, int] | None = None,
    nodata: float | None = None,
):
    """Reproject a 2D or 3D dask array using the Rust chunk planner.

    Builds a HighLevelGraph that references source blocks by key, avoiding
    the graph-duplication problem of dask.delayed.

    Args:
        src_data: Source dask array (2D ``(y, x)`` or 3D ``(n, y, x)``).
        src_geobox: Source GeoBox.
        dst_geobox: Destination GeoBox.
        resampling: Resampling method name.
        dst_chunks: Destination chunk size as (rows, cols). If None, uses
            source chunk sizes or full image.
        nodata: Optional nodata value.

    Returns:
        Reprojected dask array with the same number of dimensions as input.
    """
    import dask.array as da
    from dask.highlevelgraph import HighLevelGraph

    if src_data.ndim == 2:
        src_chunks_y = src_data.chunks[0]
        src_chunks_x = src_data.chunks[1]
    elif src_data.ndim == 3:
        src_chunks_y = src_data.chunks[1]
        src_chunks_x = src_data.chunks[2]
    else:
        raise ValueError(f"src_data must be 2D or 3D, got {src_data.ndim}D")

    if dst_chunks is None:
        if src_chunks_y and src_chunks_x:
            dst_chunks = (src_chunks_y[0], src_chunks_x[0])
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

    name = f"reproject-{uuid4().hex}"
    dsk = {}
    chunks_y = _compute_chunk_sizes(dst_geobox.shape[0], dst_chunks[0])
    chunks_x = _compute_chunk_sizes(dst_geobox.shape[1], dst_chunks[1])

    if src_data.ndim == 2:
        row_bounds = tuple(int(x) for x in np.cumsum((0,) + src_data.chunks[0]))
        col_bounds = tuple(int(x) for x in np.cumsum((0,) + src_data.chunks[1]))
        src_keys = src_data.__dask_keys__()

        proc = partial(
            _reproject_from_blocks,
            row_bounds=row_bounds,
            col_bounds=col_bounds,
            src_crs=src_geobox.crs,
            dst_crs=dst_geobox.crs,
            resampling=resampling,
            nodata=nodata,
        )

        for plan in plans:
            r0, _r1, c0, _c1 = plan["dst_slice"]
            tile_shape = plan["dst_tile_shape"]
            row_idx = r0 // dst_chunks[0]
            col_idx = c0 // dst_chunks[1]
            k = (name, row_idx, col_idx)

            if not plan["has_data"]:
                dsk[k] = (np.full, tile_shape, fill_value, dtype)
            else:
                sr0, sr1, sc0, sc1 = plan["src_slice"]
                indices = _slice_to_block_indices(sr0, sr1, sc0, sc1, row_bounds, col_bounds)
                block_deps = tuple(src_keys[bi][bj] for bi, bj in indices)
                dsk[k] = (proc, {**plan, "_block_indices": indices}, *block_deps)

        graph = HighLevelGraph.from_collections(name, dsk, dependencies=(src_data,))
        return da.Array(graph, name, chunks=(chunks_y, chunks_x), dtype=dtype)

    else:  # ndim == 3
        n_chunk_sizes = src_data.chunks[0]
        row_bounds = tuple(int(x) for x in np.cumsum((0,) + src_data.chunks[1]))
        col_bounds = tuple(int(x) for x in np.cumsum((0,) + src_data.chunks[2]))
        src_keys = src_data.__dask_keys__()  # 3-level nested list

        for n_chunk_idx, n_chunk_sz in enumerate(n_chunk_sizes):
            proc_3d = partial(
                _reproject_from_blocks_3d,
                n_slices=n_chunk_sz,
                row_bounds=row_bounds,
                col_bounds=col_bounds,
                src_crs=src_geobox.crs,
                dst_crs=dst_geobox.crs,
                resampling=resampling,
                nodata=nodata,
            )

            for plan in plans:
                r0, _r1, c0, _c1 = plan["dst_slice"]
                tile_shape = plan["dst_tile_shape"]
                row_idx = r0 // dst_chunks[0]
                col_idx = c0 // dst_chunks[1]
                k = (name, n_chunk_idx, row_idx, col_idx)

                if not plan["has_data"]:
                    dsk[k] = (np.full, (n_chunk_sz, *tile_shape), fill_value, dtype)
                else:
                    sr0, sr1, sc0, sc1 = plan["src_slice"]
                    indices = _slice_to_block_indices(sr0, sr1, sc0, sc1, row_bounds, col_bounds)
                    block_deps = tuple(src_keys[n_chunk_idx][bi][bj] for bi, bj in indices)
                    dsk[k] = (proc_3d, {**plan, "_block_indices": indices}, *block_deps)

        output_chunks = (n_chunk_sizes, chunks_y, chunks_x)
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=(src_data,))
        return da.Array(graph, name, chunks=output_chunks, dtype=dtype)
