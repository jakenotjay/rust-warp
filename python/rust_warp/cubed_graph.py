"""Cubed graph builder for chunked reprojection.

Uses ``cubed.core.ops.map_selection`` to express per-tile source dependencies,
giving cubed its bounded-memory guarantees while avoiding eager computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rust_warp._backend import _compute_chunk_sizes
from rust_warp._rust import plan_reproject, reproject_array

if TYPE_CHECKING:
    from rust_warp.geobox import GeoBox


def _count_overlapping_chunks(s0, s1, chunk_boundaries):
    """Count how many chunks in *chunk_boundaries* overlap the range [s0, s1)."""
    count = 0
    for i in range(len(chunk_boundaries) - 1):
        if chunk_boundaries[i] < s1 and chunk_boundaries[i + 1] > s0:
            count += 1
    return count


def reproject_cubed(
    src_data,
    src_geobox: GeoBox,
    dst_geobox: GeoBox,
    resampling: str = "bilinear",
    dst_chunks: tuple[int, int] | None = None,
    nodata: float | None = None,
):
    """Reproject a 2D or 3D cubed array using the Rust chunk planner.

    Uses ``cubed.core.ops.map_selection`` so each destination tile reads only
    the source blocks it needs, preserving cubed's bounded-memory guarantees.

    Falls back to eager (compute-then-wrap) if ``map_selection`` is unavailable.

    Args:
        src_data: Source cubed array (2D ``(y, x)`` or 3D ``(n, y, x)``).
        src_geobox: Source GeoBox.
        dst_geobox: Destination GeoBox.
        resampling: Resampling method name.
        dst_chunks: Destination chunk size as (rows, cols). If None, uses
            source chunk sizes or full image.
        nodata: Optional nodata value.

    Returns:
        Reprojected cubed array with the same number of dimensions as input.
    """
    if src_data.ndim == 2:
        return _reproject_cubed_2d(src_data, src_geobox, dst_geobox, resampling, dst_chunks, nodata)
    if src_data.ndim == 3:
        return _reproject_cubed_3d(src_data, src_geobox, dst_geobox, resampling, dst_chunks, nodata)
    raise ValueError(f"src_data must be 2D or 3D, got {src_data.ndim}D")


def _reproject_cubed_2d(src_data, src_geobox, dst_geobox, resampling, dst_chunks, nodata):
    """Reproject a 2D cubed array."""
    if dst_chunks is None:
        if src_data.chunks[0] and src_data.chunks[1]:
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

    # Build plan lookup: (row_idx, col_idx) -> plan dict
    plan_lookup = {}
    for plan in plans:
        r0, _r1, c0, _c1 = plan["dst_slice"]
        ri = r0 // dst_chunks[0]
        ci = c0 // dst_chunks[1]
        plan_lookup[(ri, ci)] = plan

    # Compute max input blocks any tile needs
    src_row_bounds = tuple(int(x) for x in np.cumsum((0,) + src_data.chunks[0]))
    src_col_bounds = tuple(int(x) for x in np.cumsum((0,) + src_data.chunks[1]))
    max_input_blocks = 1
    for plan in plans:
        if not plan["has_data"]:
            continue
        sr0, sr1, sc0, sc1 = plan["src_slice"]
        n_row = _count_overlapping_chunks(sr0, sr1, src_row_bounds)
        n_col = _count_overlapping_chunks(sc0, sc1, src_col_bounds)
        max_input_blocks = max(max_input_blocks, n_row * n_col)

    chunks_y = _compute_chunk_sizes(dst_geobox.shape[0], dst_chunks[0])
    chunks_x = _compute_chunk_sizes(dst_geobox.shape[1], dst_chunks[1])

    try:
        from cubed.core.ops import map_selection
    except ImportError:
        return _reproject_cubed_eager(
            src_data, src_geobox, dst_geobox, resampling, dst_chunks, nodata
        )

    def selection_fn(out_key):
        # out_key is ('out', row_idx, col_idx) — strip the name prefix
        _, ri, ci = out_key
        plan = plan_lookup.get((ri, ci))
        if plan is None or not plan["has_data"]:
            return (slice(0, 1), slice(0, 1))
        sr0, sr1, sc0, sc1 = plan["src_slice"]
        return (slice(sr0, sr1), slice(sc0, sc1))

    def worker(src_region, block_id=None):
        # block_id is (row_idx, col_idx) — no name prefix
        plan = plan_lookup.get(block_id)
        if plan is None or not plan["has_data"]:
            return np.full(plan["dst_tile_shape"] if plan else (1, 1), fill_value, dtype=dtype)
        return reproject_array(
            np.ascontiguousarray(src_region),
            src_geobox.crs,
            plan["src_transform"],
            dst_geobox.crs,
            plan["dst_transform"],
            plan["dst_tile_shape"],
            resampling=resampling,
            nodata=nodata,
        )

    return map_selection(
        func=worker,
        selection_function=selection_fn,
        x=src_data,
        shape=dst_geobox.shape,
        dtype=dtype,
        chunks=(chunks_y, chunks_x),
        max_num_input_blocks=max_input_blocks,
    )


def _reproject_cubed_3d(src_data, src_geobox, dst_geobox, resampling, dst_chunks, nodata):
    """Reproject a 3D (n, y, x) cubed array."""
    if dst_chunks is None:
        if src_data.chunks[1] and src_data.chunks[2]:
            dst_chunks = (src_data.chunks[1][0], src_data.chunks[2][0])
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

    n_total = src_data.shape[0]
    n_chunk_sizes = src_data.chunks[0]

    # Build plan lookup: (row_idx, col_idx) -> plan dict
    plan_lookup = {}
    for plan in plans:
        r0, _r1, c0, _c1 = plan["dst_slice"]
        ri = r0 // dst_chunks[0]
        ci = c0 // dst_chunks[1]
        plan_lookup[(ri, ci)] = plan

    # Compute max input blocks
    src_row_bounds = tuple(int(x) for x in np.cumsum((0,) + src_data.chunks[1]))
    src_col_bounds = tuple(int(x) for x in np.cumsum((0,) + src_data.chunks[2]))
    max_input_blocks = 1
    for plan in plans:
        if not plan["has_data"]:
            continue
        sr0, sr1, sc0, sc1 = plan["src_slice"]
        n_row = _count_overlapping_chunks(sr0, sr1, src_row_bounds)
        n_col = _count_overlapping_chunks(sc0, sc1, src_col_bounds)
        # Each n-chunk is one block, and each spatial block pair is one block
        max_input_blocks = max(max_input_blocks, n_row * n_col)

    chunks_y = _compute_chunk_sizes(dst_geobox.shape[0], dst_chunks[0])
    chunks_x = _compute_chunk_sizes(dst_geobox.shape[1], dst_chunks[1])

    try:
        from cubed.core.ops import map_selection
    except ImportError:
        return _reproject_cubed_eager(
            src_data, src_geobox, dst_geobox, resampling, dst_chunks, nodata
        )

    def selection_fn(out_key):
        # out_key is ('out', n_idx, row_idx, col_idx) — strip the name prefix
        _, n_idx, ri, ci = out_key
        # Compute n-slice boundaries
        n_start = sum(n_chunk_sizes[:n_idx])
        n_end = n_start + n_chunk_sizes[n_idx]
        plan = plan_lookup.get((ri, ci))
        if plan is None or not plan["has_data"]:
            return (slice(n_start, n_end), slice(0, 1), slice(0, 1))
        sr0, sr1, sc0, sc1 = plan["src_slice"]
        return (slice(n_start, n_end), slice(sr0, sr1), slice(sc0, sc1))

    def worker(src_region, block_id=None):
        # block_id is (n_idx, row_idx, col_idx) — no name prefix
        _n_idx, ri, ci = block_id
        plan = plan_lookup.get((ri, ci))
        n_slices = src_region.shape[0]
        dst_shape = plan["dst_tile_shape"] if plan else (1, 1)
        if plan is None or not plan["has_data"]:
            return np.full((n_slices, *dst_shape), fill_value, dtype=dtype)
        result = np.empty((n_slices, *dst_shape), dtype=dtype)
        for i in range(n_slices):
            result[i] = reproject_array(
                np.ascontiguousarray(src_region[i]),
                src_geobox.crs,
                plan["src_transform"],
                dst_geobox.crs,
                plan["dst_transform"],
                dst_shape,
                resampling=resampling,
                nodata=nodata,
            )
        return result

    return map_selection(
        func=worker,
        selection_function=selection_fn,
        x=src_data,
        shape=(n_total, *dst_geobox.shape),
        dtype=dtype,
        chunks=(n_chunk_sizes, chunks_y, chunks_x),
        max_num_input_blocks=max_input_blocks,
    )


def _reproject_cubed_eager(src_data, src_geobox, dst_geobox, resampling, dst_chunks, nodata):
    """Fallback: compute eagerly, reproject as numpy, wrap back in cubed.

    Used when ``cubed.core.ops.map_selection`` is unavailable (older cubed).
    """
    import cubed

    from rust_warp.reproject import reproject

    src_np = np.asarray(src_data.compute())

    if src_np.ndim == 2:
        result_np = reproject(
            src_np,
            src_geobox,
            dst_geobox,
            resampling=resampling,
            nodata=nodata,
        )
    elif src_np.ndim == 3:
        dst_rows, dst_cols = dst_geobox.shape
        result_np = np.empty((src_np.shape[0], dst_rows, dst_cols), dtype=src_np.dtype)
        for i in range(src_np.shape[0]):
            result_np[i] = reproject(
                src_np[i],
                src_geobox,
                dst_geobox,
                resampling=resampling,
                nodata=nodata,
            )
    else:
        raise ValueError(f"src_data must be 2D or 3D, got {src_np.ndim}D")

    if dst_chunks is None:
        dst_chunks = dst_geobox.shape

    if result_np.ndim == 2:
        cubed_chunks = dst_chunks
    else:
        cubed_chunks = (src_data.chunks[0], dst_chunks[0], dst_chunks[1])

    spec = getattr(src_data, "spec", None)
    kwargs = {"spec": spec} if spec is not None else {}
    return cubed.from_array(result_np, chunks=cubed_chunks, **kwargs)
