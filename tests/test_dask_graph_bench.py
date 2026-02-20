"""Benchmarks for dask graph construction speed.

Measures the time to build a dask reprojection graph (graph build only,
no compute) at various image sizes and chunk sizes.
"""

from __future__ import annotations

import numpy as np
import pytest
from rust_warp import GeoBox, reproject

da = pytest.importorskip("dask.array")


def _make_dask_setup(src_size, chunk_size, dst_size=None):
    """Create source dask array and geoboxes for benchmarking."""
    if dst_size is None:
        dst_size = src_size

    src_crs = "EPSG:32633"
    dst_crs = "EPSG:4326"
    pixel_size = 100.0
    origin_x, origin_y = 500000.0, 6600000.0

    src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)
    src_shape = (src_size, src_size)
    src_geobox = GeoBox(crs=src_crs, shape=src_shape, affine=src_transform)

    # Compute destination transform covering the same area in EPSG:4326
    import rasterio.warp
    from rasterio.crs import CRS

    dst_transform_affine, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        CRS.from_user_input(src_crs),
        CRS.from_user_input(dst_crs),
        src_size,
        src_size,
        left=origin_x,
        bottom=origin_y - src_size * pixel_size,
        right=origin_x + src_size * pixel_size,
        top=origin_y,
    )
    dst_transform = tuple(dst_transform_affine)[:6]
    dst_shape = (dst_height, dst_width)
    dst_geobox = GeoBox(crs=dst_crs, shape=dst_shape, affine=dst_transform)

    src_np = np.arange(src_size * src_size, dtype=np.float64).reshape(src_shape)
    src_dask = da.from_array(src_np, chunks=(chunk_size, chunk_size))

    return src_dask, src_geobox, dst_geobox, (chunk_size, chunk_size)


# --- Graph build benchmarks ---


@pytest.mark.parametrize(
    ("src_size", "chunk_size"),
    [
        (256, 64),
        (1024, 256),
        (4096, 512),
        (16384, 512),
    ],
)
def test_graph_build(benchmark, src_size, chunk_size):
    """Benchmark dask graph construction (no compute)."""
    src_dask, src_geobox, dst_geobox, dst_chunks = _make_dask_setup(src_size, chunk_size)

    def build_graph():
        return reproject(
            src_dask,
            src_geobox,
            dst_geobox,
            resampling="nearest",
            dst_chunks=dst_chunks,
        )

    result = benchmark(build_graph)
    assert isinstance(result, da.Array)
    assert result.shape == dst_geobox.shape


@pytest.mark.parametrize(
    ("src_size", "chunk_size"),
    [
        (1024, 256),
        (4096, 512),
    ],
)
def test_graph_build_and_compute(benchmark, src_size, chunk_size):
    """Benchmark graph build + compute end-to-end."""
    src_dask, src_geobox, dst_geobox, dst_chunks = _make_dask_setup(src_size, chunk_size)

    def build_and_compute():
        result = reproject(
            src_dask,
            src_geobox,
            dst_geobox,
            resampling="nearest",
            dst_chunks=dst_chunks,
        )
        return result.compute()

    result = benchmark.pedantic(build_and_compute, rounds=3, warmup_rounds=1)
    assert isinstance(result, np.ndarray)
    assert result.shape == dst_geobox.shape
