"""Performance benchmarks: rust-warp vs rasterio/GDAL."""

import dask.array as _da
import numpy as np
import pytest
import rasterio.warp
from rasterio.crs import CRS
from rust_warp import plan_reproject, reproject_array
from rust_warp.dask_graph import reproject_dask as _reproject_dask
from rust_warp.geobox import GeoBox as _GeoBox


def _make_test_data(size):
    """Create a synthetic raster and UTM->4326 transform parameters."""
    src_crs = "EPSG:32633"
    dst_crs = "EPSG:4326"
    pixel_size = 100.0
    origin_x, origin_y = 500000.0, 6600000.0

    src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    src = (rows * size + cols).astype(np.float64)

    dst_transform_affine, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        CRS.from_user_input(src_crs),
        CRS.from_user_input(dst_crs),
        size,
        size,
        left=origin_x,
        bottom=origin_y - size * pixel_size,
        right=origin_x + size * pixel_size,
        top=origin_y,
    )
    dst_transform = tuple(dst_transform_affine)[:6]
    dst_shape = (dst_height, dst_width)

    return src, src_crs, src_transform, dst_crs, dst_transform, dst_shape


def _make_downscale_data(src_size, dst_size):
    """Create a synthetic raster for downsampling benchmarks (same CRS)."""
    crs = "EPSG:32633"
    pixel_size = 100.0
    origin_x, origin_y = 500000.0, 6600000.0

    src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

    rows, cols = np.meshgrid(np.arange(src_size), np.arange(src_size), indexing="ij")
    src = (rows * src_size + cols).astype(np.float64)

    dst_pixel_size = pixel_size * (src_size / dst_size)
    dst_transform = (dst_pixel_size, 0.0, origin_x, 0.0, -dst_pixel_size, origin_y)
    dst_shape = (dst_size, dst_size)

    return src, crs, src_transform, crs, dst_transform, dst_shape


# ---------------------------------------------------------------------------
# Nearest
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="nearest")
@pytest.mark.parametrize("size", [256, 512, 1024])
def test_rustwarp_nearest(benchmark, size):
    src, src_crs, src_transform, dst_crs, dst_transform, dst_shape = _make_test_data(size)

    benchmark(
        reproject_array,
        src,
        src_crs,
        src_transform,
        dst_crs,
        dst_transform,
        dst_shape,
        resampling="nearest",
    )


@pytest.mark.benchmark(group="nearest")
@pytest.mark.parametrize("size", [256, 512, 1024])
def test_rasterio_nearest(benchmark, size):
    src, src_crs, src_transform, dst_crs, dst_transform, dst_shape = _make_test_data(size)

    def _reproject():
        dst = np.full(dst_shape, np.nan, dtype=np.float64)
        rasterio.warp.reproject(
            source=src,
            destination=dst,
            src_transform=rasterio.transform.Affine(*src_transform),
            src_crs=CRS.from_user_input(src_crs),
            dst_transform=rasterio.transform.Affine(*dst_transform),
            dst_crs=CRS.from_user_input(dst_crs),
            resampling=rasterio.warp.Resampling.nearest,
        )
        return dst

    benchmark(_reproject)


# ---------------------------------------------------------------------------
# Bilinear
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="bilinear")
@pytest.mark.parametrize("size", [256, 512, 1024])
def test_rustwarp_bilinear(benchmark, size):
    src, src_crs, src_transform, dst_crs, dst_transform, dst_shape = _make_test_data(size)

    benchmark(
        reproject_array,
        src,
        src_crs,
        src_transform,
        dst_crs,
        dst_transform,
        dst_shape,
        resampling="bilinear",
    )


@pytest.mark.benchmark(group="bilinear")
@pytest.mark.parametrize("size", [256, 512, 1024])
def test_rasterio_bilinear(benchmark, size):
    src, src_crs, src_transform, dst_crs, dst_transform, dst_shape = _make_test_data(size)

    def _reproject():
        dst = np.full(dst_shape, np.nan, dtype=np.float64)
        rasterio.warp.reproject(
            source=src,
            destination=dst,
            src_transform=rasterio.transform.Affine(*src_transform),
            src_crs=CRS.from_user_input(src_crs),
            dst_transform=rasterio.transform.Affine(*dst_transform),
            dst_crs=CRS.from_user_input(dst_crs),
            resampling=rasterio.warp.Resampling.bilinear,
        )
        return dst

    benchmark(_reproject)


# ---------------------------------------------------------------------------
# Cubic
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="cubic")
@pytest.mark.parametrize("size", [256, 512, 1024])
def test_rustwarp_cubic(benchmark, size):
    src, src_crs, src_transform, dst_crs, dst_transform, dst_shape = _make_test_data(size)

    benchmark(
        reproject_array,
        src,
        src_crs,
        src_transform,
        dst_crs,
        dst_transform,
        dst_shape,
        resampling="cubic",
    )


@pytest.mark.benchmark(group="cubic")
@pytest.mark.parametrize("size", [256, 512, 1024])
def test_rasterio_cubic(benchmark, size):
    src, src_crs, src_transform, dst_crs, dst_transform, dst_shape = _make_test_data(size)

    def _reproject():
        dst = np.full(dst_shape, np.nan, dtype=np.float64)
        rasterio.warp.reproject(
            source=src,
            destination=dst,
            src_transform=rasterio.transform.Affine(*src_transform),
            src_crs=CRS.from_user_input(src_crs),
            dst_transform=rasterio.transform.Affine(*dst_transform),
            dst_crs=CRS.from_user_input(dst_crs),
            resampling=rasterio.warp.Resampling.cubic,
        )
        return dst

    benchmark(_reproject)


# ---------------------------------------------------------------------------
# Lanczos
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="lanczos")
@pytest.mark.parametrize("size", [256, 512, 1024])
def test_rustwarp_lanczos(benchmark, size):
    src, src_crs, src_transform, dst_crs, dst_transform, dst_shape = _make_test_data(size)

    benchmark(
        reproject_array,
        src,
        src_crs,
        src_transform,
        dst_crs,
        dst_transform,
        dst_shape,
        resampling="lanczos",
    )


@pytest.mark.benchmark(group="lanczos")
@pytest.mark.parametrize("size", [256, 512, 1024])
def test_rasterio_lanczos(benchmark, size):
    src, src_crs, src_transform, dst_crs, dst_transform, dst_shape = _make_test_data(size)

    def _reproject():
        dst = np.full(dst_shape, np.nan, dtype=np.float64)
        rasterio.warp.reproject(
            source=src,
            destination=dst,
            src_transform=rasterio.transform.Affine(*src_transform),
            src_crs=CRS.from_user_input(src_crs),
            dst_transform=rasterio.transform.Affine(*dst_transform),
            dst_crs=CRS.from_user_input(dst_crs),
            resampling=rasterio.warp.Resampling.lanczos,
        )
        return dst

    benchmark(_reproject)


# ---------------------------------------------------------------------------
# Average
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="average")
@pytest.mark.parametrize(("src_size", "dst_size"), [(256, 64), (512, 128), (1024, 256)])
def test_rustwarp_average(benchmark, src_size, dst_size):
    src, src_crs, src_transform, dst_crs, dst_transform, dst_shape = _make_downscale_data(
        src_size, dst_size
    )

    benchmark(
        reproject_array,
        src,
        src_crs,
        src_transform,
        dst_crs,
        dst_transform,
        dst_shape,
        resampling="average",
    )


@pytest.mark.benchmark(group="average")
@pytest.mark.parametrize(("src_size", "dst_size"), [(256, 64), (512, 128), (1024, 256)])
def test_rasterio_average(benchmark, src_size, dst_size):
    src, src_crs, src_transform, dst_crs, dst_transform, dst_shape = _make_downscale_data(
        src_size, dst_size
    )

    def _reproject():
        dst = np.full(dst_shape, np.nan, dtype=np.float64)
        rasterio.warp.reproject(
            source=src,
            destination=dst,
            src_transform=rasterio.transform.Affine(*src_transform),
            src_crs=CRS.from_user_input(src_crs),
            dst_transform=rasterio.transform.Affine(*dst_transform),
            dst_crs=CRS.from_user_input(dst_crs),
            resampling=rasterio.warp.Resampling.average,
        )
        return dst

    benchmark(_reproject)


# ---------------------------------------------------------------------------
# Plan overhead
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="plan")
@pytest.mark.parametrize("n_tiles", [4, 16, 64, 256, 1024])
def test_plan_reproject_overhead(benchmark, n_tiles):
    """Measure plan_reproject overhead at various tile counts."""
    import math

    tile_size = max(32, int(1024 / math.isqrt(n_tiles)))
    dst_size = tile_size * math.isqrt(n_tiles)

    def _plan():
        return plan_reproject(
            src_crs="EPSG:32633",
            src_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
            src_shape=(1024, 1024),
            dst_crs="EPSG:4326",
            dst_transform=(0.001, 0.0, 14.0, 0.0, -0.001, 60.0),
            dst_shape=(dst_size, dst_size),
            dst_chunks=(tile_size, tile_size),
        )

    benchmark(_plan)


@pytest.mark.benchmark(group="plan")
def test_plan_reproject_sparse_overlap(benchmark):
    """plan_reproject with sparse source coverage — footprint pre-filter active."""

    def _plan():
        return plan_reproject(
            src_crs="EPSG:32633",
            src_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
            src_shape=(64, 64),  # small source
            dst_crs="EPSG:4326",
            dst_transform=(0.001, 0.0, 14.0, 0.0, -0.001, 60.0),
            dst_shape=(512, 512),  # large destination
            dst_chunks=(32, 32),  # 256 tiles total
        )

    # Guard: verify tile count and that the footprint pre-filter is active
    # (source at ~15°E lies outside destination x-range 14°–14.512°E, so
    # all tiles should be fast-rejected with has_data=False).
    result = _plan()
    n_expected = (512 // 32) ** 2  # 256
    assert len(result) == n_expected, f"Expected {n_expected} tiles, got {len(result)}"
    assert not any(t["has_data"] for t in result), (
        "Source (~15°E) is outside destination (14°–14.512°E): all tiles should be no-data"
    )

    benchmark(_plan)


# ---------------------------------------------------------------------------
# Multi-band benchmark
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="multiband")
@pytest.mark.parametrize("n_bands", [3, 64])
def test_rustwarp_multiband(benchmark, n_bands):
    """Multi-band reprojection (band-by-band)."""
    src, src_crs, src_transform, dst_crs, dst_transform, dst_shape = _make_test_data(256)
    bands = [src.copy() for _ in range(n_bands)]

    def _reproject_all():
        return [
            reproject_array(
                band,
                src_crs,
                src_transform,
                dst_crs,
                dst_transform,
                dst_shape,
                resampling="bilinear",
            )
            for band in bands
        ]

    benchmark(_reproject_all)


@pytest.mark.benchmark(group="multiband")
@pytest.mark.parametrize("n_bands", [3, 64])
def test_rasterio_multiband(benchmark, n_bands):
    """Multi-band rasterio reprojection (band-by-band)."""
    src, src_crs, src_transform, dst_crs, dst_transform, dst_shape = _make_test_data(256)
    bands = [src.copy() for _ in range(n_bands)]

    def _reproject_all():
        results = []
        for band in bands:
            dst = np.full(dst_shape, np.nan, dtype=np.float64)
            rasterio.warp.reproject(
                source=band,
                destination=dst,
                src_transform=rasterio.transform.Affine(*src_transform),
                src_crs=CRS.from_user_input(src_crs),
                dst_transform=rasterio.transform.Affine(*dst_transform),
                dst_crs=CRS.from_user_input(dst_crs),
                resampling=rasterio.warp.Resampling.bilinear,
            )
            results.append(dst)
        return results

    benchmark(_reproject_all)


# ---------------------------------------------------------------------------
# Batched vs unbatched dask graph construction
# ---------------------------------------------------------------------------

_BATCH_SRC_GEOBOX = _GeoBox(
    crs="EPSG:32633",
    shape=(1024, 1024),
    affine=(25.0, 0.0, 500000.0, 0.0, -25.0, 6600000.0),
)
_BATCH_DST_GEOBOX = _GeoBox(
    crs="EPSG:4326",
    shape=(1024, 1024),
    affine=(0.001, 0.0, 14.0, 0.0, -0.001, 60.0),
)
_N_VARS = 16
_DST_CHUNKS = (256, 256)


@pytest.mark.benchmark(group="batch_reproject")
def test_bench_graph_construction_unbatched(benchmark):
    """Time to build 16 separate 2D reproject graphs (legacy per-variable path)."""
    single = _da.zeros((1024, 1024), dtype=np.float32, chunks=(256, 256))

    def build():
        return [
            _reproject_dask(
                single,
                _BATCH_SRC_GEOBOX,
                _BATCH_DST_GEOBOX,
                resampling="bilinear",
                dst_chunks=_DST_CHUNKS,
            )
            for _ in range(_N_VARS)
        ]

    benchmark(build)


@pytest.mark.benchmark(group="batch_reproject")
def test_bench_graph_construction_batched(benchmark):
    """Time to build one 3D reproject graph covering 16 variables."""
    single = _da.zeros((1024, 1024), dtype=np.float32, chunks=(256, 256))
    stacked = _da.stack([single] * _N_VARS, axis=0).rechunk({0: _N_VARS})

    def build():
        return _reproject_dask(
            stacked,
            _BATCH_SRC_GEOBOX,
            _BATCH_DST_GEOBOX,
            resampling="bilinear",
            dst_chunks=_DST_CHUNKS,
        )

    benchmark(build)


@pytest.mark.benchmark(group="batch_reproject")
def test_bench_task_count_unbatched(benchmark):
    """Record task count for 16 separate 2D reproject graphs."""
    single = _da.zeros((1024, 1024), dtype=np.float32, chunks=(256, 256))

    def count():
        graphs = [
            _reproject_dask(
                single,
                _BATCH_SRC_GEOBOX,
                _BATCH_DST_GEOBOX,
                resampling="bilinear",
                dst_chunks=_DST_CHUNKS,
            )
            for _ in range(_N_VARS)
        ]
        return sum(len(g.__dask_graph__()) for g in graphs)

    benchmark(count)


@pytest.mark.benchmark(group="batch_reproject")
def test_bench_task_count_batched(benchmark):
    """Record task count for one 3D reproject graph covering 16 variables."""
    single = _da.zeros((1024, 1024), dtype=np.float32, chunks=(256, 256))
    stacked = _da.stack([single] * _N_VARS, axis=0).rechunk({0: _N_VARS})

    def count():
        result = _reproject_dask(
            stacked,
            _BATCH_SRC_GEOBOX,
            _BATCH_DST_GEOBOX,
            resampling="bilinear",
            dst_chunks=_DST_CHUNKS,
        )
        return len(result.__dask_graph__())

    benchmark(count)


@pytest.mark.slow
@pytest.mark.benchmark(group="batch_reproject_compute")
def test_bench_compute_unbatched(benchmark):
    """Compute 16 separate 2D reprojections (threaded scheduler).

    Each variable uses a distinct source array so dask cannot fuse common
    subgraph nodes across the 16 results, giving a fair worst-case comparison.
    """
    import dask

    rng = np.random.default_rng(0)
    # Distinct source arrays — prevents dask from deduplicating source tasks
    # across the 16 reproject graphs when computed together.
    src_arrays = [
        _da.from_array(rng.random((1024, 1024)).astype(np.float32), chunks=(128, 128))
        for _ in range(_N_VARS)
    ]

    def run():
        results = [
            _reproject_dask(
                src,
                _BATCH_SRC_GEOBOX,
                _BATCH_DST_GEOBOX,
                resampling="bilinear",
                dst_chunks=(256, 256),
            )
            for src in src_arrays
        ]
        return dask.compute(*results, scheduler="threads")

    benchmark(run)


@pytest.mark.slow
@pytest.mark.benchmark(group="batch_reproject_compute")
def test_bench_compute_batched(benchmark):
    """Compute 16-variable 3D reproject in one call (threaded scheduler)."""
    src_np = np.random.default_rng(0).random((1024, 1024)).astype(np.float32)
    single = _da.from_array(src_np, chunks=(128, 128))
    stacked = _da.stack([single] * _N_VARS, axis=0).rechunk({0: _N_VARS})

    def run():
        result = _reproject_dask(
            stacked,
            _BATCH_SRC_GEOBOX,
            _BATCH_DST_GEOBOX,
            resampling="bilinear",
            dst_chunks=(256, 256),
        )
        return result.compute(scheduler="threads")

    benchmark(run)
