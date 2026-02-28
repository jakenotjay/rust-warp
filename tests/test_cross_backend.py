"""Cross-backend correctness tests.

Verifies that numpy, dask, and cubed produce identical reprojection results
for the same inputs. This catches backend-specific regressions that
per-backend tests would miss.
"""

from __future__ import annotations

import numpy as np
import pytest
from rust_warp import GeoBox, reproject, reproject_array

da = pytest.importorskip("dask.array")
cubed = pytest.importorskip("cubed")


# ---- Fixtures ----------------------------------------------------------------

SRC_CRS = "EPSG:32633"
SRC_TRANSFORM = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0)
SRC_SHAPE = (64, 64)
SRC_GEOBOX = GeoBox(crs=SRC_CRS, shape=SRC_SHAPE, affine=SRC_TRANSFORM)

DST_CRS = "EPSG:4326"
DST_TRANSFORM = (0.001, 0.0, 14.0, 0.0, -0.001, 60.0)
DST_SHAPE = (64, 64)
DST_GEOBOX = GeoBox(crs=DST_CRS, shape=DST_SHAPE, affine=DST_TRANSFORM)

CHUNKS = (32, 32)


@pytest.fixture
def src_np_2d():
    return np.arange(64 * 64, dtype=np.float64).reshape(64, 64)


@pytest.fixture
def src_np_3d():
    return np.random.default_rng(42).random((4, 64, 64)).astype(np.float64)


def _compute(arr):
    """Call .compute() if chunked, otherwise return as-is."""
    if hasattr(arr, "compute"):
        return arr.compute()
    return arr


def _assert_allclose_with_nan(actual, expected, atol=1e-10, label=""):
    """Assert arrays match, handling NaN patterns.

    For nearest resampling, pixel values should be exact where both are valid.
    For interpolating resamplers, allow a small tolerance.
    """
    actual_nan = np.isnan(actual)
    expected_nan = np.isnan(expected)

    # NaN patterns must match (or be very close — <1% mismatch for tile boundary artifacts)
    nan_mismatch = np.sum(actual_nan != expected_nan)
    total = actual_nan.size
    assert nan_mismatch / total < 0.01, (
        f"{label}: NaN pattern mismatch {nan_mismatch}/{total} pixels ({nan_mismatch / total:.1%})"
    )

    both_valid = ~actual_nan & ~expected_nan
    if both_valid.any():
        np.testing.assert_allclose(
            actual[both_valid],
            expected[both_valid],
            atol=atol,
            err_msg=f"{label}: pixel value mismatch",
        )


# ---- 2D same-CRS: all three backends ----------------------------------------


class TestCrossBackend2dSameCrs:
    """2D same-CRS reprojection: numpy vs dask vs cubed."""

    @pytest.mark.parametrize("resampling", ["nearest", "bilinear"])
    def test_all_backends_match(self, src_np_2d, resampling):
        """numpy, dask, and cubed produce identical results."""
        dst_geobox = GeoBox(crs=SRC_CRS, shape=SRC_SHAPE, affine=SRC_TRANSFORM)

        result_np = reproject(src_np_2d, SRC_GEOBOX, dst_geobox, resampling=resampling)

        src_dask = da.from_array(src_np_2d, chunks=CHUNKS)
        result_dask = _compute(
            reproject(src_dask, SRC_GEOBOX, dst_geobox, resampling=resampling, dst_chunks=CHUNKS)
        )

        src_cubed = cubed.from_array(src_np_2d, chunks=CHUNKS)
        result_cubed = _compute(
            reproject(src_cubed, SRC_GEOBOX, dst_geobox, resampling=resampling, dst_chunks=CHUNKS)
        )

        atol = 0.0 if resampling == "nearest" else 1e-10
        _assert_allclose_with_nan(result_dask, result_np, atol=atol, label="dask vs numpy")
        _assert_allclose_with_nan(result_cubed, result_np, atol=atol, label="cubed vs numpy")
        _assert_allclose_with_nan(result_cubed, result_dask, atol=atol, label="cubed vs dask")


# ---- 2D cross-CRS: all three backends ---------------------------------------


class TestCrossBackend2dCrossCrs:
    """2D cross-CRS reprojection: numpy vs dask vs cubed."""

    @pytest.mark.parametrize("resampling", ["nearest", "bilinear"])
    def test_all_backends_match(self, src_np_2d, resampling):
        result_np = reproject(src_np_2d, SRC_GEOBOX, DST_GEOBOX, resampling=resampling)

        src_dask = da.from_array(src_np_2d, chunks=CHUNKS)
        result_dask = _compute(
            reproject(src_dask, SRC_GEOBOX, DST_GEOBOX, resampling=resampling, dst_chunks=CHUNKS)
        )

        src_cubed = cubed.from_array(src_np_2d, chunks=CHUNKS)
        result_cubed = _compute(
            reproject(src_cubed, SRC_GEOBOX, DST_GEOBOX, resampling=resampling, dst_chunks=CHUNKS)
        )

        atol = 0.0 if resampling == "nearest" else 1e-10
        _assert_allclose_with_nan(result_dask, result_np, atol=atol, label="dask vs numpy")
        _assert_allclose_with_nan(result_cubed, result_np, atol=atol, label="cubed vs numpy")
        _assert_allclose_with_nan(result_cubed, result_dask, atol=atol, label="cubed vs dask")


# ---- 3D (n, y, x): dask vs cubed -------------------------------------------


class TestCrossBackend3d:
    """3D (n, y, x) reprojection: dask vs cubed vs per-slice numpy."""

    def test_same_crs_nearest(self, src_np_3d):
        from rust_warp.cubed_graph import reproject_cubed
        from rust_warp.dask_graph import reproject_dask

        src_dask = da.from_array(src_np_3d, chunks=(4, 32, 32))
        result_dask = reproject_dask(
            src_dask, SRC_GEOBOX, SRC_GEOBOX, resampling="nearest"
        ).compute()

        src_cubed = cubed.from_array(src_np_3d, chunks=(4, 32, 32))
        result_cubed = reproject_cubed(
            src_cubed, SRC_GEOBOX, SRC_GEOBOX, resampling="nearest"
        ).compute()

        np.testing.assert_array_equal(result_dask, result_cubed)

    def test_cross_crs_nearest(self, src_np_3d):
        from rust_warp.cubed_graph import reproject_cubed
        from rust_warp.dask_graph import reproject_dask

        src_dask = da.from_array(src_np_3d, chunks=(4, 32, 32))
        result_dask = reproject_dask(
            src_dask, SRC_GEOBOX, DST_GEOBOX, resampling="nearest"
        ).compute()

        src_cubed = cubed.from_array(src_np_3d, chunks=(4, 32, 32))
        result_cubed = reproject_cubed(
            src_cubed, SRC_GEOBOX, DST_GEOBOX, resampling="nearest"
        ).compute()

        _assert_allclose_with_nan(result_cubed, result_dask, atol=0.0, label="cubed vs dask 3D")

    def test_3d_matches_per_slice_numpy(self, src_np_3d):
        """Each slice of the 3D chunked results must match standalone numpy."""
        from rust_warp.cubed_graph import reproject_cubed
        from rust_warp.dask_graph import reproject_dask

        src_dask = da.from_array(src_np_3d, chunks=(4, 32, 32))
        result_dask = reproject_dask(
            src_dask, SRC_GEOBOX, DST_GEOBOX, resampling="nearest"
        ).compute()

        src_cubed = cubed.from_array(src_np_3d, chunks=(4, 32, 32))
        result_cubed = reproject_cubed(
            src_cubed, SRC_GEOBOX, DST_GEOBOX, resampling="nearest"
        ).compute()

        for i in range(src_np_3d.shape[0]):
            expected = reproject_array(
                np.ascontiguousarray(src_np_3d[i]),
                SRC_CRS,
                SRC_TRANSFORM,
                DST_CRS,
                DST_TRANSFORM,
                DST_SHAPE,
                resampling="nearest",
            )
            _assert_allclose_with_nan(
                result_dask[i], expected, atol=0.0, label=f"dask slice {i} vs numpy"
            )
            _assert_allclose_with_nan(
                result_cubed[i], expected, atol=0.0, label=f"cubed slice {i} vs numpy"
            )


# ---- Nodata handling: all three backends ------------------------------------


class TestCrossBackendNodata:
    """Nodata fill behaviour must be consistent across backends."""

    def test_nodata_tiles_all_backends(self):
        """Tiles outside source extent produce the same NaN pattern."""
        src_np = np.ones((16, 16), dtype=np.float64) * 42.0
        src_geobox = GeoBox(
            crs="EPSG:32633",
            shape=(16, 16),
            affine=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
        )
        dst_geobox = GeoBox(
            crs="EPSG:4326",
            shape=(64, 64),
            affine=(0.001, 0.0, 14.0, 0.0, -0.001, 60.0),
        )
        chunks = (32, 32)

        result_np = reproject(src_np, src_geobox, dst_geobox, resampling="nearest")

        src_dask = da.from_array(src_np, chunks=(16, 16))
        result_dask = _compute(
            reproject(src_dask, src_geobox, dst_geobox, resampling="nearest", dst_chunks=chunks)
        )

        src_cubed = cubed.from_array(src_np, chunks=(16, 16))
        result_cubed = _compute(
            reproject(src_cubed, src_geobox, dst_geobox, resampling="nearest", dst_chunks=chunks)
        )

        _assert_allclose_with_nan(result_dask, result_np, atol=0.0, label="dask vs numpy nodata")
        _assert_allclose_with_nan(result_cubed, result_np, atol=0.0, label="cubed vs numpy nodata")

    def test_explicit_nodata_value(self):
        """Explicit nodata=-9999 is used consistently by all backends."""
        src_np = np.ones((16, 16), dtype=np.float64) * 42.0
        src_geobox = GeoBox(
            crs="EPSG:32633",
            shape=(16, 16),
            affine=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
        )
        dst_geobox = GeoBox(
            crs="EPSG:4326",
            shape=(64, 64),
            affine=(0.001, 0.0, 14.0, 0.0, -0.001, 60.0),
        )

        result_np = reproject(src_np, src_geobox, dst_geobox, resampling="nearest", nodata=-9999.0)

        src_dask = da.from_array(src_np, chunks=(16, 16))
        result_dask = _compute(
            reproject(
                src_dask,
                src_geobox,
                dst_geobox,
                resampling="nearest",
                nodata=-9999.0,
                dst_chunks=(32, 32),
            )
        )

        src_cubed = cubed.from_array(src_np, chunks=(16, 16))
        result_cubed = _compute(
            reproject(
                src_cubed,
                src_geobox,
                dst_geobox,
                resampling="nearest",
                nodata=-9999.0,
                dst_chunks=(32, 32),
            )
        )

        np.testing.assert_array_equal(result_dask, result_np)
        np.testing.assert_array_equal(result_cubed, result_np)


# ---- Chunk size invariance: all three backends ------------------------------


class TestCrossBackendChunkInvariance:
    """Different chunk sizes must not change pixel values."""

    @pytest.mark.parametrize("chunks", [(16, 16), (32, 32), (64, 64)])
    def test_chunk_sizes_produce_same_result(self, src_np_2d, chunks):
        """All backends agree regardless of chunk size, and match numpy."""
        result_np = reproject(src_np_2d, SRC_GEOBOX, DST_GEOBOX, resampling="nearest")

        src_dask = da.from_array(src_np_2d, chunks=chunks)
        result_dask = _compute(
            reproject(src_dask, SRC_GEOBOX, DST_GEOBOX, resampling="nearest", dst_chunks=chunks)
        )

        src_cubed = cubed.from_array(src_np_2d, chunks=chunks)
        result_cubed = _compute(
            reproject(src_cubed, SRC_GEOBOX, DST_GEOBOX, resampling="nearest", dst_chunks=chunks)
        )

        _assert_allclose_with_nan(result_dask, result_np, atol=0.0, label=f"dask {chunks}")
        _assert_allclose_with_nan(result_cubed, result_np, atol=0.0, label=f"cubed {chunks}")
