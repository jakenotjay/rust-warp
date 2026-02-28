"""Cubed integration tests for chunked reprojection."""

from __future__ import annotations

import numpy as np
import pytest
from rust_warp import GeoBox, reproject, reproject_array

cubed = pytest.importorskip("cubed")


# Common test fixtures
SRC_CRS = "EPSG:32633"
SRC_TRANSFORM = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0)
SRC_SHAPE = (64, 64)
SRC_GEOBOX = GeoBox(crs=SRC_CRS, shape=SRC_SHAPE, affine=SRC_TRANSFORM)

DST_GEOBOX = GeoBox(
    crs="EPSG:4326",
    shape=(64, 64),
    affine=(0.001, 0.0, 14.0, 0.0, -0.001, 60.0),
)


class TestReprojectCubed:
    """Chunked cubed reprojection."""

    def test_returns_cubed_array(self):
        """reproject() with a cubed array should return a cubed.Array."""
        src_np = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        src_cubed = cubed.from_array(src_np, chunks=(32, 32))
        dst_geobox = GeoBox(crs=SRC_CRS, shape=SRC_SHAPE, affine=SRC_TRANSFORM)

        result = reproject(src_cubed, SRC_GEOBOX, dst_geobox, resampling="nearest")
        assert isinstance(result, cubed.Array)
        assert result.shape == SRC_SHAPE

    def test_same_crs_chunked_matches_numpy(self):
        """Same-CRS cubed result should match numpy reproject_array."""
        src_np = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        src_cubed = cubed.from_array(src_np, chunks=(32, 32))
        dst_geobox = GeoBox(crs=SRC_CRS, shape=SRC_SHAPE, affine=SRC_TRANSFORM)

        # Numpy path
        expected = reproject_array(
            src_np,
            SRC_CRS,
            SRC_TRANSFORM,
            SRC_CRS,
            SRC_TRANSFORM,
            SRC_SHAPE,
            resampling="nearest",
        )

        # Cubed path
        result = reproject(
            src_cubed,
            SRC_GEOBOX,
            dst_geobox,
            resampling="nearest",
            dst_chunks=(32, 32),
        )
        result_np = result.compute()

        np.testing.assert_array_equal(result_np, expected)

    def test_cross_crs_chunked_matches_numpy(self):
        """Cross-CRS cubed result should match numpy reproject_array."""
        src_np = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        src_cubed = cubed.from_array(src_np, chunks=(32, 32))

        dst_transform = (0.001, 0.0, 14.0, 0.0, -0.001, 60.0)
        dst_shape = (64, 64)
        dst_geobox = GeoBox(crs="EPSG:4326", shape=dst_shape, affine=dst_transform)

        # Numpy path
        expected = reproject_array(
            src_np,
            SRC_CRS,
            SRC_TRANSFORM,
            "EPSG:4326",
            dst_transform,
            dst_shape,
            resampling="nearest",
        )

        # Cubed path
        result = reproject(
            src_cubed,
            SRC_GEOBOX,
            dst_geobox,
            resampling="nearest",
            dst_chunks=(32, 32),
        )
        result_np = result.compute()

        # Both should have the same NaN pattern
        expected_nan = np.isnan(expected)
        result_nan = np.isnan(result_np)
        np.testing.assert_array_equal(expected_nan, result_nan)

        # Where both are valid, values should match
        both_valid = ~expected_nan & ~result_nan
        if both_valid.any():
            np.testing.assert_array_equal(result_np[both_valid], expected[both_valid])

    def test_different_chunk_sizes_same_result(self):
        """Different chunk sizes should produce the same result."""
        src_np = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        dst_geobox = GeoBox(crs=SRC_CRS, shape=SRC_SHAPE, affine=SRC_TRANSFORM)

        results = []
        for chunks in [(16, 16), (32, 32), (64, 64)]:
            src_cubed = cubed.from_array(src_np, chunks=chunks)
            result = reproject(
                src_cubed,
                SRC_GEOBOX,
                dst_geobox,
                resampling="nearest",
                dst_chunks=chunks,
            )
            results.append(result.compute())

        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_has_data_false_tiles_filled_with_nodata(self):
        """Tiles outside source extent should be filled with NaN, not crash."""
        src_np = np.ones((16, 16), dtype=np.float64) * 42.0
        src_cubed = cubed.from_array(src_np, chunks=(16, 16))
        src_geobox = GeoBox(
            crs="EPSG:32633",
            shape=(16, 16),
            affine=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
        )
        # Destination is far away — most tiles should have has_data=False
        dst_geobox = GeoBox(
            crs="EPSG:4326",
            shape=(64, 64),
            affine=(0.001, 0.0, 14.0, 0.0, -0.001, 60.0),
        )

        result = reproject(
            src_cubed,
            src_geobox,
            dst_geobox,
            resampling="nearest",
            dst_chunks=(32, 32),
        )
        result_np = result.compute()

        assert result_np.shape == (64, 64)
        assert np.any(np.isnan(result_np))


class TestReprojectCubed3d:
    """3D (n, y, x) cubed reprojection."""

    def test_3d_returns_3d_cubed_array(self):
        from rust_warp.cubed_graph import reproject_cubed

        src_np = np.random.default_rng(0).random((4, 64, 64)).astype(np.float32)
        src_cubed = cubed.from_array(src_np, chunks=(4, 32, 32))
        result = reproject_cubed(src_cubed, SRC_GEOBOX, SRC_GEOBOX, resampling="nearest")
        assert isinstance(result, cubed.Array)
        assert result.ndim == 3
        assert result.shape == (4, 64, 64)

    def test_3d_matches_per_slice_2d(self):
        """Each slice reprojected in a 3D batch must match standalone 2D reproject."""
        from rust_warp.cubed_graph import reproject_cubed

        src_np = np.random.default_rng(1).random((4, 64, 64)).astype(np.float32)
        src_cubed = cubed.from_array(src_np, chunks=(4, 32, 32))
        result_3d = reproject_cubed(
            src_cubed, SRC_GEOBOX, DST_GEOBOX, resampling="nearest"
        ).compute()
        for i in range(4):
            slice_cubed = cubed.from_array(src_np[i], chunks=(32, 32))
            expected = reproject_cubed(
                slice_cubed, SRC_GEOBOX, DST_GEOBOX, resampling="nearest"
            ).compute()
            np.testing.assert_array_equal(result_3d[i], expected)

    def test_3d_no_data_tiles_filled(self):
        """Tiles outside source extent are filled with NaN, not zero."""
        from rust_warp.cubed_graph import reproject_cubed

        src_np = np.ones((2, 16, 16), dtype=np.float32)
        src_cubed = cubed.from_array(src_np, chunks=(2, 16, 16))
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
        result = reproject_cubed(
            src_cubed, src_geobox, dst_geobox, resampling="nearest", dst_chunks=(32, 32)
        ).compute()
        assert result.shape == (2, 64, 64)
        assert np.any(np.isnan(result))


class TestDetectBackend:
    """Backend detection utilities."""

    def test_cubed_detected(self):
        from rust_warp._backend import detect_backend

        arr = cubed.from_array(np.zeros((4, 4)), chunks=(2, 2))
        assert detect_backend(arr) == "cubed"

    def test_numpy_returns_none(self):
        from rust_warp._backend import detect_backend

        assert detect_backend(np.zeros((4, 4))) is None

    def test_is_chunked(self):
        from rust_warp._backend import is_chunked

        arr = cubed.from_array(np.zeros((4, 4)), chunks=(2, 2))
        assert is_chunked(arr) is True
        assert is_chunked(np.zeros((4, 4))) is False
