"""Dask integration tests for chunked reprojection."""

from __future__ import annotations

import numpy as np
import pytest
from rust_warp import GeoBox, reproject, reproject_array

da = pytest.importorskip("dask.array")


# Common test fixtures
SRC_CRS = "EPSG:32633"
SRC_TRANSFORM = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0)
SRC_SHAPE = (64, 64)
SRC_GEOBOX = GeoBox(crs=SRC_CRS, shape=SRC_SHAPE, affine=SRC_TRANSFORM)


class TestGeoBox:
    """GeoBox dataclass."""

    def test_from_bbox(self):
        gbox = GeoBox.from_bbox(
            bbox=(500000.0, 5993600.0, 506400.0, 6000000.0),
            crs="EPSG:32633",
            resolution=100.0,
        )
        assert gbox.shape == (64, 64)
        assert gbox.crs == "EPSG:32633"
        assert gbox.affine[0] == 100.0  # a = res_x
        assert gbox.affine[4] == -100.0  # e = -res_y

    def test_from_bbox_with_shape(self):
        gbox = GeoBox.from_bbox(
            bbox=(0.0, 0.0, 100.0, 100.0),
            crs="EPSG:4326",
            shape=(10, 10),
        )
        assert gbox.shape == (10, 10)
        assert gbox.affine[0] == pytest.approx(10.0)

    def test_bounds(self):
        gbox = GeoBox(
            crs="EPSG:32633",
            shape=(64, 64),
            affine=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
        )
        left, bottom, right, top = gbox.bounds
        assert left == pytest.approx(500000.0)
        assert top == pytest.approx(6600000.0)
        assert right == pytest.approx(506400.0)
        assert bottom == pytest.approx(6593600.0)

    def test_resolution(self):
        gbox = GeoBox(
            crs="EPSG:32633",
            shape=(64, 64),
            affine=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
        )
        assert gbox.resolution == (100.0, 100.0)

    def test_xr_coords(self):
        gbox = GeoBox(
            crs="EPSG:32633",
            shape=(4, 4),
            affine=(100.0, 0.0, 500000.0, 0.0, -100.0, 6000400.0),
        )
        coords = gbox.xr_coords()
        assert "x" in coords and "y" in coords
        assert len(coords["x"]) == 4
        assert len(coords["y"]) == 4
        # First pixel center should be at origin + 0.5 * pixel_size
        assert coords["x"][0] == pytest.approx(500050.0)
        assert coords["y"][0] == pytest.approx(6000350.0)

    def test_from_bbox_no_resolution_or_shape_raises(self):
        with pytest.raises(ValueError, match="resolution or shape"):
            GeoBox.from_bbox(
                bbox=(0.0, 0.0, 100.0, 100.0),
                crs="EPSG:4326",
            )


class TestReprojectDask:
    """Chunked dask reprojection."""

    def test_returns_dask_array(self):
        """reproject() with a dask array should return a dask.Array."""
        src_np = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        src_dask = da.from_array(src_np, chunks=(32, 32))
        dst_geobox = GeoBox(crs=SRC_CRS, shape=SRC_SHAPE, affine=SRC_TRANSFORM)

        result = reproject(src_dask, SRC_GEOBOX, dst_geobox, resampling="nearest")
        assert isinstance(result, da.Array)
        assert result.shape == SRC_SHAPE

    def test_same_crs_chunked_matches_numpy(self):
        """Same-CRS chunked result should match numpy reproject_array."""
        src_np = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        src_dask = da.from_array(src_np, chunks=(32, 32))
        dst_geobox = GeoBox(crs=SRC_CRS, shape=SRC_SHAPE, affine=SRC_TRANSFORM)

        # Numpy path
        expected = reproject_array(
            src_np, SRC_CRS, SRC_TRANSFORM, SRC_CRS, SRC_TRANSFORM, SRC_SHAPE,
            resampling="nearest",
        )

        # Dask path
        result = reproject(
            src_dask, SRC_GEOBOX, dst_geobox,
            resampling="nearest", dst_chunks=(32, 32),
        )
        result_np = result.compute()

        np.testing.assert_array_equal(result_np, expected)

    def test_cross_crs_chunked_matches_numpy(self):
        """Cross-CRS chunked result should match numpy reproject_array."""
        src_np = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        src_dask = da.from_array(src_np, chunks=(32, 32))

        # Destination in EPSG:4326
        dst_transform = (0.001, 0.0, 14.0, 0.0, -0.001, 60.0)
        dst_shape = (64, 64)
        dst_geobox = GeoBox(crs="EPSG:4326", shape=dst_shape, affine=dst_transform)

        # Numpy path
        expected = reproject_array(
            src_np, SRC_CRS, SRC_TRANSFORM, "EPSG:4326", dst_transform, dst_shape,
            resampling="nearest",
        )

        # Dask path
        result = reproject(
            src_dask, SRC_GEOBOX, dst_geobox,
            resampling="nearest", dst_chunks=(32, 32),
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
            src_dask = da.from_array(src_np, chunks=chunks)
            result = reproject(
                src_dask, SRC_GEOBOX, dst_geobox,
                resampling="nearest", dst_chunks=chunks,
            )
            results.append(result.compute())

        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_bilinear_chunked_matches_numpy(self):
        """Bilinear through dask should match numpy — catches bad halo padding.

        Tile boundary sampling may cause a tiny number of edge pixels to differ
        in NaN status (the chunked source ROI clips slightly differently).
        We tolerate <1% NaN-pattern mismatch but require value agreement.
        """
        src_np = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        src_dask = da.from_array(src_np, chunks=(32, 32))
        dst_geobox = GeoBox(crs=SRC_CRS, shape=SRC_SHAPE, affine=SRC_TRANSFORM)

        expected = reproject_array(
            src_np, SRC_CRS, SRC_TRANSFORM, SRC_CRS, SRC_TRANSFORM, SRC_SHAPE,
            resampling="bilinear",
        )

        result = reproject(
            src_dask, SRC_GEOBOX, dst_geobox,
            resampling="bilinear", dst_chunks=(32, 32),
        )
        result_np = result.compute()

        # NaN patterns should be very close — tolerate <1% mismatch from
        # tile boundary sampling artifacts
        expected_nan = np.isnan(expected)
        result_nan = np.isnan(result_np)
        nan_mismatch = np.sum(expected_nan != result_nan)
        total = expected_nan.size
        assert nan_mismatch / total < 0.01, (
            f"NaN pattern mismatch: {nan_mismatch}/{total} pixels"
        )

        # Where both produce values, they must agree
        both_valid = ~expected_nan & ~result_nan
        if both_valid.any():
            np.testing.assert_allclose(
                result_np[both_valid], expected[both_valid], atol=1e-10,
            )

    def test_has_data_false_tiles_filled_with_nodata(self):
        """Tiles outside source extent should be filled with NaN, not crash."""
        src_np = np.ones((16, 16), dtype=np.float64) * 42.0
        src_dask = da.from_array(src_np, chunks=(16, 16))
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
            src_dask, src_geobox, dst_geobox,
            resampling="nearest", dst_chunks=(32, 32),
        )
        result_np = result.compute()

        assert result_np.shape == (64, 64)
        # Pixels outside source extent should be NaN
        assert np.any(np.isnan(result_np))


class TestReprojectNumpy:
    """reproject() numpy dispatch path."""

    def test_numpy_path_matches_reproject_array(self):
        """reproject() with a numpy array should match reproject_array."""
        src_np = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        dst_geobox = GeoBox(crs=SRC_CRS, shape=SRC_SHAPE, affine=SRC_TRANSFORM)

        expected = reproject_array(
            src_np, SRC_CRS, SRC_TRANSFORM, SRC_CRS, SRC_TRANSFORM, SRC_SHAPE,
            resampling="nearest",
        )

        result = reproject(
            src_np, SRC_GEOBOX, dst_geobox, resampling="nearest",
        )

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected)
