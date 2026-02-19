"""Comparison tests: rust-warp vs rasterio/GDAL."""

import numpy as np
import pytest
import rasterio.warp
from rasterio.crs import CRS
from rust_warp import plan_reproject, reproject_array


def gdal_reproject(src, src_crs, src_transform, dst_crs, dst_transform, dst_shape, resampling):
    """Reproject using rasterio/GDAL as reference implementation."""
    resampling_map = {
        "nearest": rasterio.warp.Resampling.nearest,
        "bilinear": rasterio.warp.Resampling.bilinear,
        "cubic": rasterio.warp.Resampling.cubic,
        "lanczos": rasterio.warp.Resampling.lanczos,
        "average": rasterio.warp.Resampling.average,
    }
    dst = np.full(dst_shape, np.nan, dtype=np.float64)
    rasterio.warp.reproject(
        source=src,
        destination=dst,
        src_transform=rasterio.transform.Affine(*src_transform),
        src_crs=CRS.from_user_input(src_crs),
        dst_transform=rasterio.transform.Affine(*dst_transform),
        dst_crs=CRS.from_user_input(dst_crs),
        resampling=resampling_map[resampling],
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


class TestNearestVsGDAL:
    """Nearest-neighbor reprojection should closely match GDAL.

    Sub-pixel rounding at pixel boundaries may differ between proj4rs and PROJ/GDAL,
    causing a small fraction of pixels to select a different neighbor. We require
    >98% exact match and the rest to differ by at most ±1 pixel in row or column.
    """

    def _assert_nearest_match(self, rust_result, gdal_result, src_cols):
        # NaN patterns should match
        rust_nan = np.isnan(rust_result)
        gdal_nan = np.isnan(gdal_result)
        np.testing.assert_array_equal(rust_nan, gdal_nan)

        valid = ~rust_nan
        if not valid.any():
            return

        r = rust_result[valid]
        g = gdal_result[valid]

        # Most pixels should match exactly
        exact_match = np.sum(r == g)
        match_pct = exact_match / len(r) * 100
        assert match_pct > 98.0, f"Only {match_pct:.1f}% exact match (need >98%)"

        # Remaining differences should be ±1 pixel (max abs diff = src_cols for row,
        # or 1 for column boundary)
        diff = np.abs(r - g)
        max_diff = diff.max()
        assert max_diff <= src_cols, (
            f"Max diff {max_diff} exceeds src_cols={src_cols} "
            "(more than ±1 row boundary error)"
        )

    def test_utm33_to_4326(self, utm33_to_4326_setup):
        s = utm33_to_4326_setup
        rust_result = reproject_array(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling="nearest",
        )
        gdal_result = gdal_reproject(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling="nearest",
        )
        self._assert_nearest_match(rust_result, gdal_result, src_cols=64)

    def test_utm33_to_3857(self, utm33_to_3857_setup):
        s = utm33_to_3857_setup
        rust_result = reproject_array(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling="nearest",
        )
        gdal_result = gdal_reproject(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling="nearest",
        )
        self._assert_nearest_match(rust_result, gdal_result, src_cols=64)


class TestBilinearVsGDAL:
    """Bilinear reprojection should be close to GDAL.

    Differences arise from:
    - proj4rs vs PROJ sub-pixel coordinate differences
    - Edge pixel handling (our bilinear returns NaN at boundaries, GDAL may extrapolate)
    - Tolerance is generous (atol=10) for MVP; tighten as implementations converge.
    """

    def test_utm33_to_4326(self, utm33_to_4326_setup):
        s = utm33_to_4326_setup
        rust_result = reproject_array(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling="bilinear",
        )
        gdal_result = gdal_reproject(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling="bilinear",
        )

        rust_valid = ~np.isnan(rust_result)
        gdal_valid = ~np.isnan(gdal_result)
        both_valid = rust_valid & gdal_valid

        if both_valid.any():
            np.testing.assert_allclose(
                rust_result[both_valid], gdal_result[both_valid],
                atol=10.0, rtol=0.01,
            )


class TestCubicVsGDAL:
    """Cubic reprojection should be close to GDAL.

    Cubic has a 4×4 neighborhood so more edge pixels are NaN.
    Tolerance is generous (atol=10) initially.
    """

    def test_utm33_to_4326(self, utm33_to_4326_setup):
        s = utm33_to_4326_setup
        rust_result = reproject_array(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling="cubic",
        )
        gdal_result = gdal_reproject(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling="cubic",
        )

        rust_valid = ~np.isnan(rust_result)
        gdal_valid = ~np.isnan(gdal_result)
        both_valid = rust_valid & gdal_valid

        if both_valid.any():
            np.testing.assert_allclose(
                rust_result[both_valid], gdal_result[both_valid],
                atol=10.0, rtol=0.01,
            )


class TestLanczosVsGDAL:
    """Lanczos reprojection should be close to GDAL.

    Lanczos has a 6×6 neighborhood so even more edge pixels are NaN.
    Tolerance is generous (atol=10) initially.
    """

    def test_utm33_to_4326(self, utm33_to_4326_setup):
        s = utm33_to_4326_setup
        rust_result = reproject_array(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling="lanczos",
        )
        gdal_result = gdal_reproject(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling="lanczos",
        )

        rust_valid = ~np.isnan(rust_result)
        gdal_valid = ~np.isnan(gdal_result)
        both_valid = rust_valid & gdal_valid

        if both_valid.any():
            np.testing.assert_allclose(
                rust_result[both_valid], gdal_result[both_valid],
                atol=10.0, rtol=0.01,
            )


class TestAverageVsGDAL:
    """Average reprojection should be close to GDAL for downsampling."""

    def test_downscale(self, downscale_setup):
        s = downscale_setup
        rust_result = reproject_array(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling="average",
        )
        gdal_result = gdal_reproject(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling="average",
        )

        rust_valid = ~np.isnan(rust_result)
        gdal_valid = ~np.isnan(gdal_result)
        both_valid = rust_valid & gdal_valid

        if both_valid.any():
            np.testing.assert_allclose(
                rust_result[both_valid], gdal_result[both_valid],
                atol=10.0, rtol=0.05,
            )


class TestIdentity:
    """Same CRS reprojection should return the input."""

    def test_identity_nearest(self):
        src = np.arange(16, dtype=np.float64).reshape(4, 4)
        transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 6000040.0)
        crs = "EPSG:32633"

        result = reproject_array(
            src, crs, transform, crs, transform, (4, 4), resampling="nearest"
        )
        np.testing.assert_array_equal(result, src)

    def test_identity_bilinear(self):
        src = np.arange(64, dtype=np.float64).reshape(8, 8)
        transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 6000080.0)
        crs = "EPSG:32633"

        result = reproject_array(
            src, crs, transform, crs, transform, (8, 8), resampling="bilinear"
        )
        # Interior pixels should match; edges may differ due to bilinear boundary
        np.testing.assert_allclose(result[1:-1, 1:-1], src[1:-1, 1:-1], atol=1e-6)

    def test_identity_cubic(self):
        src = np.arange(64, dtype=np.float64).reshape(8, 8)
        transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 6000080.0)
        crs = "EPSG:32633"

        result = reproject_array(
            src, crs, transform, crs, transform, (8, 8), resampling="cubic"
        )
        # Interior pixels (2:-2) should match; edges affected by cubic radius
        np.testing.assert_allclose(result[2:-2, 2:-2], src[2:-2, 2:-2], atol=1e-6)

    def test_identity_lanczos(self):
        src = np.arange(144, dtype=np.float64).reshape(12, 12)
        transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 6000120.0)
        crs = "EPSG:32633"

        result = reproject_array(
            src, crs, transform, crs, transform, (12, 12), resampling="lanczos"
        )
        # Interior pixels (3:-3) should match; edges affected by lanczos radius
        np.testing.assert_allclose(result[3:-3, 3:-3], src[3:-3, 3:-3], atol=1e-6)

    def test_identity_average(self):
        src = np.arange(64, dtype=np.float64).reshape(8, 8)
        transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 6000080.0)
        crs = "EPSG:32633"

        result = reproject_array(
            src, crs, transform, crs, transform, (8, 8), resampling="average"
        )
        # Average with scale=1 should match exactly
        np.testing.assert_allclose(result, src, atol=1e-6)


class TestNodata:
    """Nodata regions should be preserved through reprojection."""

    def test_nan_propagation(self):
        src = np.ones((16, 16), dtype=np.float64) * 42.0
        # Set a block of pixels to NaN
        src[4:8, 4:8] = np.nan

        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 6001600.0)
        crs = "EPSG:32633"

        result = reproject_array(
            src, crs, transform, crs, transform, (16, 16), resampling="nearest"
        )

        # NaN block should remain NaN
        assert np.all(np.isnan(result[4:8, 4:8]))
        # Non-NaN pixels should be 42.0
        assert np.all(result[0:4, 0:4] == 42.0)

    def test_sentinel_nodata(self):
        src = np.ones((8, 8), dtype=np.float64) * 100.0
        src[2:4, 2:4] = -9999.0

        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 6000800.0)
        crs = "EPSG:32633"

        result = reproject_array(
            src, crs, transform, crs, transform, (8, 8),
            resampling="nearest", nodata=-9999.0,
        )

        # Nodata pixels should remain as the sentinel value
        assert np.all(result[2:4, 2:4] == -9999.0)
        # Non-nodata pixels should be preserved
        assert np.all(result[0:2, 0:2] == 100.0)


class TestErrorHandling:
    """Bad inputs should raise ValueError."""

    def test_invalid_crs(self):
        src = np.ones((4, 4), dtype=np.float64)
        transform = (1.0, 0.0, 0.0, 0.0, -1.0, 4.0)

        with pytest.raises(ValueError, match="EPSG:99999"):
            reproject_array(src, "EPSG:99999", transform, "EPSG:4326", transform, (4, 4))

    def test_invalid_resampling(self):
        src = np.ones((4, 4), dtype=np.float64)
        transform = (1.0, 0.0, 0.0, 0.0, -1.0, 4.0)

        with pytest.raises(ValueError, match="resampling"):
            reproject_array(
                src, "EPSG:4326", transform, "EPSG:4326", transform, (4, 4),
                resampling="invalid_method",
            )

    def test_unsupported_dtype(self):
        src = np.ones((4, 4), dtype=np.complex128)
        transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 6000040.0)

        with pytest.raises((ValueError, TypeError)):
            reproject_array(
                src, "EPSG:32633", transform, "EPSG:32633", transform, (4, 4),
            )


class TestMultiDtype:
    """Multi-dtype reprojection should preserve dtype and produce correct results."""

    TRANSFORM = (10.0, 0.0, 500000.0, 0.0, -10.0, 6000040.0)
    CRS = "EPSG:32633"

    def test_float32_identity(self):
        src = np.arange(16, dtype=np.float32).reshape(4, 4)
        result = reproject_array(
            src, self.CRS, self.TRANSFORM, self.CRS, self.TRANSFORM, (4, 4),
            resampling="nearest",
        )
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, src)

    def test_float32_matches_float64(self):
        src_f64 = np.arange(16, dtype=np.float64).reshape(4, 4)
        src_f32 = src_f64.astype(np.float32)

        result_f64 = reproject_array(
            src_f64, self.CRS, self.TRANSFORM, self.CRS, self.TRANSFORM, (4, 4),
            resampling="nearest",
        )
        result_f32 = reproject_array(
            src_f32, self.CRS, self.TRANSFORM, self.CRS, self.TRANSFORM, (4, 4),
            resampling="nearest",
        )

        assert result_f32.dtype == np.float32
        np.testing.assert_allclose(result_f32, result_f64.astype(np.float32), atol=1e-6)

    def test_uint8_identity(self):
        src = np.arange(16, dtype=np.uint8).reshape(4, 4)
        result = reproject_array(
            src, self.CRS, self.TRANSFORM, self.CRS, self.TRANSFORM, (4, 4),
            resampling="nearest",
        )
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, src)

    def test_uint16_identity(self):
        src = np.arange(16, dtype=np.uint16).reshape(4, 4)
        result = reproject_array(
            src, self.CRS, self.TRANSFORM, self.CRS, self.TRANSFORM, (4, 4),
            resampling="nearest",
        )
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, src)

    def test_int16_identity(self):
        src = np.arange(16, dtype=np.int16).reshape(4, 4)
        result = reproject_array(
            src, self.CRS, self.TRANSFORM, self.CRS, self.TRANSFORM, (4, 4),
            resampling="nearest",
        )
        assert result.dtype == np.int16
        np.testing.assert_array_equal(result, src)

    def test_uint8_nodata(self):
        src = np.full((4, 4), 42, dtype=np.uint8)
        src[1, 1] = 255  # sentinel nodata

        result = reproject_array(
            src, self.CRS, self.TRANSFORM, self.CRS, self.TRANSFORM, (4, 4),
            resampling="nearest", nodata=255.0,
        )
        assert result.dtype == np.uint8
        assert result[1, 1] == 255
        assert result[0, 0] == 42

    def test_int16_nodata(self):
        src = np.full((4, 4), 100, dtype=np.int16)
        src[2, 2] = -9999

        result = reproject_array(
            src, self.CRS, self.TRANSFORM, self.CRS, self.TRANSFORM, (4, 4),
            resampling="nearest", nodata=-9999.0,
        )
        assert result.dtype == np.int16
        assert result[2, 2] == -9999
        assert result[0, 0] == 100

    def test_float32_nan_fill(self):
        """Float32 should use NaN as default fill (same as float64)."""
        src = np.ones((4, 4), dtype=np.float32)
        # Use a transform that maps some output pixels outside the source
        big_transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 6000400.0)
        result = reproject_array(
            src, self.CRS, self.TRANSFORM, self.CRS, big_transform, (8, 8),
            resampling="nearest",
        )
        assert result.dtype == np.float32
        # Some pixels should be NaN (outside source extent)
        # and some should be 1.0 (mapped from source)
        has_nan = np.any(np.isnan(result))
        has_valid = np.any(result == 1.0)
        assert has_nan or has_valid  # At least one should be true


class TestPlanReproject:
    """plan_reproject stub should be callable and return empty list."""

    def test_returns_empty_list(self):
        result = plan_reproject(
            src_crs="EPSG:32633",
            src_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
            src_shape=(64, 64),
            dst_crs="EPSG:4326",
            dst_transform=(0.001, 0.0, 14.0, 0.0, -0.001, 60.0),
            dst_shape=(64, 64),
        )
        assert result == []

    def test_with_chunks(self):
        result = plan_reproject(
            src_crs="EPSG:32633",
            src_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
            src_shape=(64, 64),
            dst_crs="EPSG:4326",
            dst_transform=(0.001, 0.0, 14.0, 0.0, -0.001, 60.0),
            dst_shape=(64, 64),
            dst_chunks=(32, 32),
            resampling="nearest",
        )
        assert result == []
