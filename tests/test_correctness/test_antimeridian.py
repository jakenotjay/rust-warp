"""Antimeridian / dateline handling tests.

Tests that reprojection across the 180° longitude discontinuity produces
correct results without artifacts, missing data, or coordinate wrapping errors.
"""

import numpy as np
import pytest
import rasterio.warp
from rasterio.crs import CRS

from rust_warp import reproject_array, transform_points


KERNELS = ["nearest", "bilinear", "cubic", "lanczos"]


def _gdal_reproject(src, src_crs, src_transform, dst_crs, dst_transform, dst_shape, resampling):
    """Reference GDAL reprojection."""
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


class TestAntimeridianTransformPoints:
    """Coordinate transforms across the 180° discontinuity."""

    def test_points_near_dateline_roundtrip(self):
        """Points near ±180° should survive a roundtrip to UTM and back."""
        # Points straddling the dateline in EPSG:4326
        x = np.array([179.5, 179.9, -179.9, -179.5], dtype=np.float64)
        y = np.array([60.0, 60.0, 60.0, 60.0], dtype=np.float64)

        # Transform to UTM zone 1 (centered at 177°W) which spans the dateline
        x_utm, y_utm = transform_points(x, y, "EPSG:4326", "EPSG:32601")
        x_back, y_back = transform_points(x_utm, y_utm, "EPSG:32601", "EPSG:4326")

        # Roundtrip should recover original coordinates (mod 360)
        # Normalize to [-180, 180]
        x_back_norm = ((x_back + 180) % 360) - 180
        np.testing.assert_allclose(x_back_norm, x, atol=1e-6)
        np.testing.assert_allclose(y_back, y, atol=1e-6)

    def test_points_at_exactly_180(self):
        """Coordinates at exactly ±180° should not produce NaN or inf."""
        x = np.array([180.0, -180.0], dtype=np.float64)
        y = np.array([0.0, 0.0], dtype=np.float64)

        x_out, y_out = transform_points(x, y, "EPSG:4326", "EPSG:3857")
        assert np.all(np.isfinite(x_out))
        assert np.all(np.isfinite(y_out))

    def test_transform_across_dateline_consistent(self):
        """Points on either side of dateline should map to consistent UTM coords."""
        # Two points that are geographically close but on opposite sides of 180°
        x = np.array([179.99, -179.99], dtype=np.float64)
        y = np.array([60.0, 60.0], dtype=np.float64)

        x_utm, y_utm = transform_points(x, y, "EPSG:4326", "EPSG:32601")

        # These points are ~0.02° apart, so UTM eastings should be close
        # (not separated by millions of meters from wrapping error)
        easting_diff = abs(x_utm[0] - x_utm[1])
        assert easting_diff < 5000, (
            f"Points 0.02° apart have {easting_diff:.0f}m UTM separation — "
            "likely a dateline wrapping error"
        )


class TestAntimeridianReproject:
    """Reprojection of rasters near the antimeridian.

    Since EPSG:4326 uses [-180, 180], we test dateline behavior by:
    1. Creating UTM zone 1/60 data (which straddle the dateline)
    2. Reprojecting to/from 4326
    3. Testing 4326 data on each side of the dateline independently
    """

    def _make_utm_near_dateline(self, size=32, zone=1):
        """Create a UTM raster near the dateline."""
        crs = f"EPSG:3260{zone}" if zone >= 10 else f"EPSG:32601"
        pixel_size = 1000.0
        origin_x, origin_y = 400000.0, 6600000.0
        transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)
        return src, crs, transform, pixel_size

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_utm_zone1_to_4326_valid(self, kernel):
        """UTM zone 1 (near dateline) to 4326 should produce valid output."""
        size = 32
        src, src_crs, src_transform, px = self._make_utm_near_dateline(size, zone=1)

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input(src_crs),
            CRS.from_user_input("EPSG:4326"),
            size, size,
            left=400000.0, bottom=6600000.0 - size * 1000.0,
            right=400000.0 + size * 1000.0, top=6600000.0,
        )
        dst_transform = tuple(dst_affine)[:6]

        result = reproject_array(
            src, src_crs, src_transform,
            "EPSG:4326", dst_transform, (dst_h, dst_w),
            resampling=kernel,
        )

        assert result.shape == (dst_h, dst_w)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 30, (
            f"[{kernel}] Only {valid_pct:.0f}% valid pixels near dateline"
        )

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_4326_east_of_dateline_to_utm(self, kernel):
        """4326 data just east of dateline (170-179°E) to UTM zone 60."""
        size = 32
        res = 0.3
        src_transform = (res, 0.0, 170.0, 0.0, -res, 65.0)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:4326"),
            CRS.from_user_input("EPSG:32660"),
            size, size,
            left=170.0, bottom=65.0 - size * res, right=170.0 + size * res, top=65.0,
        )
        dst_transform = tuple(dst_affine)[:6]

        result = reproject_array(
            src, "EPSG:4326", src_transform,
            "EPSG:32660", dst_transform, (dst_h, dst_w),
            resampling=kernel,
        )

        assert result.shape == (dst_h, dst_w)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 30, (
            f"[{kernel}] Only {valid_pct:.0f}% valid pixels east of dateline"
        )

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_4326_west_of_dateline_to_utm(self, kernel):
        """4326 data just west of dateline (-179 to -170°) to UTM zone 1."""
        size = 32
        res = 0.3
        src_transform = (res, 0.0, -179.0, 0.0, -res, 65.0)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:4326"),
            CRS.from_user_input("EPSG:32601"),
            size, size,
            left=-179.0, bottom=65.0 - size * res,
            right=-179.0 + size * res, top=65.0,
        )
        dst_transform = tuple(dst_affine)[:6]

        result = reproject_array(
            src, "EPSG:4326", src_transform,
            "EPSG:32601", dst_transform, (dst_h, dst_w),
            resampling=kernel,
        )

        assert result.shape == (dst_h, dst_w)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 30, (
            f"[{kernel}] Only {valid_pct:.0f}% valid pixels west of dateline"
        )

    def test_dateline_nearest_matches_gdal(self):
        """UTM zone 1 to 4326 nearest should match GDAL near dateline."""
        size = 32
        src, src_crs, src_transform, px = self._make_utm_near_dateline(size, zone=1)

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input(src_crs),
            CRS.from_user_input("EPSG:4326"),
            size, size,
            left=400000.0, bottom=6600000.0 - size * 1000.0,
            right=400000.0 + size * 1000.0, top=6600000.0,
        )
        dst_transform = tuple(dst_affine)[:6]
        dst_shape = (dst_h, dst_w)

        rust_result = reproject_array(
            src, src_crs, src_transform,
            "EPSG:4326", dst_transform, dst_shape,
            resampling="nearest",
        )
        gdal_result = _gdal_reproject(
            src, src_crs, src_transform,
            "EPSG:4326", dst_transform, dst_shape,
            resampling="nearest",
        )

        rust_valid = ~np.isnan(rust_result)
        gdal_valid = ~np.isnan(gdal_result)
        both_valid = rust_valid & gdal_valid

        if both_valid.sum() > 0:
            diff = np.abs(rust_result[both_valid] - gdal_result[both_valid])
            match_pct = np.sum(diff == 0) / both_valid.sum() * 100
            assert match_pct > 80 or diff.max() < 100, (
                f"Only {match_pct:.0f}% exact match, max_diff={diff.max():.1f}"
            )


class TestAntimeridianUTMToGeographic:
    """UTM rasters near the dateline reprojected to geographic CRS."""

    def test_utm_zone1_to_4326(self):
        """UTM zone 1 (177°W center) reprojected to 4326 should produce valid output."""
        size = 32
        # UTM zone 1 raster
        pixel_size = 1000.0
        origin_x, origin_y = 400000.0, 6600000.0
        src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        dst_transform_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:32601"),
            CRS.from_user_input("EPSG:4326"),
            size, size,
            left=origin_x,
            bottom=origin_y - size * pixel_size,
            right=origin_x + size * pixel_size,
            top=origin_y,
        )
        dst_transform = tuple(dst_transform_affine)[:6]

        result = reproject_array(
            src, "EPSG:32601", src_transform,
            "EPSG:4326", dst_transform, (dst_h, dst_w),
            resampling="nearest",
        )

        assert result.shape == (dst_h, dst_w)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 50, f"Only {valid_pct:.0f}% valid pixels"

    def test_utm_zone60_to_4326(self):
        """UTM zone 60 (near 180°E) reprojected to 4326."""
        size = 32
        pixel_size = 1000.0
        origin_x, origin_y = 400000.0, 6600000.0
        src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        dst_transform_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:32660"),
            CRS.from_user_input("EPSG:4326"),
            size, size,
            left=origin_x,
            bottom=origin_y - size * pixel_size,
            right=origin_x + size * pixel_size,
            top=origin_y,
        )
        dst_transform = tuple(dst_transform_affine)[:6]

        result = reproject_array(
            src, "EPSG:32660", src_transform,
            "EPSG:4326", dst_transform, (dst_h, dst_w),
            resampling="nearest",
        )

        assert result.shape == (dst_h, dst_w)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 50, f"Only {valid_pct:.0f}% valid pixels"
