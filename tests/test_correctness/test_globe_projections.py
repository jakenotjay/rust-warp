"""Full-globe and exotic projection tests.

Tests reprojecting data that spans large extents to various projection families:
sinusoidal, polar stereographic, Mercator bounds, and cross-CRS chains.
"""

import numpy as np
import pytest
import rasterio.warp
from rasterio.crs import CRS
from rust_warp import reproject_array

KERNELS = ["nearest", "bilinear", "cubic", "lanczos"]


def _gdal_reproject(src, src_crs, src_transform, dst_crs, dst_transform, dst_shape, resampling):
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


def _make_global_4326(size=64):
    """Create a global EPSG:4326 raster."""
    # Cover -180 to 180 lon, -80 to 80 lat (avoid poles for most projections)
    left, right = -180.0, 180.0
    bottom, top = -80.0, 80.0
    res_x = (right - left) / size
    res_y = (top - bottom) / size
    transform = (res_x, 0.0, left, 0.0, -res_y, top)
    src = np.arange(size * size, dtype=np.float64).reshape(size, size)
    return src, "EPSG:4326", transform


class TestGlobalToWebMercator:
    """EPSG:4326 global extent to Web Mercator (EPSG:3857)."""

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_global_to_3857(self, kernel):
        """Global data should reproject to Web Mercator without crashing."""
        src, src_crs, src_transform = _make_global_4326(size=64)

        # Web Mercator valid range is roughly ±85° latitude
        dst_transform_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input(src_crs),
            CRS.from_user_input("EPSG:3857"),
            64,
            64,
            left=-180.0,
            bottom=-80.0,
            right=180.0,
            top=80.0,
        )
        dst_transform = tuple(dst_transform_affine)[:6]

        result = reproject_array(
            src,
            src_crs,
            src_transform,
            "EPSG:3857",
            dst_transform,
            (dst_h, dst_w),
            resampling=kernel,
        )

        assert result.shape == (dst_h, dst_w)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 30, f"[{kernel}] Only {valid_pct:.0f}% valid"

    def test_global_3857_matches_gdal(self):
        """Global to Web Mercator nearest should broadly match GDAL."""
        src, src_crs, src_transform = _make_global_4326(size=32)

        dst_transform_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input(src_crs),
            CRS.from_user_input("EPSG:3857"),
            32,
            32,
            left=-180.0,
            bottom=-80.0,
            right=180.0,
            top=80.0,
        )
        dst_transform = tuple(dst_transform_affine)[:6]
        dst_shape = (dst_h, dst_w)

        rust = reproject_array(
            src,
            src_crs,
            src_transform,
            "EPSG:3857",
            dst_transform,
            dst_shape,
            resampling="nearest",
        )
        gdal = _gdal_reproject(
            src,
            src_crs,
            src_transform,
            "EPSG:3857",
            dst_transform,
            dst_shape,
            resampling="nearest",
        )

        both_valid = ~np.isnan(rust) & ~np.isnan(gdal)
        if both_valid.sum() > 0:
            diff = np.abs(rust[both_valid] - gdal[both_valid])
            match_pct = np.sum(diff == 0) / both_valid.sum() * 100
            assert match_pct > 80 or diff.max() < 200, (
                f"Only {match_pct:.0f}% exact match, max_diff={diff.max():.1f}"
            )


class TestPolarStereographic:
    """Polar stereographic projections for Arctic/Antarctic data."""

    def test_arctic_to_north_polar_stereo(self):
        """Arctic data (EPSG:4326) to North Polar Stereographic (EPSG:3413)."""
        size = 32
        # Arctic region: 60-90°N, 0-90°E
        res = 1.0
        src_transform = (res, 0.0, 0.0, 0.0, -res, 90.0)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        dst_transform_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:4326"),
            CRS.from_user_input("EPSG:3413"),
            size,
            size,
            left=0.0,
            bottom=90.0 - size * res,
            right=size * res,
            top=90.0,
        )
        dst_transform = tuple(dst_transform_affine)[:6]

        result = reproject_array(
            src,
            "EPSG:4326",
            src_transform,
            "EPSG:3413",
            dst_transform,
            (dst_h, dst_w),
            resampling="nearest",
        )

        assert result.shape == (dst_h, dst_w)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 20, f"Only {valid_pct:.0f}% valid"

    def test_antarctic_to_south_polar_stereo(self):
        """Antarctic data to South Polar Stereographic (EPSG:3031)."""
        size = 32
        # Antarctic region: 60-90°S, 0-90°E
        res = 1.0
        src_transform = (res, 0.0, 0.0, 0.0, -res, -60.0)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        dst_transform_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:4326"),
            CRS.from_user_input("EPSG:3031"),
            size,
            size,
            left=0.0,
            bottom=-90.0,
            right=size * res,
            top=-60.0,
        )
        dst_transform = tuple(dst_transform_affine)[:6]

        result = reproject_array(
            src,
            "EPSG:4326",
            src_transform,
            "EPSG:3031",
            dst_transform,
            (dst_h, dst_w),
            resampling="nearest",
        )

        assert result.shape == (dst_h, dst_w)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 20, f"Only {valid_pct:.0f}% valid"

    def test_polar_stereo_roundtrip(self):
        """North polar stereo data should survive roundtrip to 4326 and back."""
        size = 32
        pixel_size = 50000.0  # 50km pixels
        origin_x, origin_y = -1000000.0, 1000000.0
        src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        # Forward: polar stereo -> 4326
        mid_affine, mid_w, mid_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:3413"),
            CRS.from_user_input("EPSG:4326"),
            size,
            size,
            left=origin_x,
            bottom=origin_y - size * pixel_size,
            right=origin_x + size * pixel_size,
            top=origin_y,
        )
        mid_transform = tuple(mid_affine)[:6]

        mid = reproject_array(
            src,
            "EPSG:3413",
            src_transform,
            "EPSG:4326",
            mid_transform,
            (mid_h, mid_w),
            resampling="nearest",
        )

        # Inverse: 4326 -> polar stereo
        result = reproject_array(
            mid,
            "EPSG:4326",
            mid_transform,
            "EPSG:3413",
            src_transform,
            (size, size),
            resampling="nearest",
        )

        # Roundtrip with nearest through different CRS loses precision;
        # just verify we get valid data back (not all NaN)
        both_valid = ~np.isnan(result) & ~np.isnan(src)
        assert both_valid.sum() > 0, "No valid pixels survived polar stereo roundtrip"
        # Values should be in the same range as source
        diff = np.abs(result[both_valid] - src[both_valid])
        assert diff.mean() < src.max(), (
            f"Mean error {diff.mean():.0f} exceeds source range {src.max():.0f}"
        )


class TestCrossProjectionChains:
    """Multi-hop CRS chains to check consistency."""

    def test_utm_to_4326_to_3857_vs_direct(self):
        """UTM -> 4326 -> 3857 should be close to UTM -> 3857 directly."""
        size = 32
        pixel_size = 500.0
        origin_x, origin_y = 500000.0, 6600000.0
        src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        # Direct: UTM33 -> 3857
        direct_affine, direct_w, direct_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:32633"),
            CRS.from_user_input("EPSG:3857"),
            size,
            size,
            left=origin_x,
            bottom=origin_y - size * pixel_size,
            right=origin_x + size * pixel_size,
            top=origin_y,
        )
        direct_transform = tuple(direct_affine)[:6]

        direct = reproject_array(
            src,
            "EPSG:32633",
            src_transform,
            "EPSG:3857",
            direct_transform,
            (direct_h, direct_w),
            resampling="bilinear",
        )

        # Two-hop: UTM33 -> 4326 -> 3857
        mid_affine, mid_w, mid_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:32633"),
            CRS.from_user_input("EPSG:4326"),
            size,
            size,
            left=origin_x,
            bottom=origin_y - size * pixel_size,
            right=origin_x + size * pixel_size,
            top=origin_y,
        )
        mid_transform = tuple(mid_affine)[:6]

        mid = reproject_array(
            src,
            "EPSG:32633",
            src_transform,
            "EPSG:4326",
            mid_transform,
            (mid_h, mid_w),
            resampling="bilinear",
        )

        hop2 = reproject_array(
            mid,
            "EPSG:4326",
            mid_transform,
            "EPSG:3857",
            direct_transform,
            (direct_h, direct_w),
            resampling="bilinear",
        )

        # Two-hop introduces extra interpolation error but should be broadly similar
        both_valid = ~np.isnan(direct) & ~np.isnan(hop2)
        if both_valid.sum() > 10:
            diff = np.abs(direct[both_valid] - hop2[both_valid])
            # Two-hop adds error but values should be in same ballpark
            assert diff.mean() < 100, f"Mean diff {diff.mean():.1f} between direct and 2-hop"


class TestLargeExtentReprojections:
    """Reprojections covering large geographic extents."""

    def test_hemisphere_to_utm(self):
        """An entire hemisphere in 4326 projected to a single UTM zone."""
        size = 32
        # Northern hemisphere, western half: -180 to 0 lon, 0 to 90 lat
        res = 5.0  # 5° pixels
        src_transform = (res, 0.0, -180.0, 0.0, -res, 90.0)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        # Project to UTM zone 17 — most data will be outside the zone
        dst_transform_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:4326"),
            CRS.from_user_input("EPSG:32617"),
            size,
            size,
            left=-180.0,
            bottom=0.0,
            right=0.0,
            top=90.0,
        )
        dst_transform = tuple(dst_transform_affine)[:6]

        result = reproject_array(
            src,
            "EPSG:4326",
            src_transform,
            "EPSG:32617",
            dst_transform,
            (dst_h, dst_w),
            resampling="nearest",
        )

        # Should not crash; most pixels will be NaN (outside UTM zone)
        assert result.shape == (dst_h, dst_w)
        assert not np.all(np.isnan(result)), "All pixels are NaN — no data mapped"

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_wide_longitude_band(self, kernel):
        """A wide longitude band (120° wide) reprojected to UTM."""
        size = 32
        # -30° to 90° longitude, 40° to 60° latitude
        res = 120.0 / size
        src_transform = (res, 0.0, -30.0, 0.0, -res, 60.0)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:4326"),
            CRS.from_user_input("EPSG:32633"),
            size,
            size,
            left=-30.0,
            bottom=40.0,
            right=90.0,
            top=60.0,
        )
        dst_transform = tuple(dst_affine)[:6]

        result = reproject_array(
            src,
            "EPSG:4326",
            src_transform,
            "EPSG:32633",
            dst_transform,
            (dst_h, dst_w),
            resampling=kernel,
        )

        assert result.shape == (dst_h, dst_w)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 10, f"[{kernel}] Only {valid_pct:.0f}% valid"


class TestHighLatitude:
    """Projections at extreme latitudes where distortion is large."""

    def test_near_pole_to_utm(self):
        """Data at 85°N should reproject to UTM without NaN everywhere."""
        size = 16
        res = 0.1
        src_transform = (res, 0.0, 10.0, 0.0, -res, 86.0)
        src = np.ones((size, size), dtype=np.float64) * 42.0

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:4326"),
            CRS.from_user_input("EPSG:32633"),
            size,
            size,
            left=10.0,
            bottom=86.0 - size * res,
            right=10.0 + size * res,
            top=86.0,
        )
        dst_transform = tuple(dst_affine)[:6]

        result = reproject_array(
            src,
            "EPSG:4326",
            src_transform,
            "EPSG:32633",
            dst_transform,
            (dst_h, dst_w),
            resampling="nearest",
        )

        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 20, f"Only {valid_pct:.0f}% valid at high latitude"

    def test_equator_to_utm(self):
        """Data exactly on the equator should reproject cleanly."""
        size = 32
        res = 0.01
        src_transform = (res, 0.0, 14.5, 0.0, -res, 0.16)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:4326"),
            CRS.from_user_input("EPSG:32633"),
            size,
            size,
            left=14.5,
            bottom=0.16 - size * res,
            right=14.5 + size * res,
            top=0.16,
        )
        dst_transform = tuple(dst_affine)[:6]

        result = reproject_array(
            src,
            "EPSG:4326",
            src_transform,
            "EPSG:32633",
            dst_transform,
            (dst_h, dst_w),
            resampling="nearest",
        )

        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 50, f"Only {valid_pct:.0f}% valid at equator"
