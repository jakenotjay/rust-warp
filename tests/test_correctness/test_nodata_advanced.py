"""Advanced nodata edge case tests.

Tests nodata handling beyond the basics: average resampling NaN propagation,
nodata at boundaries, mixed nodata patterns, and cross-dtype nodata behavior.
"""

import numpy as np
import pytest
import rasterio.warp
from rasterio.crs import CRS
from rust_warp import reproject_array

KERNELS = ["nearest", "bilinear", "cubic", "lanczos"]
CRS_STR = "EPSG:32633"


def _make_identity_transform(size, pixel_size=100.0):
    """Create a same-CRS transform for testing."""
    origin_y = 6600000.0 + size * pixel_size
    return (pixel_size, 0.0, 500000.0, 0.0, -pixel_size, origin_y)


def _gdal_reproject(
    src, src_crs, src_transform, dst_crs, dst_transform, dst_shape, resampling, nodata=np.nan
):
    resampling_map = {
        "nearest": rasterio.warp.Resampling.nearest,
        "bilinear": rasterio.warp.Resampling.bilinear,
        "cubic": rasterio.warp.Resampling.cubic,
        "lanczos": rasterio.warp.Resampling.lanczos,
        "average": rasterio.warp.Resampling.average,
    }
    if np.issubdtype(src.dtype, np.floating):
        dst = np.full(dst_shape, np.nan, dtype=src.dtype)
        src_nodata = np.nan if np.isnan(nodata) else nodata
        dst_nodata = np.nan if np.isnan(nodata) else nodata
    else:
        fill = int(nodata) if not np.isnan(nodata) else 0
        dst = np.full(dst_shape, fill, dtype=src.dtype)
        src_nodata = fill if np.isnan(nodata) else int(nodata)
        dst_nodata = src_nodata
    rasterio.warp.reproject(
        source=src,
        destination=dst,
        src_transform=rasterio.transform.Affine(*src_transform),
        src_crs=CRS.from_user_input(src_crs),
        dst_transform=rasterio.transform.Affine(*dst_transform),
        dst_crs=CRS.from_user_input(dst_crs),
        resampling=resampling_map[resampling],
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
    )
    return dst


class TestAverageNodataHandling:
    """Average resampling with various nodata patterns."""

    def test_average_nan_propagation_downscale(self):
        """NaN pixels in average downscaling should propagate correctly."""
        src_size = 64
        dst_size = 16
        src = np.ones((src_size, src_size), dtype=np.float64) * 100.0
        # Set a quadrant to NaN
        src[:32, :32] = np.nan
        transform_src = _make_identity_transform(src_size)

        dst_px = 100.0 * (src_size / dst_size)
        origin_y = 6600000.0 + src_size * 100.0
        dst_transform = (dst_px, 0.0, 500000.0, 0.0, -dst_px, origin_y)

        result = reproject_array(
            src,
            CRS_STR,
            transform_src,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling="average",
        )

        # Top-left quadrant of output should be NaN (all source pixels were NaN)
        assert np.all(np.isnan(result[:4, :4])), "NaN quadrant not preserved in average"
        # Bottom-right quadrant should have valid data
        valid_br = result[8:, 8:]
        assert np.any(~np.isnan(valid_br)), "Valid quadrant lost in average"

    def test_average_partial_nan_block(self):
        """Average with partially NaN 4x4 blocks should still produce values."""
        src_size = 32
        dst_size = 8  # 4x downscale
        src = np.ones((src_size, src_size), dtype=np.float64) * 50.0
        # Set half of each 4x4 block to NaN (scattered pattern)
        for r in range(0, src_size, 4):
            for c in range(0, src_size, 4):
                src[r : r + 2, c : c + 2] = np.nan

        transform_src = _make_identity_transform(src_size)
        dst_px = 100.0 * (src_size / dst_size)
        origin_y = 6600000.0 + src_size * 100.0
        dst_transform = (dst_px, 0.0, 500000.0, 0.0, -dst_px, origin_y)

        result = reproject_array(
            src,
            CRS_STR,
            transform_src,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling="average",
        )

        # Should have some valid output — average of valid pixels in each block
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 0, "Average with partial NaN produced no valid output"

    def test_average_sentinel_nodata_downscale(self):
        """Average downscaling with sentinel nodata."""
        src_size = 32
        dst_size = 8
        src = np.arange(src_size * src_size, dtype=np.float64).reshape(src_size, src_size)
        src[0:8, 0:8] = -9999.0

        transform_src = _make_identity_transform(src_size)
        dst_px = 100.0 * (src_size / dst_size)
        origin_y = 6600000.0 + src_size * 100.0
        dst_transform = (dst_px, 0.0, 500000.0, 0.0, -dst_px, origin_y)

        result = reproject_array(
            src,
            CRS_STR,
            transform_src,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling="average",
            nodata=-9999.0,
        )

        # Top-left corner of output corresponds to src[0:8,0:8] which is all nodata
        # Check [0,0] — the center of the all-nodata block
        tl_val = result[0, 0]
        is_nodata = (tl_val == -9999.0) or np.isnan(tl_val)
        assert is_nodata, f"Center of all-nodata block has value {tl_val}, expected -9999 or NaN"


class TestNodataAtBoundaries:
    """Nodata handling at image boundaries during reprojection."""

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_nodata_border_cross_crs(self, kernel):
        """Nodata border should be preserved through cross-CRS reprojection."""
        size = 32
        src = np.full((size, size), np.nan, dtype=np.float64)
        # Only the center 16x16 has valid data
        src[8:24, 8:24] = np.arange(256, dtype=np.float64).reshape(16, 16)

        pixel_size = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * pixel_size
        src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input(CRS_STR),
            CRS.from_user_input("EPSG:4326"),
            size,
            size,
            left=origin_x,
            bottom=origin_y - size * pixel_size,
            right=origin_x + size * pixel_size,
            top=origin_y,
        )
        dst_transform = tuple(dst_affine)[:6]

        result = reproject_array(
            src,
            CRS_STR,
            src_transform,
            "EPSG:4326",
            dst_transform,
            (dst_h, dst_w),
            resampling=kernel,
        )

        # Should have both valid and NaN pixels
        has_valid = np.any(~np.isnan(result))
        has_nan = np.any(np.isnan(result))
        assert has_valid, f"[{kernel}] No valid pixels in output"
        assert has_nan, f"[{kernel}] No NaN pixels — border not preserved"

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_single_valid_row(self, kernel):
        """A raster with only one valid row should not crash."""
        size = 16
        src = np.full((size, size), np.nan, dtype=np.float64)
        src[size // 2, :] = np.arange(size, dtype=np.float64)
        transform = _make_identity_transform(size)

        result = reproject_array(
            src,
            CRS_STR,
            transform,
            CRS_STR,
            transform,
            (size, size),
            resampling=kernel,
        )

        assert result.shape == (size, size)
        if kernel == "nearest":
            np.testing.assert_array_equal(result[size // 2, :], src[size // 2, :])

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_single_valid_column(self, kernel):
        """A raster with only one valid column should not crash."""
        size = 16
        src = np.full((size, size), np.nan, dtype=np.float64)
        src[:, size // 2] = np.arange(size, dtype=np.float64)
        transform = _make_identity_transform(size)

        result = reproject_array(
            src,
            CRS_STR,
            transform,
            CRS_STR,
            transform,
            (size, size),
            resampling=kernel,
        )

        assert result.shape == (size, size)
        if kernel == "nearest":
            np.testing.assert_array_equal(result[:, size // 2], src[:, size // 2])


class TestNodataPatterns:
    """Various spatial patterns of nodata."""

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_checkerboard_nodata(self, kernel):
        """Checkerboard NaN pattern should not crash and should preserve pattern."""
        size = 16
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)
        r, c = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        src[(r + c) % 2 == 0] = np.nan
        transform = _make_identity_transform(size)

        result = reproject_array(
            src,
            CRS_STR,
            transform,
            CRS_STR,
            transform,
            (size, size),
            resampling=kernel,
        )

        assert result.shape == (size, size)
        if kernel == "nearest":
            # Nearest should preserve the checkerboard exactly
            np.testing.assert_array_equal(np.isnan(result), np.isnan(src))

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_diagonal_nodata_stripe(self, kernel):
        """Diagonal stripe of NaN across the raster."""
        size = 32
        src = np.ones((size, size), dtype=np.float64) * 42.0
        for i in range(size):
            src[i, i] = np.nan
            if i + 1 < size:
                src[i, i + 1] = np.nan
        transform = _make_identity_transform(size)

        result = reproject_array(
            src,
            CRS_STR,
            transform,
            CRS_STR,
            transform,
            (size, size),
            resampling=kernel,
        )

        assert result.shape == (size, size)
        # Should have both valid and NaN
        assert np.any(np.isnan(result))
        assert np.any(~np.isnan(result))

    def test_scattered_nodata_pixels(self):
        """Randomly scattered NaN pixels — 25% nodata."""
        size = 32
        rng = np.random.default_rng(42)
        src = rng.random((size, size)) * 1000.0
        mask = rng.random((size, size)) < 0.25
        src[mask] = np.nan
        transform = _make_identity_transform(size)

        for kernel in KERNELS:
            result = reproject_array(
                src,
                CRS_STR,
                transform,
                CRS_STR,
                transform,
                (size, size),
                resampling=kernel,
            )
            assert result.shape == (size, size)
            if kernel == "nearest":
                np.testing.assert_array_equal(np.isnan(result), np.isnan(src))


class TestSentinelNodataValues:
    """Different sentinel nodata values across data types."""

    @pytest.mark.parametrize("nodata_val", [0.0, -9999.0, -1e38, 9.969209968386869e36])
    def test_float64_sentinel_values(self, nodata_val):
        """Various float64 sentinel values should round-trip through nearest."""
        size = 16
        src = np.ones((size, size), dtype=np.float64) * 42.0
        src[4:8, 4:8] = nodata_val
        transform = _make_identity_transform(size)

        result = reproject_array(
            src,
            CRS_STR,
            transform,
            CRS_STR,
            transform,
            (size, size),
            resampling="nearest",
            nodata=nodata_val,
        )

        assert np.all(result[4:8, 4:8] == nodata_val), f"Sentinel {nodata_val} not preserved"
        assert np.all(result[0:4, 0:4] == 42.0)

    @pytest.mark.parametrize(
        ("dtype", "nodata_val"),
        [
            (np.uint8, 0),
            (np.uint8, 255),
            (np.uint16, 0),
            (np.uint16, 65535),
            (np.int16, -9999),
            (np.int16, -32768),
        ],
    )
    def test_integer_sentinel_values(self, dtype, nodata_val):
        """Integer sentinel values should be preserved through nearest."""
        size = 8
        src = np.full((size, size), 42, dtype=dtype)
        src[2:4, 2:4] = nodata_val
        transform = _make_identity_transform(size)

        result = reproject_array(
            src,
            CRS_STR,
            transform,
            CRS_STR,
            transform,
            (size, size),
            resampling="nearest",
            nodata=float(nodata_val),
        )

        assert result.dtype == dtype
        assert np.all(result[2:4, 2:4] == nodata_val)
        assert np.all(result[0:2, 0:2] == 42)


class TestNodataCrossCRSConsistency:
    """Nodata handling should be consistent between same-CRS and cross-CRS paths."""

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_nodata_pattern_preserved_cross_crs(self, kernel):
        """NaN pattern should survive UTM -> 4326 -> UTM roundtrip (approximately)."""
        size = 32
        src = np.ones((size, size), dtype=np.float64) * 100.0
        src[:8, :] = np.nan  # top 8 rows are NaN

        pixel_size = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * pixel_size
        src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

        # UTM -> 4326
        mid_affine, mid_w, mid_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input(CRS_STR),
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
            CRS_STR,
            src_transform,
            "EPSG:4326",
            mid_transform,
            (mid_h, mid_w),
            resampling=kernel,
        )

        # 4326 -> UTM
        result = reproject_array(
            mid,
            "EPSG:4326",
            mid_transform,
            CRS_STR,
            src_transform,
            (size, size),
            resampling=kernel,
        )

        # The NaN region should still be substantially NaN after roundtrip
        top_nan_pct = np.sum(np.isnan(result[:4, 4:-4])) / result[:4, 4:-4].size * 100
        assert top_nan_pct >= 50, (
            f"[{kernel}] NaN region eroded: only {top_nan_pct:.0f}% NaN after roundtrip"
        )
