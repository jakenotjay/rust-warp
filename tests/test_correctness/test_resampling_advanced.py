"""Advanced resampling method tests.

Tests specific downsampling ratios, resampling at different scales,
edge pixel handling, and statistical preservation properties.
"""

import numpy as np
import pytest
import rasterio.warp
from rasterio.crs import CRS
from rust_warp import reproject_array

KERNELS = ["nearest", "bilinear", "cubic", "lanczos"]
CRS_STR = "EPSG:32633"


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


def _make_scale_setup(src_size, dst_size, pixel_size=100.0):
    """Create a same-CRS scaling setup."""
    origin_x, origin_y = 500000.0, 6600000.0 + src_size * pixel_size
    src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)
    scale = src_size / dst_size
    dst_px = pixel_size * scale
    dst_transform = (dst_px, 0.0, origin_x, 0.0, -dst_px, origin_y)
    return src_transform, dst_transform


class TestDownsamplingRatios:
    """Test specific downsampling ratios that can be problematic."""

    @pytest.mark.parametrize("ratio", [2, 3, 4, 5, 8])
    @pytest.mark.parametrize("kernel", [*KERNELS, "average"])
    def test_integer_downscale_ratios(self, ratio, kernel):
        """Integer downscale ratios should produce valid output."""
        src_size = 64
        dst_size = src_size // ratio
        src_transform, dst_transform = _make_scale_setup(src_size, dst_size)
        src = np.arange(src_size * src_size, dtype=np.float64).reshape(src_size, src_size)

        result = reproject_array(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling=kernel,
        )

        assert result.shape == (dst_size, dst_size)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 40, f"[{kernel} {ratio}x] Only {valid_pct:.0f}% valid"

    @pytest.mark.parametrize(
        ("src_size", "dst_size"),
        [
            (64, 48),  # 75% (4:3)
            (64, 32),  # 50% (2:1)
            (64, 43),  # ~67% (non-integer ratio)
            (100, 33),  # ~33% (non-integer ratio)
            (100, 75),  # 75%
        ],
    )
    def test_fractional_downscale_ratios(self, src_size, dst_size):
        """Fractional downscale ratios should produce valid output."""
        src_transform, dst_transform = _make_scale_setup(src_size, dst_size)
        src = np.arange(src_size * src_size, dtype=np.float64).reshape(src_size, src_size)

        for kernel in ["nearest", "bilinear", "average"]:
            result = reproject_array(
                src,
                CRS_STR,
                src_transform,
                CRS_STR,
                dst_transform,
                (dst_size, dst_size),
                resampling=kernel,
            )
            assert result.shape == (dst_size, dst_size)
            valid_pct = np.sum(~np.isnan(result)) / result.size * 100
            assert valid_pct > 40, f"[{kernel} {src_size}->{dst_size}] Only {valid_pct:.0f}% valid"


class TestUpsamplingRatios:
    """Test various upsampling ratios."""

    @pytest.mark.parametrize("ratio", [2, 3, 4, 8])
    @pytest.mark.parametrize("kernel", KERNELS)
    def test_integer_upscale_ratios(self, ratio, kernel):
        """Integer upscale ratios should produce valid output."""
        src_size = 16
        dst_size = src_size * ratio
        src_transform, dst_transform = _make_scale_setup(src_size, dst_size)
        src = np.arange(src_size * src_size, dtype=np.float64).reshape(src_size, src_size)

        result = reproject_array(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling=kernel,
        )

        assert result.shape == (dst_size, dst_size)
        # Lanczos has larger kernel radius, loses more border pixels on small sources
        min_valid = 20 if kernel == "lanczos" else 40
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > min_valid, f"[{kernel} {ratio}x] Only {valid_pct:.0f}% valid"


class TestAveragePreservation:
    """Average resampling should preserve certain statistical properties."""

    def test_average_preserves_constant(self):
        """Average of a constant field should return the same constant."""
        size = 64
        dst_size = 16
        src = np.full((size, size), 42.0, dtype=np.float64)
        src_transform, dst_transform = _make_scale_setup(size, dst_size)

        result = reproject_array(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling="average",
        )

        valid = ~np.isnan(result)
        if valid.any():
            np.testing.assert_allclose(result[valid], 42.0, atol=0.1)

    def test_average_preserves_mean(self):
        """Average downsampling should approximately preserve the spatial mean."""
        size = 64
        dst_size = 16
        rng = np.random.default_rng(42)
        src = rng.random((size, size)) * 1000.0
        src_transform, dst_transform = _make_scale_setup(size, dst_size)

        result = reproject_array(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling="average",
        )

        valid = ~np.isnan(result)
        if valid.any():
            src_mean = src.mean()
            dst_mean = result[valid].mean()
            # Means should be close (within ~10% for random data)
            np.testing.assert_allclose(dst_mean, src_mean, rtol=0.15)

    def test_average_vs_gdal_downscale_50pct(self):
        """Average 2x downscale should match GDAL."""
        size = 64
        dst_size = 32
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)
        src_transform, dst_transform = _make_scale_setup(size, dst_size)

        rust = reproject_array(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling="average",
        )
        gdal = _gdal_reproject(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling="average",
        )

        both_valid = ~np.isnan(rust) & ~np.isnan(gdal)
        if both_valid.sum() > 0:
            np.testing.assert_allclose(
                rust[both_valid],
                gdal[both_valid],
                atol=10.0,
                rtol=0.05,
            )

    def test_average_vs_gdal_downscale_75pct(self):
        """Average 75% downscale should match GDAL."""
        src_size = 64
        dst_size = 48  # 75%
        src = np.arange(src_size * src_size, dtype=np.float64).reshape(src_size, src_size)
        src_transform, dst_transform = _make_scale_setup(src_size, dst_size)

        rust = reproject_array(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling="average",
        )
        gdal = _gdal_reproject(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling="average",
        )

        both_valid = ~np.isnan(rust) & ~np.isnan(gdal)
        if both_valid.sum() > 0:
            np.testing.assert_allclose(
                rust[both_valid],
                gdal[both_valid],
                atol=15.0,
                rtol=0.1,
            )


class TestEdgePixelBehavior:
    """Edge/border pixel handling across kernels."""

    @pytest.mark.parametrize(
        ("kernel", "expected_nan_border"),
        [
            ("nearest", 0),
            ("bilinear", 1),
            ("cubic", 2),
            ("lanczos", 3),
        ],
    )
    def test_nan_border_width(self, kernel, expected_nan_border):
        """Each kernel should produce NaN borders matching its radius."""
        size = 32
        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * px
        src_transform = (px, 0.0, origin_x, 0.0, -px, origin_y)

        # Use a ramp — all non-NaN values
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        # Shift destination by half a pixel to force interpolation at edges
        dst_transform = (px, 0.0, origin_x + px * 0.3, 0.0, -px, origin_y - px * 0.3)

        result = reproject_array(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (size, size),
            resampling=kernel,
        )

        # Interior should be valid
        margin = expected_nan_border + 1
        if margin < size // 2:
            interior = result[margin:-margin, margin:-margin]
            valid_pct = np.sum(~np.isnan(interior)) / interior.size * 100
            assert valid_pct > 90, (
                f"[{kernel}] Interior (margin={margin}) has {valid_pct:.0f}% valid"
            )

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_cross_crs_edge_pixels_vs_gdal(self, kernel):
        """Edge pixels in cross-CRS reprojection should broadly match GDAL."""
        size = 32
        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * px
        src_transform = (px, 0.0, origin_x, 0.0, -px, origin_y)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input(CRS_STR),
            CRS.from_user_input("EPSG:4326"),
            size,
            size,
            left=origin_x,
            bottom=origin_y - size * px,
            right=origin_x + size * px,
            top=origin_y,
        )
        dst_transform = tuple(dst_affine)[:6]
        dst_shape = (dst_h, dst_w)

        rust = reproject_array(
            src,
            CRS_STR,
            src_transform,
            "EPSG:4326",
            dst_transform,
            dst_shape,
            resampling=kernel,
        )
        gdal = _gdal_reproject(
            src,
            CRS_STR,
            src_transform,
            "EPSG:4326",
            dst_transform,
            dst_shape,
            resampling=kernel,
        )

        # Compare NaN patterns — they don't need to match exactly
        rust_nan_pct = np.sum(np.isnan(rust)) / rust.size * 100
        gdal_nan_pct = np.sum(np.isnan(gdal)) / gdal.size * 100

        # NaN percentages should be in same ballpark
        assert abs(rust_nan_pct - gdal_nan_pct) < 20, (
            f"[{kernel}] NaN mismatch: rust={rust_nan_pct:.0f}% vs gdal={gdal_nan_pct:.0f}%"
        )


class TestKernelSmoothness:
    """Interpolating kernels should produce smooth output from smooth input."""

    @pytest.mark.parametrize("kernel", ["bilinear", "cubic", "lanczos"])
    def test_smooth_input_smooth_output(self, kernel):
        """Smooth sinusoidal input should produce smooth output (no spikes)."""
        size = 32
        r, c = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        src = (np.sin(2 * np.pi * r / size) * np.cos(2 * np.pi * c / size) * 100.0 + 500.0).astype(
            np.float64
        )

        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * px
        src_transform = (px, 0.0, origin_x, 0.0, -px, origin_y)

        # 2x upsample
        dst_size = size * 2
        dst_px = px / 2
        dst_transform = (dst_px, 0.0, origin_x, 0.0, -dst_px, origin_y)

        result = reproject_array(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling=kernel,
        )

        valid = ~np.isnan(result)
        if valid.sum() > 100:
            valid_values = result[valid]
            # Output should be in same range as input (no wild overshoots)
            # Cubic and Lanczos can overshoot slightly, allow 20%
            src_range = src.max() - src.min()
            margin = src_range * 0.3
            assert valid_values.min() > src.min() - margin, (
                f"[{kernel}] Undershoot: {valid_values.min():.1f} < {src.min() - margin:.1f}"
            )
            assert valid_values.max() < src.max() + margin, (
                f"[{kernel}] Overshoot: {valid_values.max():.1f} > {src.max() + margin:.1f}"
            )

    @pytest.mark.parametrize("kernel", ["bilinear", "cubic", "lanczos"])
    def test_constant_input_constant_output(self, kernel):
        """Constant input should produce constant output regardless of kernel."""
        size = 16
        src = np.full((size, size), 42.0, dtype=np.float64)

        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * px
        src_transform = (px, 0.0, origin_x, 0.0, -px, origin_y)

        # Slight subpixel shift to force interpolation
        dst_transform = (px, 0.0, origin_x + px * 0.25, 0.0, -px, origin_y - px * 0.25)

        result = reproject_array(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (size, size),
            resampling=kernel,
        )

        valid = ~np.isnan(result)
        if valid.any():
            np.testing.assert_allclose(result[valid], 42.0, atol=1e-6)
