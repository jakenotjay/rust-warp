"""Edge-case tests for rust-warp reprojection.

Tests: nodata propagation, up/downscaling, small rasters, all-nodata regions.
"""

import os
import sys

import numpy as np
import pytest
from rust_warp import reproject_array

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


KERNELS = ["nearest", "bilinear", "cubic", "lanczos"]
CRS = "EPSG:32633"
BASE_TRANSFORM = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0)


class TestNodataPropagation:
    """Nodata/NaN handling edge cases."""

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_nan_block_propagation(self, kernel):
        """A block of NaN in the source should produce NaN in output."""
        size = 32
        src = np.ones((size, size), dtype=np.float64) * 42.0
        src[8:16, 8:16] = np.nan

        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0 + size * 100.0)

        result = reproject_array(
            src,
            CRS,
            transform,
            CRS,
            transform,
            (size, size),
            resampling=kernel,
        )

        # Interior NaN pixels should remain NaN
        # The exact boundary depends on kernel radius, but the center should be NaN
        center = result[10:14, 10:14]
        assert np.all(np.isnan(center)), f"[{kernel}] NaN block center not preserved"

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_sentinel_nodata_propagation(self, kernel):
        """Integer-style sentinel nodata should be preserved."""
        size = 16
        src = np.ones((size, size), dtype=np.float64) * 100.0
        src[4:8, 4:8] = -9999.0

        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0 + size * 100.0)

        result = reproject_array(
            src,
            CRS,
            transform,
            CRS,
            transform,
            (size, size),
            resampling=kernel,
            nodata=-9999.0,
        )

        if kernel == "nearest":
            # Nearest should preserve the sentinel exactly
            assert np.all(result[4:8, 4:8] == -9999.0)
        else:
            # Interpolating kernels: center of nodata block should be nodata
            center = result[5:7, 5:7]
            assert np.all(center == -9999.0) or np.all(np.isnan(center)), (
                f"[{kernel}] Sentinel nodata not preserved in center"
            )

    def test_nan_propagation_float32(self):
        """NaN propagation should work for float32 dtype."""
        size = 16
        src = np.ones((size, size), dtype=np.float32) * 42.0
        src[4:8, 4:8] = np.nan

        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0 + size * 100.0)

        result = reproject_array(
            src,
            CRS,
            transform,
            CRS,
            transform,
            (size, size),
            resampling="nearest",
        )

        assert result.dtype == np.float32
        assert np.all(np.isnan(result[4:8, 4:8]))


class TestScaling:
    """Upscale and downscale edge cases."""

    @pytest.mark.parametrize("kernel", [*KERNELS, "average"])
    def test_downscale_4x(self, kernel):
        """4x downsampling should produce valid output."""
        src_size = 64
        dst_size = 16
        src = np.arange(src_size * src_size, dtype=np.float64).reshape(src_size, src_size)

        pixel_size = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + src_size * pixel_size
        src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

        dst_pixel = pixel_size * (src_size / dst_size)
        dst_transform = (dst_pixel, 0.0, origin_x, 0.0, -dst_pixel, origin_y)

        result = reproject_array(
            src,
            CRS,
            src_transform,
            CRS,
            dst_transform,
            (dst_size, dst_size),
            resampling=kernel,
        )

        assert result.shape == (dst_size, dst_size)
        # Should have valid (non-NaN) data
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 50, f"[{kernel}] Only {valid_pct:.0f}% valid after 4x downscale"

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_upscale_4x(self, kernel):
        """4x upsampling should produce valid output."""
        src_size = 16
        dst_size = 64
        src = np.arange(src_size * src_size, dtype=np.float64).reshape(src_size, src_size)

        pixel_size = 400.0
        origin_x, origin_y = 500000.0, 6600000.0 + src_size * pixel_size
        src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

        dst_pixel = pixel_size * (src_size / dst_size)
        dst_transform = (dst_pixel, 0.0, origin_x, 0.0, -dst_pixel, origin_y)

        result = reproject_array(
            src,
            CRS,
            src_transform,
            CRS,
            dst_transform,
            (dst_size, dst_size),
            resampling=kernel,
        )

        assert result.shape == (dst_size, dst_size)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        # Lanczos has 6-pixel radius, losing many border pixels on small sources
        min_valid = 30 if kernel == "lanczos" else 50
        assert valid_pct > min_valid, f"[{kernel}] Only {valid_pct:.0f}% valid after 4x upscale"


class TestSmallRasters:
    """Very small rasters should not crash."""

    @pytest.mark.parametrize("size", [4, 8])
    @pytest.mark.parametrize("kernel", [*KERNELS, "average"])
    def test_small_identity(self, size, kernel):
        """Small raster with identity transform should not crash."""
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)
        pixel_size = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * pixel_size
        transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

        result = reproject_array(
            src,
            CRS,
            transform,
            CRS,
            transform,
            (size, size),
            resampling=kernel,
        )
        assert result.shape == (size, size)

    @pytest.mark.parametrize("size", [4, 8])
    def test_small_nearest_identity_exact(self, size):
        """Small raster nearest identity should match exactly."""
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)
        pixel_size = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * pixel_size
        transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

        result = reproject_array(
            src,
            CRS,
            transform,
            CRS,
            transform,
            (size, size),
            resampling="nearest",
        )
        np.testing.assert_array_equal(result, src)


class TestAllNodataRegions:
    """Rasters with all-nodata regions."""

    def test_all_nan_source(self):
        """All-NaN source should produce all-NaN output."""
        size = 16
        src = np.full((size, size), np.nan, dtype=np.float64)
        pixel_size = 100.0
        origin_y = 6600000.0 + size * pixel_size
        transform = (pixel_size, 0.0, 500000.0, 0.0, -pixel_size, origin_y)

        result = reproject_array(
            src,
            CRS,
            transform,
            CRS,
            transform,
            (size, size),
            resampling="nearest",
        )
        assert np.all(np.isnan(result))

    def test_partial_nodata_border(self):
        """Raster with nodata border should have valid interior."""
        size = 32
        src = np.full((size, size), np.nan, dtype=np.float64)
        src[8:24, 8:24] = 42.0  # valid interior
        pixel_size = 100.0
        origin_y = 6600000.0 + size * pixel_size
        transform = (pixel_size, 0.0, 500000.0, 0.0, -pixel_size, origin_y)

        result = reproject_array(
            src,
            CRS,
            transform,
            CRS,
            transform,
            (size, size),
            resampling="nearest",
        )
        # Interior should have valid data
        assert np.any(result == 42.0)
        # Border should be NaN
        assert np.all(np.isnan(result[0, :]))

    @pytest.mark.parametrize("kernel", [*KERNELS, "average"])
    def test_all_sentinel_nodata(self, kernel):
        """All-sentinel source should produce all-sentinel output."""
        size = 16
        src = np.full((size, size), -9999.0, dtype=np.float64)
        pixel_size = 100.0
        origin_y = 6600000.0 + size * pixel_size
        transform = (pixel_size, 0.0, 500000.0, 0.0, -pixel_size, origin_y)

        result = reproject_array(
            src,
            CRS,
            transform,
            CRS,
            transform,
            (size, size),
            resampling=kernel,
            nodata=-9999.0,
        )
        # All output should be nodata (either -9999 or NaN depending on implementation)
        is_nodata = (result == -9999.0) | np.isnan(result)
        assert np.all(is_nodata), f"[{kernel}] Not all output is nodata"
