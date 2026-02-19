"""Large-scale stress tests for rust-warp.

These tests verify that large rasters can be reprojected without
memory blow-up and with correct results. Marked with @pytest.mark.stress
so they can be excluded from fast CI runs.
"""

import os
import sys

import numpy as np
import pytest
from rust_warp import reproject_array

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import gdal_reproject


@pytest.mark.stress
class TestLargeScale:
    """Large raster reprojection tests."""

    CRS = "EPSG:32633"
    DST_CRS = "EPSG:4326"

    def _make_large_raster(self, size):
        """Create a large synthetic raster."""
        pixel_size = 100.0
        origin_x, origin_y = 500000.0, 6600000.0
        src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

        # Use sinusoidal pattern (bounded values, smooth)
        rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        src = (128.0 + 100.0 * np.sin(2 * np.pi * rows / size) +
               50.0 * np.cos(4 * np.pi * cols / size)).astype(np.float64)

        # Approximate destination
        dst_pixel = 0.001 * (size / 1024)
        dst_transform = (dst_pixel, 0.0, 14.5, 0.0, -dst_pixel * 0.65, 59.6)
        dst_shape = (size, size)

        return src, src_transform, dst_transform, dst_shape

    @pytest.mark.parametrize("kernel", ["nearest", "bilinear", "cubic", "lanczos"])
    def test_4096x4096(self, kernel):
        """4096x4096 reprojection should complete and produce valid data."""
        src, src_transform, dst_transform, dst_shape = self._make_large_raster(4096)

        result = reproject_array(
            src, self.CRS, src_transform,
            self.DST_CRS, dst_transform, dst_shape,
            resampling=kernel,
        )

        assert result.shape == dst_shape
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 10, f"[{kernel}] Only {valid_pct:.0f}% valid pixels"

    def test_4096x4096_average_downscale(self):
        """4096 -> 1024 average downscale."""
        src_size = 4096
        dst_size = 1024
        pixel_size = 100.0
        origin_x, origin_y = 500000.0, 6600000.0

        src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

        rows, cols = np.meshgrid(np.arange(src_size), np.arange(src_size), indexing="ij")
        src = (rows / src_size * 255).astype(np.float64)

        dst_pixel = pixel_size * (src_size / dst_size)
        dst_transform = (dst_pixel, 0.0, origin_x, 0.0, -dst_pixel, origin_y)

        result = reproject_array(
            src, self.CRS, src_transform,
            self.CRS, dst_transform, (dst_size, dst_size),
            resampling="average",
        )

        assert result.shape == (dst_size, dst_size)
        # Average of vertical gradient should still show gradient pattern
        valid = ~np.isnan(result)
        assert valid.sum() > dst_size * dst_size * 0.5

    @pytest.mark.parametrize("kernel", ["nearest", "bilinear"])
    def test_spot_check_vs_gdal(self, kernel):
        """Spot-check 100 random pixels against GDAL for a large raster."""
        size = 2048
        src, src_transform, dst_transform, dst_shape = self._make_large_raster(size)

        rust_result = reproject_array(
            src, self.CRS, src_transform,
            self.DST_CRS, dst_transform, dst_shape,
            resampling=kernel,
        )
        gdal_result = gdal_reproject(
            src, self.CRS, src_transform,
            self.DST_CRS, dst_transform, dst_shape,
            resampling=kernel,
        )

        both_valid = ~np.isnan(rust_result) & ~np.isnan(gdal_result)
        valid_indices = np.argwhere(both_valid)

        if len(valid_indices) < 100:
            pytest.skip("Not enough valid pixels for spot check")

        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(valid_indices), size=100, replace=False)

        diffs = []
        for idx in sample_idx:
            r, c = valid_indices[idx]
            diff = abs(float(rust_result[r, c]) - float(gdal_result[r, c]))
            diffs.append(diff)

        mean_diff = np.mean(diffs)

        if kernel == "nearest":
            # Most should match exactly
            exact = sum(1 for d in diffs if d == 0)
            assert exact > 80, f"Only {exact}/100 exact matches"
        else:
            assert mean_diff < 5.0, f"Mean diff {mean_diff:.2f} too high"
