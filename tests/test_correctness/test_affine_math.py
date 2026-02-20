"""Affine transform math tests.

Tests affine transform construction, decomposition, composition,
and edge cases in the Rust affine module via the Python API.
"""

import numpy as np
import pytest
from rust_warp import reproject_array, transform_grid

CRS_STR = "EPSG:32633"


class TestAffineComposition:
    """Test that affine transforms compose correctly through reproject_array."""

    def test_double_resolution_halves_pixels(self):
        """Doubling resolution should produce 2x as many pixels."""
        size = 16
        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * px
        src_transform = (px, 0.0, origin_x, 0.0, -px, origin_y)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        # Double resolution: half the pixel size, double the output size
        dst_transform = (px / 2, 0.0, origin_x, 0.0, -px / 2, origin_y)
        dst_size = size * 2

        result = reproject_array(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling="nearest",
        )

        # Each src pixel should map to a 2x2 block in output
        for r in range(size):
            for c in range(size):
                block = result[r * 2 : (r + 1) * 2, c * 2 : (c + 1) * 2]
                if not np.any(np.isnan(block)):
                    np.testing.assert_allclose(block, src[r, c], atol=1e-6)

    def test_half_resolution_doubles_pixels(self):
        """Halving resolution should produce half as many pixels."""
        size = 16
        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * px
        src_transform = (px, 0.0, origin_x, 0.0, -px, origin_y)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        # Half resolution: double the pixel size, half the output size
        dst_transform = (px * 2, 0.0, origin_x, 0.0, -px * 2, origin_y)
        dst_size = size // 2

        result = reproject_array(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling="nearest",
        )

        assert result.shape == (dst_size, dst_size)
        # Output pixel (0,0) should sample from around src pixel (1,1)
        assert not np.all(np.isnan(result))

    def test_translation_shifts_pixels(self):
        """Translating the destination should shift the output."""
        size = 16
        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * px
        src_transform = (px, 0.0, origin_x, 0.0, -px, origin_y)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        # Shift destination by 3 pixels east
        shift = 3
        dst_transform = (px, 0.0, origin_x + shift * px, 0.0, -px, origin_y)

        result = reproject_array(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (size, size),
            resampling="nearest",
        )

        # result[r, 0] should equal src[r, 3] (shifted 3 cols)
        for r in range(size):
            for c in range(size - shift):
                if not np.isnan(result[r, c]):
                    np.testing.assert_allclose(result[r, c], src[r, c + shift], atol=1e-6)


class TestAffineInverse:
    """Test affine inverse through transform_grid."""

    def test_grid_identity_is_pixel_centers(self):
        """Identity grid should produce pixel-center coordinates."""
        size = 8
        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0
        transform = (px, 0.0, origin_x, 0.0, -px, origin_y)

        col_grid, row_grid = transform_grid(
            CRS_STR,
            transform,
            CRS_STR,
            transform,
            (size, size),
        )

        for r in range(size):
            for c in range(size):
                np.testing.assert_allclose(col_grid[r, c], c + 0.5, atol=0.01)
                np.testing.assert_allclose(row_grid[r, c], r + 0.5, atol=0.01)

    def test_grid_preserves_pixel_count(self):
        """Grid dimensions should match requested shape."""
        for rows, cols in [(8, 8), (16, 32), (1, 100), (100, 1)]:
            transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0)
            col_grid, row_grid = transform_grid(
                CRS_STR,
                transform,
                CRS_STR,
                transform,
                (rows, cols),
            )
            assert col_grid.shape == (rows, cols)
            assert row_grid.shape == (rows, cols)


class TestAffineScaleVariants:
    """Test various scale factors and their effects."""

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 4.0, 10.0])
    def test_isotropic_scale(self, scale):
        """Isotropic scaling should work at various factors."""
        size = 16
        px_src = 100.0
        px_dst = px_src * scale
        origin_x, origin_y = 500000.0, 6600000.0 + size * px_src
        src_transform = (px_src, 0.0, origin_x, 0.0, -px_src, origin_y)

        dst_size = max(1, int(size / scale))
        dst_transform = (px_dst, 0.0, origin_x, 0.0, -px_dst, origin_y)

        src = np.arange(size * size, dtype=np.float64).reshape(size, size)
        result = reproject_array(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (dst_size, dst_size),
            resampling="nearest",
        )

        assert result.shape == (dst_size, dst_size)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 50, f"Scale {scale}: only {valid_pct:.0f}% valid"

    @pytest.mark.parametrize(
        ("scale_x", "scale_y"),
        [
            (0.5, 2.0),
            (2.0, 0.5),
            (1.0, 3.0),
            (3.0, 1.0),
        ],
    )
    def test_anisotropic_scale(self, scale_x, scale_y):
        """Anisotropic scaling should produce correct output shape."""
        size = 16
        px_src = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * px_src
        src_transform = (px_src, 0.0, origin_x, 0.0, -px_src, origin_y)

        dst_cols = max(1, int(size / scale_x))
        dst_rows = max(1, int(size / scale_y))
        px_dst_x = px_src * scale_x
        px_dst_y = px_src * scale_y
        dst_transform = (px_dst_x, 0.0, origin_x, 0.0, -px_dst_y, origin_y)

        src = np.arange(size * size, dtype=np.float64).reshape(size, size)
        result = reproject_array(
            src,
            CRS_STR,
            src_transform,
            CRS_STR,
            dst_transform,
            (dst_rows, dst_cols),
            resampling="nearest",
        )

        assert result.shape == (dst_rows, dst_cols)


class TestAffineNumericalStability:
    """Numerical stability of affine operations."""

    def test_very_small_pixels(self):
        """Very small pixel sizes should not cause numerical issues."""
        size = 16
        px = 0.001  # 1mm pixels
        origin_x, origin_y = 500000.0, 6600000.0 + size * px
        transform = (px, 0.0, origin_x, 0.0, -px, origin_y)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        result = reproject_array(
            src,
            CRS_STR,
            transform,
            CRS_STR,
            transform,
            (size, size),
            resampling="nearest",
        )

        np.testing.assert_array_equal(result, src)

    def test_very_large_pixels(self):
        """Very large pixel sizes should not cause numerical issues."""
        size = 8
        px = 100000.0  # 100km pixels
        origin_x, origin_y = 500000.0, 6600000.0 + size * px
        transform = (px, 0.0, origin_x, 0.0, -px, origin_y)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        result = reproject_array(
            src,
            CRS_STR,
            transform,
            CRS_STR,
            transform,
            (size, size),
            resampling="nearest",
        )

        np.testing.assert_array_equal(result, src)

    def test_large_origin_coordinates(self):
        """Very large origin coordinates should not lose precision."""
        size = 8
        px = 100.0
        # EPSG:32633 can have large northings
        origin_x, origin_y = 500000.0, 9999999.0
        transform = (px, 0.0, origin_x, 0.0, -px, origin_y)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        result = reproject_array(
            src,
            CRS_STR,
            transform,
            CRS_STR,
            transform,
            (size, size),
            resampling="nearest",
        )

        np.testing.assert_array_equal(result, src)

    def test_fractional_pixel_size(self):
        """Non-integer pixel sizes should work correctly."""
        size = 16
        px = 33.333
        origin_x, origin_y = 500000.0, 6600000.0 + size * px
        transform = (px, 0.0, origin_x, 0.0, -px, origin_y)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        result = reproject_array(
            src,
            CRS_STR,
            transform,
            CRS_STR,
            transform,
            (size, size),
            resampling="nearest",
        )

        np.testing.assert_array_equal(result, src)
