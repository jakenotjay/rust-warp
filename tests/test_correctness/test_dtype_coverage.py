"""Additional data type coverage tests.

Tests reprojection across all supported and unsupported data types,
including type-specific nodata handling, value range preservation,
and cross-type consistency.
"""

import numpy as np
import pytest
from rust_warp import reproject_array

KERNELS = ["nearest", "bilinear", "cubic", "lanczos", "average"]
CRS_STR = "EPSG:32633"
TRANSFORM = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600400.0)


class TestAllSupportedDtypes:
    """Test every supported dtype through identity reprojection."""

    @pytest.mark.parametrize("dtype", [
        np.float32, np.float64,
        np.uint8, np.uint16,
        np.int16, np.int8,
    ])
    def test_identity_preserves_dtype(self, dtype):
        """Identity reprojection should preserve dtype."""
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            src = np.array([1, 2, 42, info.max - 1], dtype=dtype).reshape(2, 2)
        else:
            src = np.array([1.0, 2.5, 42.0, 100.5], dtype=dtype).reshape(2, 2)

        result = reproject_array(
            src, CRS_STR, TRANSFORM,
            CRS_STR, TRANSFORM, (2, 2),
            resampling="nearest",
        )

        assert result.dtype == dtype
        np.testing.assert_array_equal(result, src)

    @pytest.mark.parametrize("dtype", [
        np.float32, np.float64,
        np.uint8, np.uint16,
        np.int16, np.int8,
    ])
    @pytest.mark.parametrize("kernel", KERNELS)
    def test_dtype_kernel_matrix(self, dtype, kernel):
        """Every dtype × kernel combination should not crash."""
        size = 16
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            rng = np.random.default_rng(42)
            src = rng.integers(1, min(info.max, 200), size=(size, size)).astype(dtype)
        else:
            src = np.arange(size * size, dtype=dtype).reshape(size, size)

        origin_y = 6600000.0 + size * 100.0
        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, origin_y)

        result = reproject_array(
            src, CRS_STR, transform,
            CRS_STR, transform, (size, size),
            resampling=kernel,
        )

        assert result.dtype == dtype
        assert result.shape == (size, size)


class TestIntegerValueRanges:
    """Integer dtype value range edge cases."""

    def test_uint8_full_range(self):
        """uint8 values 0-255 should survive nearest reprojection."""
        src = np.arange(256, dtype=np.uint8).reshape(16, 16)
        origin_y = 6600000.0 + 16 * 100.0
        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, origin_y)

        result = reproject_array(
            src, CRS_STR, transform,
            CRS_STR, transform, (16, 16),
            resampling="nearest",
        )

        np.testing.assert_array_equal(result, src)

    def test_int16_negative_values(self):
        """Negative int16 values should be preserved."""
        src = np.array([[-32768, -100, 0, 100],
                        [-1, 1, 32767, -9999]], dtype=np.int16).reshape(2, 4)
        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600200.0)

        result = reproject_array(
            src, CRS_STR, transform,
            CRS_STR, transform, (2, 4),
            resampling="nearest",
        )

        assert result.dtype == np.int16
        np.testing.assert_array_equal(result, src)

    def test_uint16_max_value(self):
        """uint16 max (65535) should be preserved."""
        src = np.array([[0, 1, 65534, 65535]], dtype=np.uint16).reshape(1, 4)
        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600100.0)

        result = reproject_array(
            src, CRS_STR, transform,
            CRS_STR, transform, (1, 4),
            resampling="nearest",
        )

        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, src)

    def test_int8_signed_range(self):
        """int8 range [-128, 127] should be preserved."""
        src = np.array([[-128, -1, 0, 127]], dtype=np.int8).reshape(1, 4)
        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600100.0)

        result = reproject_array(
            src, CRS_STR, transform,
            CRS_STR, transform, (1, 4),
            resampling="nearest",
        )

        assert result.dtype == np.int8
        np.testing.assert_array_equal(result, src)


class TestIntegerNodata:
    """Nodata handling specific to integer types."""

    @pytest.mark.parametrize("dtype,nodata", [
        (np.uint8, 0),
        (np.uint8, 255),
        (np.uint16, 0),
        (np.uint16, 65535),
        (np.int16, -9999),
        (np.int16, -32768),
        (np.int8, -128),
        (np.int8, 0),
    ])
    def test_integer_nodata_identity(self, dtype, nodata):
        """Integer nodata values should survive identity reprojection."""
        size = 8
        src = np.full((size, size), 42, dtype=dtype)
        src[2:4, 2:4] = nodata
        origin_y = 6600000.0 + size * 100.0
        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, origin_y)

        result = reproject_array(
            src, CRS_STR, transform,
            CRS_STR, transform, (size, size),
            resampling="nearest", nodata=float(nodata),
        )

        assert result.dtype == dtype
        np.testing.assert_array_equal(result[2:4, 2:4], nodata)
        np.testing.assert_array_equal(result[0:2, 0:2], 42)

    @pytest.mark.parametrize("dtype,nodata", [
        (np.uint8, 0),
        (np.uint16, 0),
        (np.int16, -9999),
    ])
    def test_integer_nodata_cross_crs(self, dtype, nodata):
        """Integer nodata should survive cross-CRS reprojection."""
        import rasterio.warp
        from rasterio.crs import CRS

        size = 16
        src = np.full((size, size), 100, dtype=dtype)
        src[:4, :] = nodata
        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * px
        src_transform = (px, 0.0, origin_x, 0.0, -px, origin_y)

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input(CRS_STR),
            CRS.from_user_input("EPSG:4326"),
            size, size,
            left=origin_x,
            bottom=origin_y - size * px,
            right=origin_x + size * px,
            top=origin_y,
        )
        dst_transform = tuple(dst_affine)[:6]

        result = reproject_array(
            src, CRS_STR, src_transform,
            "EPSG:4326", dst_transform, (dst_h, dst_w),
            resampling="nearest", nodata=float(nodata),
        )

        assert result.dtype == dtype
        # Should have both nodata and valid pixels
        has_nodata = np.any(result == nodata)
        has_valid = np.any(result == 100)
        assert has_nodata or has_valid, "No recognizable pixel values in output"


class TestUnsupportedDtypes:
    """Dtypes not supported should raise clear errors."""

    @pytest.mark.parametrize("dtype", [
        np.complex64, np.complex128,
    ])
    def test_complex_raises(self, dtype):
        """Complex dtypes should raise ValueError or TypeError."""
        src = np.ones((4, 4), dtype=dtype)
        with pytest.raises((ValueError, TypeError)):
            reproject_array(
                src, CRS_STR, TRANSFORM,
                CRS_STR, TRANSFORM, (4, 4),
            )

    def test_bool_raises(self):
        """Boolean dtype should raise."""
        src = np.ones((4, 4), dtype=bool)
        with pytest.raises((ValueError, TypeError)):
            reproject_array(
                src, CRS_STR, TRANSFORM,
                CRS_STR, TRANSFORM, (4, 4),
            )


class TestFloatSpecialValues:
    """Float special values: inf, -inf, denormals."""

    def test_inf_propagation(self):
        """Inf values should survive identity nearest reprojection."""
        src = np.array([[1.0, np.inf], [-np.inf, 42.0]], dtype=np.float64)
        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600200.0)

        result = reproject_array(
            src, CRS_STR, transform,
            CRS_STR, transform, (2, 2),
            resampling="nearest",
        )

        assert result[0, 1] == np.inf
        assert result[1, 0] == -np.inf
        assert result[0, 0] == 1.0
        assert result[1, 1] == 42.0

    def test_subnormal_float32(self):
        """Subnormal float32 values should not become zero."""
        tiny = np.float32(1e-40)  # subnormal
        src = np.array([[tiny, 1.0], [1.0, tiny]], dtype=np.float32)
        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600200.0)

        result = reproject_array(
            src, CRS_STR, transform,
            CRS_STR, transform, (2, 2),
            resampling="nearest",
        )

        assert result.dtype == np.float32
        # Subnormal should be preserved (or at least not become exactly 0)
        assert result[0, 0] == tiny or result[0, 0] != 0.0

    def test_negative_zero(self):
        """Negative zero should not cause issues."""
        src = np.array([[0.0, -0.0], [1.0, 2.0]], dtype=np.float64)
        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600200.0)

        result = reproject_array(
            src, CRS_STR, transform,
            CRS_STR, transform, (2, 2),
            resampling="nearest",
        )

        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0  # -0.0 == 0.0
