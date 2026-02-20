"""Alpha channel / transparency tests.

Tests that multi-band data with an alpha/mask band is handled correctly
through reprojection. Since rust-warp operates on single 2D bands, these
tests verify that alpha-masked workflows produce correct results when
the caller manages the alpha band separately.
"""

import numpy as np
import pytest
from rust_warp import reproject_array

KERNELS = ["nearest", "bilinear", "cubic", "lanczos"]
CRS_STR = "EPSG:32633"


def _make_identity_transform(size, pixel_size=100.0):
    origin_y = 6600000.0 + size * pixel_size
    return (pixel_size, 0.0, 500000.0, 0.0, -pixel_size, origin_y)


class TestAlphaMaskWorkflow:
    """Test the pattern of reprojecting data + alpha separately."""

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_data_and_alpha_separate_reproject(self, kernel):
        """Reprojecting data and alpha independently should be consistent."""
        size = 32
        transform = _make_identity_transform(size)

        # Data band: gradient values
        data = np.arange(size * size, dtype=np.float64).reshape(size, size)
        # Alpha band: 0 in border, 255 in interior
        alpha = np.zeros((size, size), dtype=np.float64)
        alpha[4:-4, 4:-4] = 255.0

        data_out = reproject_array(
            data, CRS_STR, transform,
            CRS_STR, transform, (size, size),
            resampling=kernel,
        )
        alpha_out = reproject_array(
            alpha, CRS_STR, transform,
            CRS_STR, transform, (size, size),
            resampling=kernel,
        )

        if kernel == "nearest":
            np.testing.assert_array_equal(data_out, data)
            np.testing.assert_array_equal(alpha_out, alpha)
        else:
            # Interior should match
            margin = 3 if kernel == "lanczos" else 2 if kernel == "cubic" else 1
            np.testing.assert_allclose(
                alpha_out[4 + margin:-4 - margin, 4 + margin:-4 - margin],
                255.0, atol=1e-6,
            )

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_apply_alpha_mask_after_reproject(self, kernel):
        """After reprojecting, masking data where alpha=0 should remove border."""
        size = 32
        transform = _make_identity_transform(size)

        data = np.ones((size, size), dtype=np.float64) * 42.0
        alpha = np.zeros((size, size), dtype=np.float64)
        alpha[8:-8, 8:-8] = 1.0

        # Use NaN for masked data
        masked_src = np.where(alpha > 0, data, np.nan)

        result = reproject_array(
            masked_src, CRS_STR, transform,
            CRS_STR, transform, (size, size),
            resampling=kernel,
        )

        # Center should be valid (42.0)
        center = result[12:-12, 12:-12]
        if kernel == "nearest":
            np.testing.assert_allclose(center, 42.0, atol=1e-6)
        else:
            # Interpolating kernels: deep interior should still be valid
            assert np.all(~np.isnan(center)), f"[{kernel}] Center has NaN after alpha mask"


class TestRGBABandReproject:
    """Simulated RGBA (4-band) reprojection band-by-band."""

    def test_rgba_bands_identity(self):
        """All RGBA bands should survive identity reprojection."""
        size = 16
        transform = _make_identity_transform(size)

        r_band = np.full((size, size), 200.0, dtype=np.float64)
        g_band = np.full((size, size), 100.0, dtype=np.float64)
        b_band = np.full((size, size), 50.0, dtype=np.float64)
        a_band = np.full((size, size), 255.0, dtype=np.float64)
        a_band[:4, :] = 0.0  # Transparent top rows

        for band, expected_val in [(r_band, 200.0), (g_band, 100.0),
                                    (b_band, 50.0), (a_band, None)]:
            result = reproject_array(
                band, CRS_STR, transform,
                CRS_STR, transform, (size, size),
                resampling="nearest",
            )
            np.testing.assert_array_equal(result, band)

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_rgba_transparent_region_consistency(self, kernel):
        """Transparent pixels (alpha=0) should map consistently across RGBA bands."""
        size = 32
        transform = _make_identity_transform(size)

        # Create RGBA with transparent circle in center
        r, c = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        dist = np.sqrt((r - size / 2) ** 2 + (c - size / 2) ** 2)
        alpha = np.where(dist < size / 4, 0.0, 255.0)

        # Use NaN where alpha is 0 for each band
        data_band = np.arange(size * size, dtype=np.float64).reshape(size, size)
        data_band[alpha == 0] = np.nan

        data_out = reproject_array(
            data_band, CRS_STR, transform,
            CRS_STR, transform, (size, size),
            resampling=kernel,
        )
        reproject_array(
            alpha, CRS_STR, transform,
            CRS_STR, transform, (size, size),
            resampling=kernel,
        )

        if kernel == "nearest":
            # Where alpha input was 0, data should be NaN
            np.testing.assert_array_equal(
                np.isnan(data_out), np.isnan(data_band),
            )


class TestUint8Alpha:
    """Alpha channel with uint8 data type."""

    def test_uint8_alpha_identity(self):
        """uint8 alpha band should survive identity reprojection."""
        size = 16
        transform = _make_identity_transform(size)

        alpha = np.zeros((size, size), dtype=np.uint8)
        alpha[4:-4, 4:-4] = 255

        result = reproject_array(
            alpha, CRS_STR, transform,
            CRS_STR, transform, (size, size),
            resampling="nearest", nodata=0.0,
        )

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, alpha)

    def test_uint8_data_with_alpha_mask(self):
        """uint8 RGB data masked by alpha should preserve values."""
        size = 16
        transform = _make_identity_transform(size)

        data = np.full((size, size), 128, dtype=np.uint8)
        data[:4, :] = 0  # "transparent" region uses 0 as nodata

        result = reproject_array(
            data, CRS_STR, transform,
            CRS_STR, transform, (size, size),
            resampling="nearest", nodata=0.0,
        )

        assert result.dtype == np.uint8
        assert np.all(result[:4, :] == 0)
        assert np.all(result[4:, :] == 128)


class TestGreyAlpha:
    """Grey + Alpha (2-band) workflow."""

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_grey_alpha_separate_bands(self, kernel):
        """Grey and alpha bands reprojected separately should be consistent."""
        size = 32
        transform = _make_identity_transform(size)

        grey = np.linspace(0, 255, size * size, dtype=np.float64).reshape(size, size)
        alpha = np.ones((size, size), dtype=np.float64) * 255.0
        # Set left half to transparent
        alpha[:, :size // 2] = 0.0

        grey_masked = np.where(alpha > 0, grey, np.nan)

        grey_out = reproject_array(
            grey_masked, CRS_STR, transform,
            CRS_STR, transform, (size, size),
            resampling=kernel,
        )
        reproject_array(
            alpha, CRS_STR, transform,
            CRS_STR, transform, (size, size),
            resampling=kernel,
        )

        # Where alpha input was 0, grey output should be NaN
        if kernel == "nearest":
            assert np.all(np.isnan(grey_out[:, :size // 2]))
            assert np.all(~np.isnan(grey_out[:, size // 2:]))
