"""Non-square / anisotropic pixel tests.

Tests that reprojection works correctly when source and/or destination pixels
have different X and Y resolutions.
"""

import numpy as np
import pytest
import rasterio.warp
from rasterio.crs import CRS
from rust_warp import reproject_array

KERNELS = ["nearest", "bilinear", "cubic", "lanczos"]
UTM_CRS = "EPSG:32633"


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


class TestNonSquareIdentity:
    """Same-CRS reprojection with non-square pixels."""

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_2x1_aspect_identity(self, kernel):
        """Pixels twice as wide as tall should reproject to themselves."""
        rows, cols = 32, 16
        res_x, res_y = 200.0, 100.0  # 2:1 aspect ratio
        origin_x, origin_y = 500000.0, 6600000.0 + rows * res_y
        src_transform = (res_x, 0.0, origin_x, 0.0, -res_y, origin_y)
        src = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)

        result = reproject_array(
            src, UTM_CRS, src_transform,
            UTM_CRS, src_transform, (rows, cols),
            resampling=kernel,
        )

        if kernel == "nearest":
            np.testing.assert_array_equal(result, src)
        else:
            # Interior should match for interpolating kernels
            margin = 3 if kernel == "lanczos" else 2 if kernel == "cubic" else 1
            interior = result[margin:-margin, margin:-margin]
            src_interior = src[margin:-margin, margin:-margin]
            np.testing.assert_allclose(interior, src_interior, atol=1e-6)

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_1x3_aspect_identity(self, kernel):
        """Pixels 3x taller than wide should reproject to themselves."""
        rows, cols = 16, 48
        res_x, res_y = 50.0, 150.0  # 1:3 aspect ratio
        origin_x, origin_y = 500000.0, 6600000.0 + rows * res_y
        src_transform = (res_x, 0.0, origin_x, 0.0, -res_y, origin_y)
        src = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)

        result = reproject_array(
            src, UTM_CRS, src_transform,
            UTM_CRS, src_transform, (rows, cols),
            resampling=kernel,
        )

        if kernel == "nearest":
            np.testing.assert_array_equal(result, src)
        else:
            margin = 3 if kernel == "lanczos" else 2 if kernel == "cubic" else 1
            interior = result[margin:-margin, margin:-margin]
            src_interior = src[margin:-margin, margin:-margin]
            np.testing.assert_allclose(interior, src_interior, atol=1e-6)


class TestNonSquareCrossProjection:
    """Cross-CRS reprojection with non-square pixels."""

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_nonsquare_utm_to_4326(self, kernel):
        """Non-square UTM pixels reprojected to 4326 should match GDAL."""
        rows, cols = 32, 64
        res_x, res_y = 200.0, 100.0  # wide pixels
        origin_x, origin_y = 500000.0, 6600000.0
        src_transform = (res_x, 0.0, origin_x, 0.0, -res_y, origin_y)

        r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        src = (r * cols + c).astype(np.float64)

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input(UTM_CRS),
            CRS.from_user_input("EPSG:4326"),
            cols, rows,
            left=origin_x,
            bottom=origin_y - rows * res_y,
            right=origin_x + cols * res_x,
            top=origin_y,
        )
        dst_transform = tuple(dst_affine)[:6]
        dst_shape = (dst_h, dst_w)

        rust = reproject_array(
            src, UTM_CRS, src_transform,
            "EPSG:4326", dst_transform, dst_shape,
            resampling=kernel,
        )
        gdal = _gdal_reproject(
            src, UTM_CRS, src_transform,
            "EPSG:4326", dst_transform, dst_shape,
            resampling=kernel,
        )

        both_valid = ~np.isnan(rust) & ~np.isnan(gdal)
        assert both_valid.sum() > 0, f"[{kernel}] No overlapping valid pixels"

        if kernel == "nearest":
            diff = np.abs(rust[both_valid] - gdal[both_valid])
            match_pct = np.sum(diff == 0) / both_valid.sum() * 100
            assert match_pct > 90, f"Only {match_pct:.0f}% exact match"
        else:
            # Non-square pixels amplify interpolation differences between
            # proj4rs and PROJ; use generous tolerance
            np.testing.assert_allclose(
                rust[both_valid], gdal[both_valid],
                atol=30.0, rtol=0.25,
            )


class TestNonSquareScaling:
    """Rescaling with non-square pixels (different src and dst aspect ratios)."""

    def test_square_to_nonsquare_downscale(self):
        """Downscale from square src to non-square dst pixels."""
        src_size = 64
        src_px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + src_size * src_px
        src_transform = (src_px, 0.0, origin_x, 0.0, -src_px, origin_y)
        src = np.arange(src_size * src_size, dtype=np.float64).reshape(src_size, src_size)

        # Destination: 32 rows (2x downscale Y) x 16 cols (4x downscale X)
        dst_rows, dst_cols = 32, 16
        dst_res_x = src_px * (src_size / dst_cols)   # 400m
        dst_res_y = src_px * (src_size / dst_rows)    # 200m
        dst_transform = (dst_res_x, 0.0, origin_x, 0.0, -dst_res_y, origin_y)

        result = reproject_array(
            src, UTM_CRS, src_transform,
            UTM_CRS, dst_transform, (dst_rows, dst_cols),
            resampling="average",
        )

        assert result.shape == (dst_rows, dst_cols)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 50, f"Only {valid_pct:.0f}% valid"

    def test_nonsquare_to_square_upscale(self):
        """Upscale from non-square src to square dst pixels."""
        src_rows, src_cols = 16, 32
        src_res_x, src_res_y = 50.0, 200.0  # wide, short pixels
        origin_x = 500000.0
        origin_y = 6600000.0 + src_rows * src_res_y
        src_transform = (src_res_x, 0.0, origin_x, 0.0, -src_res_y, origin_y)

        r, c = np.meshgrid(np.arange(src_rows), np.arange(src_cols), indexing="ij")
        src = (r * src_cols + c).astype(np.float64)

        # Square 100m pixels, covering same extent
        extent_x = src_cols * src_res_x  # 1600m
        extent_y = src_rows * src_res_y  # 3200m
        dst_px = 100.0
        dst_cols = int(extent_x / dst_px)  # 16
        dst_rows = int(extent_y / dst_px)  # 32
        dst_transform = (dst_px, 0.0, origin_x, 0.0, -dst_px, origin_y)

        result = reproject_array(
            src, UTM_CRS, src_transform,
            UTM_CRS, dst_transform, (dst_rows, dst_cols),
            resampling="bilinear",
        )

        assert result.shape == (dst_rows, dst_cols)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 50, f"Only {valid_pct:.0f}% valid"

    @pytest.mark.parametrize("kernel", KERNELS + ["average"])
    def test_extreme_aspect_ratio(self, kernel):
        """Very extreme aspect ratio (10:1) should not crash."""
        rows, cols = 8, 80
        res_x, res_y = 10.0, 100.0  # 10:1
        origin_x, origin_y = 500000.0, 6600000.0 + rows * res_y
        src_transform = (res_x, 0.0, origin_x, 0.0, -res_y, origin_y)
        src = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)

        result = reproject_array(
            src, UTM_CRS, src_transform,
            UTM_CRS, src_transform, (rows, cols),
            resampling=kernel,
        )

        assert result.shape == (rows, cols)
        if kernel == "nearest":
            np.testing.assert_array_equal(result, src)


class TestNonSquareGeodesic:
    """Non-square pixels in geographic (degree) coordinates."""

    def test_geographic_nonsquare(self):
        """Geographic CRS with non-square degree pixels."""
        rows, cols = 32, 64
        # 0.02° lon x 0.01° lat pixels — non-square in degrees
        res_x, res_y = 0.02, 0.01
        src_transform = (res_x, 0.0, 14.0, 0.0, -res_y, 60.0)
        src = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:4326"),
            CRS.from_user_input("EPSG:32633"),
            cols, rows,
            left=14.0,
            bottom=60.0 - rows * res_y,
            right=14.0 + cols * res_x,
            top=60.0,
        )
        dst_transform = tuple(dst_affine)[:6]

        result = reproject_array(
            src, "EPSG:4326", src_transform,
            "EPSG:32633", dst_transform, (dst_h, dst_w),
            resampling="bilinear",
        )

        assert result.shape == (dst_h, dst_w)
        valid_pct = np.sum(~np.isnan(result)) / result.size * 100
        assert valid_pct > 30, f"Only {valid_pct:.0f}% valid"
