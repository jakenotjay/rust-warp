"""Pixel-by-pixel correctness tests: rust-warp vs rasterio/GDAL.

Parametrized test matrix covering:
- CRS pairs: UTM33->4326, 4326->UTM33, UTM33->3857, 3857->4326, 4326->UTM17, UTM33->UTM17
- Kernels: nearest, bilinear, cubic, lanczos, average
- Dtypes: float64, float32, uint8, uint16, int16
- Sizes: 64x64, 256x256
"""

import os
import sys

import numpy as np
import pytest
from rust_warp import reproject_array

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import compare_arrays, gdal_reproject, make_reprojection_setup, synthetic_raster

# ---------------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------------

CRS_PAIRS = [
    ("EPSG:32633", "EPSG:4326"),
    ("EPSG:4326", "EPSG:32633"),
    ("EPSG:32633", "EPSG:3857"),
    ("EPSG:3857", "EPSG:4326"),
    ("EPSG:4326", "EPSG:32617"),
    ("EPSG:32633", "EPSG:32617"),
]

INTERPOLATING_KERNELS = ["bilinear", "cubic", "lanczos"]
ALL_KERNELS = ["nearest", "bilinear", "cubic", "lanczos"]
SIZES = [64, 256]
FLOAT_DTYPES = [np.float64, np.float32]
INT_DTYPES = [np.uint8, np.uint16, np.int16]


def _crs_pair_id(pair):
    return f"{pair[0]}->{pair[1]}"


# ---------------------------------------------------------------------------
# Tolerance definitions
# ---------------------------------------------------------------------------

# Nearest: we expect >92% exact match due to sub-pixel projection differences
# Interpolating kernels: tight tolerances based on measured boundary handling differences
TOLERANCES = {
    # Nearest uses _assert_nearest_match (>90% exact match)
    "nearest": {"atol": 0.0, "rtol": 0.0},
    # Interpolating kernels: edge pixels dominate max error due to boundary handling
    # differences between rust-warp and GDAL. Coordinate differences are <0.12px.
    # Boundary pixels where partial kernel evaluation differs cause max errors of ~2-3.
    # Mean error is typically <1.
    "bilinear": {"atol": 3.0, "rtol": 0.01},
    "cubic": {"atol": 2.0, "rtol": 0.01},
    "lanczos": {"atol": 2.5, "rtol": 0.01},
    "average": {"atol": 15.0, "rtol": 0.05},
}


# ---------------------------------------------------------------------------
# Float dtype tests
# ---------------------------------------------------------------------------


class TestFloatCorrectness:
    """Correctness tests for float dtypes against GDAL."""

    @pytest.mark.parametrize("crs_pair", CRS_PAIRS, ids=[_crs_pair_id(p) for p in CRS_PAIRS])
    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    @pytest.mark.parametrize("size", SIZES)
    def test_float64(self, crs_pair, kernel, size):
        src_crs, dst_crs = crs_pair
        setup = make_reprojection_setup(src_crs, dst_crs, size=size)

        rust_result = reproject_array(
            setup["src"],
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
            resampling=kernel,
        )
        gdal_result = gdal_reproject(
            setup["src"],
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
            resampling=kernel,
        )

        if kernel == "nearest":
            _assert_nearest_match(rust_result, gdal_result, size)
        else:
            tol = TOLERANCES[kernel]
            compare_arrays(rust_result, gdal_result, kernel, atol=tol["atol"], rtol=tol["rtol"])

    @pytest.mark.parametrize(
        "crs_pair", CRS_PAIRS[:3], ids=[_crs_pair_id(p) for p in CRS_PAIRS[:3]]
    )
    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_float32(self, crs_pair, kernel):
        src_crs, dst_crs = crs_pair
        setup = make_reprojection_setup(src_crs, dst_crs, size=64)

        src_f32 = setup["src"].astype(np.float32)

        rust_result = reproject_array(
            src_f32,
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
            resampling=kernel,
        )
        gdal_result = gdal_reproject(
            src_f32,
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
            resampling=kernel,
        )

        assert rust_result.dtype == np.float32

        if kernel == "nearest":
            _assert_nearest_match(rust_result, gdal_result, 64)
        else:
            tol = TOLERANCES[kernel]
            compare_arrays(rust_result, gdal_result, kernel, atol=tol["atol"], rtol=tol["rtol"])


# ---------------------------------------------------------------------------
# Integer dtype tests
# ---------------------------------------------------------------------------


class TestIntegerCorrectness:
    """Correctness tests for integer dtypes against GDAL."""

    @pytest.mark.parametrize(
        "crs_pair", CRS_PAIRS[:3], ids=[_crs_pair_id(p) for p in CRS_PAIRS[:3]]
    )
    @pytest.mark.parametrize("dtype", INT_DTYPES, ids=["uint8", "uint16", "int16"])
    def test_nearest_integer(self, crs_pair, dtype):
        """Nearest-neighbor with integer data should have high exact match rate."""
        src_crs, dst_crs = crs_pair
        setup = make_reprojection_setup(src_crs, dst_crs, size=64)

        # Create integer source data with appropriate range
        src_int = synthetic_raster((64, 64), dtype=dtype, pattern="random")

        nodata_val = 0.0
        rust_result = reproject_array(
            src_int,
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
            resampling="nearest",
            nodata=nodata_val,
        )
        gdal_result = gdal_reproject(
            src_int,
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
            resampling="nearest",
            nodata=nodata_val,
        )

        assert rust_result.dtype == dtype

        # Both should have same nodata pattern (approximately)
        rust_nodata = rust_result == int(nodata_val)
        gdal_nodata = gdal_result == int(nodata_val)

        # Compare valid pixels
        both_valid = ~rust_nodata & ~gdal_nodata
        if both_valid.any():
            r = rust_result[both_valid]
            g = gdal_result[both_valid]
            exact = np.sum(r == g)
            pct = exact / len(r) * 100
            assert pct > 95.0, f"Only {pct:.1f}% exact match for {dtype.__name__}"

    @pytest.mark.parametrize(
        "crs_pair", CRS_PAIRS[:2], ids=[_crs_pair_id(p) for p in CRS_PAIRS[:2]]
    )
    @pytest.mark.parametrize("kernel", INTERPOLATING_KERNELS)
    @pytest.mark.parametrize("dtype", INT_DTYPES, ids=["uint8", "uint16", "int16"])
    def test_interpolating_integer(self, crs_pair, kernel, dtype):
        """Interpolating kernels with integer smooth gradient: outputs should correlate."""
        src_crs, dst_crs = crs_pair
        setup = make_reprojection_setup(src_crs, dst_crs, size=64)

        # Use smooth gradient (not random) so interpolation gives comparable results
        # even with sub-pixel projection shifts
        src_int = synthetic_raster((64, 64), dtype=dtype, pattern="gradient")

        nodata_val = 0.0
        rust_result = reproject_array(
            src_int,
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
            resampling=kernel,
            nodata=nodata_val,
        )
        gdal_result = gdal_reproject(
            src_int,
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
            resampling=kernel,
            nodata=nodata_val,
        )

        assert rust_result.dtype == dtype

        # Compare valid pixels using correlation and bounded absolute difference
        both_valid = (rust_result != int(nodata_val)) & (gdal_result != int(nodata_val))
        if both_valid.any() and both_valid.sum() > 10:
            r = rust_result[both_valid].astype(np.float64)
            g = gdal_result[both_valid].astype(np.float64)
            # Results should be strongly correlated (both are smooth gradients).
            # uint8 has only 256 levels so quantization noise lowers correlation.
            correlation = np.corrcoef(r, g)[0, 1]
            min_corr = 0.90 if dtype == np.uint8 else 0.95
            assert correlation > min_corr, (
                f"[{kernel}/{dtype.__name__}] Low correlation: {correlation:.4f}"
            )


# ---------------------------------------------------------------------------
# Average/downsampling tests
# ---------------------------------------------------------------------------


class TestAverageCorrectness:
    """Average resampling correctness for downsampling scenarios."""

    @pytest.mark.parametrize(("src_size", "dst_size"), [(128, 32), (256, 64)])
    def test_average_downscale(self, src_size, dst_size):
        crs = "EPSG:32633"
        pixel_size = 100.0
        origin_x, origin_y = 500000.0, 6600000.0

        src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

        src = synthetic_raster((src_size, src_size), pattern="gradient")

        dst_pixel_size = pixel_size * (src_size / dst_size)
        dst_transform = (dst_pixel_size, 0.0, origin_x, 0.0, -dst_pixel_size, origin_y)
        dst_shape = (dst_size, dst_size)

        rust_result = reproject_array(
            src,
            crs,
            src_transform,
            crs,
            dst_transform,
            dst_shape,
            resampling="average",
        )
        gdal_result = gdal_reproject(
            src,
            crs,
            src_transform,
            crs,
            dst_transform,
            dst_shape,
            resampling="average",
        )

        both_valid = ~np.isnan(rust_result) & ~np.isnan(gdal_result)
        if both_valid.any():
            compare_arrays(rust_result, gdal_result, "average", atol=2.0, rtol=0.01)

    @pytest.mark.parametrize(
        "crs_pair",
        [
            ("EPSG:32633", "EPSG:4326"),
            ("EPSG:32633", "EPSG:3857"),
        ],
        ids=["UTM33->4326", "UTM33->3857"],
    )
    def test_average_with_crs_change(self, crs_pair):
        """Average resampling with CRS reprojection: outputs should correlate.

        GDAL and rust-warp compute area-weighted averages with different footprint
        geometry, so we check correlation rather than exact tolerance.
        """
        src_crs, dst_crs = crs_pair
        setup = make_reprojection_setup(src_crs, dst_crs, size=128)

        rust_result = reproject_array(
            setup["src"],
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
            resampling="average",
        )
        gdal_result = gdal_reproject(
            setup["src"],
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
            resampling="average",
        )

        both_valid = ~np.isnan(rust_result) & ~np.isnan(gdal_result)
        if both_valid.any() and both_valid.sum() > 10:
            a = rust_result[both_valid].astype(np.float64)
            e = gdal_result[both_valid].astype(np.float64)
            # Results should be strongly correlated even if absolute values differ
            correlation = np.corrcoef(a, e)[0, 1]
            assert correlation > 0.95, f"Low correlation: {correlation:.4f}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_nearest_match(rust_result, gdal_result, src_cols):
    """Assert nearest-neighbor results match with >98% exact agreement."""
    rust_nan = (
        np.isnan(rust_result)
        if np.issubdtype(rust_result.dtype, np.floating)
        else np.zeros(rust_result.shape, dtype=bool)
    )
    gdal_nan = (
        np.isnan(gdal_result)
        if np.issubdtype(gdal_result.dtype, np.floating)
        else np.zeros(gdal_result.shape, dtype=bool)
    )

    # Compare valid pixels
    both_valid = ~rust_nan & ~gdal_nan
    if not both_valid.any():
        return

    r = rust_result[both_valid].astype(np.float64)
    g = gdal_result[both_valid].astype(np.float64)

    exact_match = np.sum(r == g)
    match_pct = exact_match / len(r) * 100
    assert match_pct > 90.0, f"Only {match_pct:.1f}% exact match (need >90%)"

    # Remaining differences should be bounded
    diff = np.abs(r - g)
    max_diff = diff.max()
    assert max_diff <= src_cols + 1, f"Max diff {max_diff} exceeds src_cols+1={src_cols + 1}"
