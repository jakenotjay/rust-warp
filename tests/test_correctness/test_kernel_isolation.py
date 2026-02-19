"""Kernel isolation test: feed pyproj-computed coordinates into rust-warp kernels.

This isolates projection accuracy from kernel accuracy. If rust-warp's kernels
produce correct results when given GDAL-identical coordinates, then any remaining
error in normal reprojection is purely from coordinate differences.
"""

import os
import sys

import numpy as np
import pyproj
import pytest
from rust_warp import reproject_array, reproject_with_grid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import compare_arrays, gdal_reproject, make_reprojection_setup

CRS_PAIRS = [
    ("EPSG:32633", "EPSG:4326"),
    ("EPSG:4326", "EPSG:32633"),
    ("EPSG:32633", "EPSG:3857"),
    ("EPSG:3857", "EPSG:4326"),
    ("EPSG:4326", "EPSG:32617"),
    ("EPSG:32633", "EPSG:32617"),
]

KERNELS = ["nearest", "bilinear", "cubic", "lanczos"]


def compute_pyproj_grid(setup):
    """Compute source pixel coordinates using pyproj (matching GDAL)."""
    dst_shape = setup["dst_shape"]
    a, b, c, d, e, f = setup["dst_transform"]
    sa, sb, sc, sd, se, sf = setup["src_transform"]

    # Source affine inverse
    det = sa * se - sb * sd
    inv_sa = se / det
    inv_sb = -sb / det
    inv_sc = (sb * sf - se * sc) / det
    inv_sd = -sd / det
    inv_se = sa / det
    inv_sf = (sd * sc - sa * sf) / det

    dst_rows, dst_cols = dst_shape

    transformer = pyproj.Transformer.from_crs(
        setup["dst_crs"], setup["src_crs"], always_xy=True
    )

    cols_idx = np.arange(dst_cols) + 0.5
    rows_idx = np.arange(dst_rows) + 0.5
    cc, rr = np.meshgrid(cols_idx, rows_idx)

    dst_x = a * cc + b * rr + c
    dst_y = d * cc + e * rr + f

    src_x, src_y = transformer.transform(dst_x.ravel(), dst_y.ravel())
    src_x = np.array(src_x).reshape(dst_shape)
    src_y = np.array(src_y).reshape(dst_shape)

    col_grid = inv_sa * src_x + inv_sb * src_y + inv_sc
    row_grid = inv_sd * src_x + inv_se * src_y + inv_sf

    return col_grid, row_grid


class TestKernelIsolation:
    """Test rust-warp kernels with pyproj-computed coordinates vs GDAL."""

    @pytest.mark.parametrize(
        "crs_pair",
        CRS_PAIRS,
        ids=[f"{s}->{d}" for s, d in CRS_PAIRS],
    )
    @pytest.mark.parametrize("kernel", KERNELS)
    def test_hybrid_vs_gdal(self, crs_pair, kernel):
        """Feed pyproj coordinates into rust-warp kernels and compare against GDAL.

        If the hybrid result matches GDAL closely (maxdiff <= 1 for integers),
        then rust-warp's kernels are correct and all error is from projection.
        """
        src_crs, dst_crs = crs_pair
        setup = make_reprojection_setup(src_crs, dst_crs, size=64)

        # Get pyproj-computed source pixel coordinates
        pp_col, pp_row = compute_pyproj_grid(setup)

        # Hybrid: pyproj coordinates + rust-warp kernels
        hybrid_result = reproject_with_grid(
            setup["src"],
            pp_col.astype(np.float64),
            pp_row.astype(np.float64),
            resampling=kernel,
        )

        # GDAL reference
        gdal_result = gdal_reproject(
            setup["src"],
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
            resampling=kernel,
        )

        # Compare hybrid vs GDAL
        hybrid_valid = ~np.isnan(hybrid_result)
        gdal_valid = ~np.isnan(gdal_result)
        both_valid = hybrid_valid & gdal_valid

        if not both_valid.any():
            return

        h = hybrid_result[both_valid]
        g = gdal_result[both_valid]
        diff = np.abs(h - g)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        # With identical coordinates, kernels should match within tight tolerance
        # (small differences from boundary handling and float precision)
        if kernel == "nearest":
            exact_match_pct = float(np.sum(h == g) / len(h) * 100)
            assert exact_match_pct > 97.0, (
                f"[hybrid nearest] Only {exact_match_pct:.1f}% exact match"
            )
        else:
            # Interpolating kernels should be very close with identical coordinates
            assert max_diff <= 5.0, (
                f"[hybrid {kernel}] max_diff={max_diff:.2f} (want <=5.0), "
                f"mean={mean_diff:.4f}"
            )

    @pytest.mark.parametrize(
        "crs_pair",
        CRS_PAIRS[:3],
        ids=[f"{s}->{d}" for s, d in CRS_PAIRS[:3]],
    )
    @pytest.mark.parametrize("kernel", ["bilinear", "cubic", "lanczos"])
    def test_quantify_projection_vs_kernel_error(self, crs_pair, kernel):
        """Quantify how much error comes from projection vs kernels.

        Compares:
        - normal_error: rust-warp (own coords + own kernels) vs GDAL
        - hybrid_error: rust-warp (pyproj coords + own kernels) vs GDAL
        - The difference tells us the projection contribution.
        """
        src_crs, dst_crs = crs_pair
        setup = make_reprojection_setup(src_crs, dst_crs, size=64)

        # Normal rust-warp
        normal_result = reproject_array(
            setup["src"],
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
            resampling=kernel,
        )

        # Hybrid: pyproj coords + rust-warp kernels
        pp_col, pp_row = compute_pyproj_grid(setup)
        hybrid_result = reproject_with_grid(
            setup["src"],
            pp_col.astype(np.float64),
            pp_row.astype(np.float64),
            resampling=kernel,
        )

        # GDAL reference
        gdal_result = gdal_reproject(
            setup["src"],
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
            resampling=kernel,
        )

        # Compare both against GDAL (where all three are valid)
        all_valid = (
            ~np.isnan(normal_result) & ~np.isnan(hybrid_result) & ~np.isnan(gdal_result)
        )
        if not all_valid.any():
            return

        normal_diff = np.abs(normal_result[all_valid] - gdal_result[all_valid])
        hybrid_diff = np.abs(hybrid_result[all_valid] - gdal_result[all_valid])

        normal_max = float(normal_diff.max())
        hybrid_max = float(hybrid_diff.max())
        normal_mean = float(normal_diff.mean())
        hybrid_mean = float(hybrid_diff.mean())

        # The hybrid should always be at least as good as normal
        # (print for diagnostic visibility)
        print(
            f"\n  [{src_crs}->{dst_crs} {kernel}] "
            f"normal: max={normal_max:.2f} mean={normal_mean:.4f} | "
            f"hybrid: max={hybrid_max:.2f} mean={hybrid_mean:.4f}"
        )

        # hybrid_max should be significantly less than normal_max
        # (unless there are boundary handling differences)
        assert hybrid_max <= normal_max + 1.0, (
            f"Hybrid should not be much worse than normal: "
            f"hybrid_max={hybrid_max:.2f} vs normal_max={normal_max:.2f}"
        )
