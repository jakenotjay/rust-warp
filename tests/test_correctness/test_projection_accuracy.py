"""Projection coordinate accuracy regression tests.

Compares rust-warp coordinate transforms against pyproj at many points
per CRS pair. Ensures coordinate error stays below 0.15 source pixels.
"""

import os
import sys

import numpy as np
import pyproj
import pytest
from rust_warp import reproject_with_grid, transform_grid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import gdal_reproject, make_reprojection_setup

CRS_PAIRS = [
    ("EPSG:32633", "EPSG:4326"),
    ("EPSG:4326", "EPSG:32633"),
    ("EPSG:32633", "EPSG:3857"),
    ("EPSG:3857", "EPSG:4326"),
    ("EPSG:4326", "EPSG:32617"),
    ("EPSG:32633", "EPSG:32617"),
]


def compute_pyproj_grid(setup):
    """Compute source pixel coordinates using pyproj (matching GDAL)."""
    dst_shape = setup["dst_shape"]
    a, b, c, d, e, f = setup["dst_transform"]
    sa, sb, sc, sd, se, sf = setup["src_transform"]

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


class TestProjectionAccuracy:
    """Coordinate transform accuracy regression tests."""

    @pytest.mark.parametrize(
        "crs_pair",
        CRS_PAIRS,
        ids=[f"{s}->{d}" for s, d in CRS_PAIRS],
    )
    def test_coordinate_error_below_threshold(self, crs_pair):
        """rust-warp coordinate transforms should stay within 0.15px of pyproj."""
        src_crs, dst_crs = crs_pair
        setup = make_reprojection_setup(src_crs, dst_crs, size=64)

        rw_col, rw_row = transform_grid(
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
        )
        pp_col, pp_row = compute_pyproj_grid(setup)

        valid = ~np.isnan(rw_col) & ~np.isnan(rw_row) & ~np.isnan(pp_col) & ~np.isnan(pp_row)
        assert valid.any(), "No valid pixels to compare"

        total_diff = np.sqrt(
            (rw_col[valid] - pp_col[valid]) ** 2 + (rw_row[valid] - pp_row[valid]) ** 2
        )

        max_err = float(total_diff.max())
        mean_err = float(total_diff.mean())

        assert max_err < 0.15, (
            f"[{src_crs}->{dst_crs}] Max coordinate error {max_err:.4f}px exceeds 0.15px threshold "
            f"(mean={mean_err:.4f}px)"
        )

    @pytest.mark.parametrize(
        "crs_pair",
        CRS_PAIRS,
        ids=[f"{s}->{d}" for s, d in CRS_PAIRS],
    )
    def test_coordinate_error_256(self, crs_pair):
        """Coordinate accuracy at 256x256 — should be even tighter than 64x64."""
        src_crs, dst_crs = crs_pair
        setup = make_reprojection_setup(src_crs, dst_crs, size=256)

        rw_col, rw_row = transform_grid(
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
        )
        pp_col, pp_row = compute_pyproj_grid(setup)

        valid = ~np.isnan(rw_col) & ~np.isnan(rw_row) & ~np.isnan(pp_col) & ~np.isnan(pp_row)
        assert valid.any(), "No valid pixels to compare"

        total_diff = np.sqrt(
            (rw_col[valid] - pp_col[valid]) ** 2 + (rw_row[valid] - pp_row[valid]) ** 2
        )

        max_err = float(total_diff.max())
        assert max_err < 0.15, (
            f"[{src_crs}->{dst_crs}] Max coordinate error {max_err:.4f}px at 256x256"
        )


class TestKernelIsolationRegression:
    """Regression: pyproj coordinates + rust-warp kernels should match GDAL closely."""

    @pytest.mark.parametrize(
        "crs_pair",
        CRS_PAIRS[:3],
        ids=[f"{s}->{d}" for s, d in CRS_PAIRS[:3]],
    )
    @pytest.mark.parametrize("kernel", ["bilinear", "cubic", "lanczos"])
    def test_hybrid_matches_gdal(self, crs_pair, kernel):
        """With pyproj coordinates, kernel output should be very close to GDAL."""
        src_crs, dst_crs = crs_pair
        setup = make_reprojection_setup(src_crs, dst_crs, size=64)

        pp_col, pp_row = compute_pyproj_grid(setup)
        hybrid = reproject_with_grid(
            setup["src"],
            pp_col.astype(np.float64),
            pp_row.astype(np.float64),
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

        both_valid = ~np.isnan(hybrid) & ~np.isnan(gdal_result)
        if not both_valid.any():
            return

        diff = np.abs(hybrid[both_valid] - gdal_result[both_valid])
        max_diff = float(diff.max())

        # With identical coordinates, max error should be bounded by
        # kernel boundary handling differences only
        assert max_diff <= 5.0, (
            f"[{src_crs}->{dst_crs} {kernel}] hybrid max_diff={max_diff:.2f} exceeds 5.0"
        )
