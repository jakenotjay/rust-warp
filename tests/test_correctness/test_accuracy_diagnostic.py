"""Accuracy diagnostic: quantify numerical inaccuracy by source.

Measures and reports error broken down by:
- Projection coordinate error (proj4rs vs pyproj/PROJ)
- Kernel interpolation error (rust-warp kernels vs GDAL kernels)
- Total end-to-end error (full rust-warp vs full GDAL)
- Per CRS pair, per kernel, per scale factor, per raster size

Run with: uv run pytest tests/test_correctness/test_accuracy_diagnostic.py -v -s
"""

import numpy as np
import pyproj
import pytest
import rasterio.warp
from rasterio.crs import CRS
from rust_warp import reproject_array, reproject_with_grid, transform_grid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_pyproj_grid(dst_shape, dst_transform, src_transform, dst_crs, src_crs):
    """Compute source pixel coordinates using pyproj (matching GDAL/PROJ)."""
    a, b, c, d, e, f = dst_transform
    sa, sb, sc, sd, se, sf = src_transform

    det = sa * se - sb * sd
    inv_sa = se / det
    inv_sb = -sb / det
    inv_sc = (sb * sf - se * sc) / det
    inv_sd = -sd / det
    inv_se = sa / det
    inv_sf = (sd * sc - sa * sf) / det

    dst_rows, dst_cols = dst_shape
    transformer = pyproj.Transformer.from_crs(dst_crs, src_crs, always_xy=True)

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


def _make_setup(src_crs, dst_crs, size=64, pixel_size=100.0):
    """Create a reprojection setup dict."""
    origins = {
        "EPSG:32633": (500000.0, 6600000.0),
        "EPSG:32617": (500000.0, 4400000.0),
        "EPSG:3857": (1600000.0, 8300000.0),
        "EPSG:4326": (15.0, 60.0),
        "EPSG:3413": (-500000.0, 500000.0),
        "EPSG:3031": (-500000.0, 500000.0),
    }
    origin_x, origin_y = origins.get(src_crs, (500000.0, 6600000.0))

    if src_crs == "EPSG:4326":
        pixel_size = 0.01

    src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)
    r, c = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    src = ((r * size + c) / (size * size) * 1000.0).astype(np.float64)

    dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
        CRS.from_user_input(src_crs),
        CRS.from_user_input(dst_crs),
        size, size,
        left=origin_x,
        bottom=origin_y - size * pixel_size,
        right=origin_x + size * pixel_size,
        top=origin_y,
    )
    return {
        "src": src,
        "src_crs": src_crs,
        "src_transform": src_transform,
        "dst_crs": dst_crs,
        "dst_transform": tuple(dst_affine)[:6],
        "dst_shape": (dst_h, dst_w),
    }


def _error_stats(a, b):
    """Compute error stats between two arrays on their valid intersection."""
    valid = ~np.isnan(a) & ~np.isnan(b)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "max": 0.0, "mean": 0.0, "p95": 0.0, "p99": 0.0, "rmse": 0.0}
    diff = np.abs(a[valid] - b[valid])
    return {
        "n": n,
        "max": float(diff.max()),
        "mean": float(diff.mean()),
        "p95": float(np.percentile(diff, 95)),
        "p99": float(np.percentile(diff, 99)),
        "rmse": float(np.sqrt((diff ** 2).mean())),
    }


def _coord_error_stats(rw_col, rw_row, pp_col, pp_row):
    """Compute coordinate error statistics in pixel units."""
    valid = (
        np.isfinite(rw_col) & np.isfinite(rw_row)
        & np.isfinite(pp_col) & np.isfinite(pp_row)
    )
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "max": 0.0, "mean": 0.0, "p95": 0.0, "p99": 0.0,
                "max_col": 0.0, "max_row": 0.0, "mean_col": 0.0, "mean_row": 0.0}
    col_diff = np.abs(rw_col[valid] - pp_col[valid])
    row_diff = np.abs(rw_row[valid] - pp_row[valid])
    total = np.sqrt(col_diff ** 2 + row_diff ** 2)
    return {
        "n": n,
        "max": float(total.max()),
        "mean": float(total.mean()),
        "p95": float(np.percentile(total, 95)),
        "p99": float(np.percentile(total, 99)),
        "max_col": float(col_diff.max()),
        "max_row": float(row_diff.max()),
        "mean_col": float(col_diff.mean()),
        "mean_row": float(row_diff.mean()),
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

CRS_PAIRS = [
    ("EPSG:32633", "EPSG:4326"),
    ("EPSG:4326", "EPSG:32633"),
    ("EPSG:32633", "EPSG:3857"),
    ("EPSG:3857", "EPSG:4326"),
    ("EPSG:4326", "EPSG:32617"),
    ("EPSG:32633", "EPSG:32617"),
]

KERNELS = ["nearest", "bilinear", "cubic", "lanczos"]


class TestProjectionErrorByPair:
    """Measure coordinate error (pixels) for each CRS pair."""

    @pytest.mark.parametrize(
        "crs_pair", CRS_PAIRS,
        ids=[f"{s}->{d}" for s, d in CRS_PAIRS],
    )
    @pytest.mark.parametrize("size", [64, 256])
    def test_coordinate_error_report(self, crs_pair, size):
        """Report coordinate error for each CRS pair and size."""
        src_crs, dst_crs = crs_pair
        s = _make_setup(src_crs, dst_crs, size=size)

        rw_col, rw_row = transform_grid(
            s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"],
            s["dst_shape"],
        )
        pp_col, pp_row = _compute_pyproj_grid(
            s["dst_shape"], s["dst_transform"],
            s["src_transform"], s["dst_crs"], s["src_crs"],
        )

        stats = _coord_error_stats(rw_col, rw_row, pp_col, pp_row)

        print(
            f"\n  COORD [{src_crs}->{dst_crs} {size}x{size}]"
            f"  n={stats['n']}  "
            f"max={stats['max']:.4f}px  mean={stats['mean']:.6f}px  "
            f"p95={stats['p95']:.4f}px  p99={stats['p99']:.4f}px  "
            f"(col: max={stats['max_col']:.4f} mean={stats['mean_col']:.6f}  "
            f"row: max={stats['max_row']:.4f} mean={stats['mean_row']:.6f})"
        )

        # Assertion: coordinate error should stay below 0.15px
        assert stats["max"] < 0.15, (
            f"Coordinate error {stats['max']:.4f}px exceeds 0.15px threshold"
        )


class TestEndToEndErrorByKernel:
    """Measure total error (value units) for each CRS pair × kernel."""

    @pytest.mark.parametrize(
        "crs_pair", CRS_PAIRS,
        ids=[f"{s}->{d}" for s, d in CRS_PAIRS],
    )
    @pytest.mark.parametrize("kernel", KERNELS)
    def test_total_error_report(self, crs_pair, kernel):
        """Report end-to-end error: full rust-warp vs full GDAL."""
        src_crs, dst_crs = crs_pair
        s = _make_setup(src_crs, dst_crs, size=64)

        rust = reproject_array(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling=kernel,
        )
        gdal = _gdal_reproject(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling=kernel,
        )

        stats = _error_stats(rust, gdal)

        print(
            f"\n  TOTAL [{src_crs}->{dst_crs} {kernel}]"
            f"  n={stats['n']}  "
            f"max={stats['max']:.4f}  mean={stats['mean']:.6f}  "
            f"p95={stats['p95']:.4f}  p99={stats['p99']:.4f}  "
            f"rmse={stats['rmse']:.6f}"
        )

        # Values are on a 0-1000 scale
        if kernel == "nearest":
            assert stats["max"] <= 64, "Nearest max error exceeds 1-row boundary"
        else:
            assert stats["max"] <= 20, f"[{kernel}] max error {stats['max']:.2f} > 20"


class TestErrorDecomposition:
    """Decompose error into projection vs kernel components."""

    @pytest.mark.parametrize(
        "crs_pair", CRS_PAIRS,
        ids=[f"{s}->{d}" for s, d in CRS_PAIRS],
    )
    @pytest.mark.parametrize("kernel", ["bilinear", "cubic", "lanczos"])
    def test_error_decomposition(self, crs_pair, kernel):
        """Break down error: projection-only, kernel-only, total."""
        src_crs, dst_crs = crs_pair
        s = _make_setup(src_crs, dst_crs, size=64)

        # 1. Full rust-warp (own projection + own kernels)
        rust_full = reproject_array(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling=kernel,
        )

        # 2. Hybrid: pyproj coordinates + rust-warp kernels
        pp_col, pp_row = _compute_pyproj_grid(
            s["dst_shape"], s["dst_transform"],
            s["src_transform"], s["dst_crs"], s["src_crs"],
        )
        rust_hybrid = reproject_with_grid(
            s["src"],
            pp_col.astype(np.float64),
            pp_row.astype(np.float64),
            resampling=kernel,
        )

        # 3. GDAL reference (PROJ + GDAL kernels)
        gdal = _gdal_reproject(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling=kernel,
        )

        # Where all three are valid
        all_valid = ~np.isnan(rust_full) & ~np.isnan(rust_hybrid) & ~np.isnan(gdal)
        if not all_valid.any():
            pytest.skip("No overlapping valid pixels")

        total_diff = np.abs(rust_full[all_valid] - gdal[all_valid])
        kernel_diff = np.abs(rust_hybrid[all_valid] - gdal[all_valid])
        proj_diff = np.abs(rust_full[all_valid] - rust_hybrid[all_valid])

        total_stats = {
            "max": float(total_diff.max()),
            "mean": float(total_diff.mean()),
            "p95": float(np.percentile(total_diff, 95)),
        }
        kernel_stats = {
            "max": float(kernel_diff.max()),
            "mean": float(kernel_diff.mean()),
            "p95": float(np.percentile(kernel_diff, 95)),
        }
        proj_stats = {
            "max": float(proj_diff.max()),
            "mean": float(proj_diff.mean()),
            "p95": float(np.percentile(proj_diff, 95)),
        }

        print(
            f"\n  DECOMP [{src_crs}->{dst_crs} {kernel}]  n={int(all_valid.sum())}"
            f"\n    total:  max={total_stats['max']:8.4f}  mean={total_stats['mean']:.6f}"
            f"  p95={total_stats['p95']:.4f}"
            f"\n    kernel: max={kernel_stats['max']:8.4f}  mean={kernel_stats['mean']:.6f}"
            f"  p95={kernel_stats['p95']:.4f}"
            f"\n    proj:   max={proj_stats['max']:8.4f}  mean={proj_stats['mean']:.6f}"
            f"  p95={proj_stats['p95']:.4f}"
        )

        # Kernel error with matched coords should be small
        assert kernel_stats["max"] <= 5.0, (
            f"Kernel error {kernel_stats['max']:.2f} with pyproj coords — kernel bug?"
        )


class TestScaleFactorError:
    """How does error change with upscale/downscale ratio?"""

    @pytest.mark.parametrize("scale", [0.25, 0.5, 1.0, 2.0, 4.0])
    @pytest.mark.parametrize("kernel", ["bilinear", "lanczos"])
    def test_error_vs_scale(self, scale, kernel):
        """Report error at different scale factors (same CRS)."""
        src_size = 64
        dst_size = max(4, int(src_size * scale))
        px_src = 100.0
        px_dst = px_src / scale
        origin_x, origin_y = 500000.0, 6600000.0 + src_size * px_src
        src_transform = (px_src, 0.0, origin_x, 0.0, -px_src, origin_y)
        dst_transform = (px_dst, 0.0, origin_x, 0.0, -px_dst, origin_y)

        src_crs = "EPSG:32633"
        src = np.arange(src_size * src_size, dtype=np.float64).reshape(src_size, src_size)

        rust = reproject_array(
            src, src_crs, src_transform,
            src_crs, dst_transform, (dst_size, dst_size),
            resampling=kernel,
        )
        gdal = _gdal_reproject(
            src, src_crs, src_transform,
            src_crs, dst_transform, (dst_size, dst_size),
            resampling=kernel,
        )

        stats = _error_stats(rust, gdal)

        print(
            f"\n  SCALE [{kernel} {scale:.2f}x ({src_size}->{dst_size})]"
            f"  n={stats['n']}  "
            f"max={stats['max']:.4f}  mean={stats['mean']:.6f}  rmse={stats['rmse']:.6f}"
        )

        if scale >= 1.0:
            # Upscale / identity: same CRS should be exact or near-exact
            assert stats["max"] < 1.0, (
                f"Same-CRS {kernel} at {scale}x: max error {stats['max']:.4f} >= 1.0"
            )
        else:
            # Downscale with interpolating kernels diverges from GDAL at borders;
            # this is expected — use "average" for downscaling.
            # Just verify it's bounded.
            assert stats["max"] < 100, (
                f"Same-CRS {kernel} at {scale}x: max error {stats['max']:.4f} >= 100"
            )


class TestNonSquarePixelError:
    """Error when source has non-square pixels."""

    @pytest.mark.parametrize("aspect", [(2, 1), (1, 2), (3, 1), (1, 3)])
    @pytest.mark.parametrize("kernel", ["bilinear", "lanczos"])
    def test_nonsquare_cross_crs_error(self, aspect, kernel):
        """Report error for non-square pixels in cross-CRS reprojection."""
        ax, ay = aspect
        size = 32
        res_x, res_y = 100.0 * ax, 100.0 * ay
        rows = size
        cols = size
        origin_x, origin_y = 500000.0, 6600000.0 + rows * res_y
        src_transform = (res_x, 0.0, origin_x, 0.0, -res_y, origin_y)

        r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        src = ((r * cols + c) / (rows * cols) * 1000.0).astype(np.float64)

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:32633"),
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
            src, "EPSG:32633", src_transform,
            "EPSG:4326", dst_transform, dst_shape,
            resampling=kernel,
        )
        gdal = _gdal_reproject(
            src, "EPSG:32633", src_transform,
            "EPSG:4326", dst_transform, dst_shape,
            resampling=kernel,
        )

        stats = _error_stats(rust, gdal)

        print(
            f"\n  NONSQUARE [{kernel} {ax}:{ay} aspect]"
            f"  n={stats['n']}  "
            f"max={stats['max']:.4f}  mean={stats['mean']:.6f}  "
            f"p95={stats['p95']:.4f}  rmse={stats['rmse']:.6f}"
        )

        # Non-square amplifies projection differences; generous threshold
        assert stats["max"] < 50, (
            f"Non-square {ax}:{ay} {kernel}: max error {stats['max']:.2f} >= 50"
        )


class TestErrorAtImageEdges:
    """Compare error at image edges vs interior."""

    @pytest.mark.parametrize("kernel", ["bilinear", "cubic", "lanczos"])
    def test_edge_vs_interior_error(self, kernel):
        """Edge pixels should have higher error than interior pixels."""
        s = _make_setup("EPSG:32633", "EPSG:4326", size=64)

        rust = reproject_array(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling=kernel,
        )
        gdal = _gdal_reproject(
            s["src"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling=kernel,
        )

        rows, cols = s["dst_shape"]
        margin = 8

        # Interior
        rust_int = rust[margin:-margin, margin:-margin]
        gdal_int = gdal[margin:-margin, margin:-margin]
        int_stats = _error_stats(rust_int, gdal_int)

        # Edge band
        edge_mask = np.ones(s["dst_shape"], dtype=bool)
        edge_mask[margin:-margin, margin:-margin] = False
        rust_edge = np.where(edge_mask, rust, np.nan)
        gdal_edge = np.where(edge_mask, gdal, np.nan)
        edge_stats = _error_stats(rust_edge, gdal_edge)

        print(
            f"\n  EDGE/INT [{kernel}]"
            f"\n    interior: n={int_stats['n']}  max={int_stats['max']:.4f}"
            f"  mean={int_stats['mean']:.6f}"
            f"\n    edge:     n={edge_stats['n']}  max={edge_stats['max']:.4f}"
            f"  mean={edge_stats['mean']:.6f}"
        )

        # Interior error should be less than edge error (or edge has fewer valid pixels)
        if int_stats["n"] > 0 and edge_stats["n"] > 0:
            assert int_stats["mean"] <= edge_stats["mean"] + 0.01, (
                f"Interior error ({int_stats['mean']:.4f}) "
                f"not better than edge ({edge_stats['mean']:.4f})"
            )
