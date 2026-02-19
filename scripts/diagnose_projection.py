"""Diagnostic script: compare rust-warp coordinate transforms vs pyproj.

For each CRS pair in the test matrix, computes source pixel coordinates via
both rust-warp's transform_grid() and pyproj, then reports the difference
in source pixel units.
"""

import numpy as np
import pyproj
import rasterio.warp
from rasterio.crs import CRS

from rust_warp import transform_grid

# ---------------------------------------------------------------------------
# Test matrix — same as conftest.py
# ---------------------------------------------------------------------------

CRS_PAIRS = [
    ("EPSG:32633", "EPSG:4326"),
    ("EPSG:4326", "EPSG:32633"),
    ("EPSG:32633", "EPSG:3857"),
    ("EPSG:3857", "EPSG:4326"),
    ("EPSG:4326", "EPSG:32617"),
    ("EPSG:32633", "EPSG:32617"),
]


def make_reprojection_setup(src_crs, dst_crs, size=64, pixel_size=100.0):
    """Same setup as conftest.py."""
    origins = {
        "EPSG:32633": (500000.0, 6600000.0),
        "EPSG:32617": (500000.0, 4400000.0),
        "EPSG:3857": (1600000.0, 8300000.0),
        "EPSG:4326": (15.0, 60.0),
    }
    origin_x, origin_y = origins.get(src_crs, (500000.0, 6600000.0))

    if src_crs == "EPSG:4326":
        pixel_size = 0.01

    src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

    dst_transform_affine, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        CRS.from_user_input(src_crs),
        CRS.from_user_input(dst_crs),
        size,
        size,
        left=origin_x,
        bottom=origin_y - size * pixel_size,
        right=origin_x + size * pixel_size,
        top=origin_y,
    )
    dst_transform = tuple(dst_transform_affine)[:6]
    dst_shape = (dst_height, dst_width)

    return {
        "src_crs": src_crs,
        "src_transform": src_transform,
        "dst_crs": dst_crs,
        "dst_transform": dst_transform,
        "dst_shape": dst_shape,
    }


def compute_pyproj_grid(setup):
    """Compute source pixel coordinates using pyproj (matching what GDAL does)."""
    dst_shape = setup["dst_shape"]
    dst_transform = setup["dst_transform"]
    src_transform = setup["src_transform"]

    # Build affine objects
    a, b, c, d, e, f = dst_transform
    sa, sb, sc, sd, se, sf = src_transform

    # Source affine inverse
    det = sa * se - sb * sd
    inv_sa = se / det
    inv_sb = -sb / det
    inv_sc = (sb * sf - se * sc) / det
    inv_sd = -sd / det
    inv_se = sa / det
    inv_sf = (sd * sc - sa * sf) / det

    dst_rows, dst_cols = dst_shape
    col_grid = np.full(dst_shape, np.nan)
    row_grid = np.full(dst_shape, np.nan)

    # Create transformer
    transformer = pyproj.Transformer.from_crs(
        setup["dst_crs"], setup["src_crs"], always_xy=True
    )

    # Build destination pixel center coordinates
    cols_idx = np.arange(dst_cols) + 0.5
    rows_idx = np.arange(dst_rows) + 0.5
    cc, rr = np.meshgrid(cols_idx, rows_idx)

    # Destination pixel -> destination CRS coords
    dst_x = a * cc + b * rr + c
    dst_y = d * cc + e * rr + f

    # Transform through pyproj (dst CRS -> src CRS)
    src_x, src_y = transformer.transform(dst_x.ravel(), dst_y.ravel())
    src_x = np.array(src_x).reshape(dst_shape)
    src_y = np.array(src_y).reshape(dst_shape)

    # Source CRS coords -> source pixel coords (using inverse affine)
    col_grid = inv_sa * src_x + inv_sb * src_y + inv_sc
    row_grid = inv_sd * src_x + inv_se * src_y + inv_sf

    return col_grid, row_grid


def main():
    print("=" * 90)
    print("Coordinate Transform Diagnostic: rust-warp vs pyproj")
    print("=" * 90)
    print()

    header = (
        f"{'CRS pair':<25s} | {'max_err_px':>10s} | {'mean_err_px':>11s} | "
        f"{'p99_err_px':>10s} | {'p50_err_px':>10s} | {'max_col':>8s} | {'max_row':>8s}"
    )
    print(header)
    print("-" * len(header))

    for src_crs, dst_crs in CRS_PAIRS:
        setup = make_reprojection_setup(src_crs, dst_crs, size=64)

        # rust-warp coordinates
        rw_col, rw_row = transform_grid(
            setup["src_crs"],
            setup["src_transform"],
            setup["dst_crs"],
            setup["dst_transform"],
            setup["dst_shape"],
        )

        # pyproj coordinates
        pp_col, pp_row = compute_pyproj_grid(setup)

        # Mask out NaN
        valid = ~np.isnan(rw_col) & ~np.isnan(rw_row) & ~np.isnan(pp_col) & ~np.isnan(pp_row)
        if not valid.any():
            print(f"{src_crs}->{dst_crs:<15s} | {'NO VALID PIXELS':>60s}")
            continue

        col_diff = np.abs(rw_col[valid] - pp_col[valid])
        row_diff = np.abs(rw_row[valid] - pp_row[valid])
        total_diff = np.sqrt(col_diff**2 + row_diff**2)

        max_err = float(total_diff.max())
        mean_err = float(total_diff.mean())
        p99_err = float(np.percentile(total_diff, 99))
        p50_err = float(np.percentile(total_diff, 50))
        max_col_err = float(col_diff.max())
        max_row_err = float(row_diff.max())

        label = f"{src_crs}->{dst_crs}"
        print(
            f"{label:<25s} | {max_err:10.4f} | {mean_err:11.4f} | "
            f"{p99_err:10.4f} | {p50_err:10.4f} | {max_col_err:8.4f} | {max_row_err:8.4f}"
        )

    print()
    print("=" * 90)
    print("Detailed analysis per CRS pair")
    print("=" * 90)

    for src_crs, dst_crs in CRS_PAIRS:
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
        if not valid.any():
            continue

        total_diff = np.sqrt(
            (rw_col[valid] - pp_col[valid]) ** 2 + (rw_row[valid] - pp_row[valid]) ** 2
        )

        # Check spatial distribution: are high errors concentrated at edges?
        dst_rows, dst_cols = setup["dst_shape"]
        error_map = np.full((dst_rows, dst_cols), np.nan)
        error_map[valid] = total_diff

        # Edge pixels: first/last 5 rows/cols
        edge_mask = np.zeros((dst_rows, dst_cols), dtype=bool)
        margin = min(5, dst_rows // 4, dst_cols // 4)
        edge_mask[:margin, :] = True
        edge_mask[-margin:, :] = True
        edge_mask[:, :margin] = True
        edge_mask[:, -margin:] = True

        interior = valid & ~edge_mask
        edge = valid & edge_mask

        label = f"{src_crs}->{dst_crs}"
        print(f"\n--- {label} ---")
        print(f"  Shape: {dst_rows}x{dst_cols}, valid pixels: {valid.sum()}")

        if interior.any():
            int_err = np.sqrt(
                (rw_col[interior] - pp_col[interior]) ** 2
                + (rw_row[interior] - pp_row[interior]) ** 2
            )
            print(f"  Interior: max={int_err.max():.4f}px, mean={int_err.mean():.4f}px")
        if edge.any():
            edge_err = np.sqrt(
                (rw_col[edge] - pp_col[edge]) ** 2 + (rw_row[edge] - pp_row[edge]) ** 2
            )
            print(f"  Edge:     max={edge_err.max():.4f}px, mean={edge_err.mean():.4f}px")

        # Error histogram
        bins = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, float("inf")]
        counts, _ = np.histogram(total_diff, bins=bins)
        print("  Error distribution:")
        for i, count in enumerate(counts):
            pct = count / len(total_diff) * 100
            lo = bins[i]
            hi = bins[i + 1]
            if hi == float("inf"):
                print(f"    >{lo:.2f}px: {count:6d} ({pct:5.1f}%)")
            else:
                print(f"    {lo:.2f}-{hi:.2f}px: {count:6d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
