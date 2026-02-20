"""Shared test fixtures and comparison utilities for rust-warp."""

import numpy as np
import pytest
import rasterio.warp
from rasterio.crs import CRS

# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def compare_arrays(actual, expected, method, atol=1.0, rtol=1e-4):
    """Compare two arrays and return detailed error statistics.

    Args:
        actual: The array produced by rust-warp.
        expected: The reference array (from GDAL/rasterio).
        method: Resampling method name (for reporting).
        atol: Absolute tolerance for the assertion.
        rtol: Relative tolerance for the assertion.

    Returns:
        Dict with error statistics.

    Raises:
        AssertionError if tolerances are exceeded.
    """
    actual_valid = (
        ~np.isnan(actual)
        if np.issubdtype(actual.dtype, np.floating)
        else np.ones(actual.shape, dtype=bool)
    )
    expected_valid = (
        ~np.isnan(expected)
        if np.issubdtype(expected.dtype, np.floating)
        else np.ones(expected.shape, dtype=bool)
    )
    both_valid = actual_valid & expected_valid

    n_valid = int(both_valid.sum())
    if n_valid == 0:
        return {
            "method": method,
            "n_valid": 0,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "rmse": 0.0,
            "pct_differing": 0.0,
            "pct_error_gt_1": 0.0,
            "worst_pixel": None,
        }

    a = actual[both_valid].astype(np.float64)
    e = expected[both_valid].astype(np.float64)
    diff = np.abs(a - e)

    max_abs_error = float(diff.max())
    mean_abs_error = float(diff.mean())
    rmse = float(np.sqrt((diff**2).mean()))
    pct_differing = float(np.sum(diff > 0) / n_valid * 100)
    pct_error_gt_1 = float(np.sum(diff > 1.0) / n_valid * 100)

    # Find worst pixel location
    worst_idx = np.argmax(diff)
    valid_indices = np.argwhere(both_valid)
    worst_pixel = tuple(valid_indices[worst_idx]) if len(valid_indices) > 0 else None

    stats = {
        "method": method,
        "n_valid": n_valid,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "rmse": rmse,
        "pct_differing": pct_differing,
        "pct_error_gt_1": pct_error_gt_1,
        "worst_pixel": worst_pixel,
    }

    # Check tolerances
    if method == "nearest":
        # For nearest, use absolute difference
        assert max_abs_error <= atol, (
            f"[{method}] max_abs_error={max_abs_error:.4f} > atol={atol}, "
            f"mean={mean_abs_error:.4f}, rmse={rmse:.4f}, "
            f"worst_pixel={worst_pixel}"
        )
    else:
        # For interpolating kernels, check both absolute and relative
        rel_denom = np.maximum(np.abs(e), 1e-10)
        max_rel_error = float((diff / rel_denom).max())
        assert max_abs_error <= atol or max_rel_error <= rtol, (
            f"[{method}] max_abs_error={max_abs_error:.4f} (atol={atol}), "
            f"max_rel_error={max_rel_error:.6f} (rtol={rtol}), "
            f"mean={mean_abs_error:.4f}, rmse={rmse:.4f}, "
            f"worst_pixel={worst_pixel}"
        )

    return stats


def synthetic_raster(shape, dtype=np.float64, pattern="gradient"):
    """Generate a synthetic test raster.

    Args:
        shape: (rows, cols) tuple.
        dtype: Output dtype.
        pattern: One of "gradient", "checkerboard", "random", "constant", "sinusoidal".

    Returns:
        2D numpy array with the specified pattern.
    """
    rows, cols = shape
    if pattern == "gradient":
        r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        data = (r * cols + c).astype(np.float64)
    elif pattern == "checkerboard":
        r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        data = ((r + c) % 2).astype(np.float64) * 255.0
    elif pattern == "random":
        rng = np.random.default_rng(42)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            data = rng.integers(1, min(info.max, 255), size=shape).astype(np.float64)
        else:
            data = rng.random(shape) * 1000.0
    elif pattern == "constant":
        data = np.full(shape, 42.0, dtype=np.float64)
    elif pattern == "sinusoidal":
        r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        data = 128.0 + 100.0 * np.sin(2 * np.pi * r / rows) + 50.0 * np.cos(4 * np.pi * c / cols)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return data.astype(dtype)


def gdal_reproject(
    src, src_crs, src_transform, dst_crs, dst_transform, dst_shape, resampling, nodata=np.nan
):
    """Reproject using rasterio/GDAL as reference implementation."""
    resampling_map = {
        "nearest": rasterio.warp.Resampling.nearest,
        "bilinear": rasterio.warp.Resampling.bilinear,
        "cubic": rasterio.warp.Resampling.cubic,
        "lanczos": rasterio.warp.Resampling.lanczos,
        "average": rasterio.warp.Resampling.average,
    }

    # Use source dtype for integer types, float64 for float
    if np.issubdtype(src.dtype, np.floating):
        dst = np.full(dst_shape, np.nan, dtype=src.dtype)
        dst_nodata = np.nan
    else:
        fill_val = int(nodata) if not np.isnan(nodata) else 0
        dst = np.full(dst_shape, fill_val, dtype=src.dtype)
        dst_nodata = fill_val

    if not np.isnan(nodata):
        src_nodata = nodata
    elif np.issubdtype(src.dtype, np.floating):
        src_nodata = np.nan
    else:
        src_nodata = None

    rasterio.warp.reproject(
        source=src,
        destination=dst,
        src_transform=rasterio.transform.Affine(*src_transform),
        src_crs=CRS.from_user_input(src_crs),
        dst_transform=rasterio.transform.Affine(*dst_transform),
        dst_crs=CRS.from_user_input(dst_crs),
        resampling=resampling_map[resampling],
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
    )
    return dst


def make_reprojection_setup(src_crs, dst_crs, size=64, pixel_size=100.0):
    """Create a reprojection test setup for arbitrary CRS pair.

    Args:
        src_crs: Source CRS string.
        dst_crs: Destination CRS string.
        size: Raster size (square).
        pixel_size: Source pixel size in CRS units.

    Returns:
        Dict with src, src_crs, src_transform, dst_crs, dst_transform, dst_shape.
    """
    # Choose appropriate origin for each CRS
    origins = {
        "EPSG:32633": (500000.0, 6600000.0),
        "EPSG:32617": (500000.0, 4400000.0),
        "EPSG:3857": (1600000.0, 8300000.0),
        "EPSG:4326": (15.0, 60.0),
    }
    origin = origins.get(src_crs, (500000.0, 6600000.0))
    origin_x, origin_y = origin

    # For geographic CRS, use degree-scale pixels
    if src_crs == "EPSG:4326":
        pixel_size = 0.01

    src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    # Normalize to [0, 1000] range so absolute errors don't scale with raster size
    src = ((rows * size + cols) / (size * size) * 1000.0).astype(np.float64)

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
        "src": src,
        "src_crs": src_crs,
        "src_transform": src_transform,
        "dst_crs": dst_crs,
        "dst_transform": dst_transform,
        "dst_shape": dst_shape,
    }


# ---------------------------------------------------------------------------
# CRS pair parametrization
# ---------------------------------------------------------------------------

CRS_PAIRS = [
    ("EPSG:32633", "EPSG:4326"),  # UTM33 → Geographic
    ("EPSG:4326", "EPSG:32633"),  # Geographic → UTM33
    ("EPSG:32633", "EPSG:3857"),  # UTM33 → Web Mercator
    ("EPSG:3857", "EPSG:4326"),  # Web Mercator → Geographic
    ("EPSG:4326", "EPSG:32617"),  # Geographic → UTM17
    ("EPSG:32633", "EPSG:32617"),  # UTM33 → UTM17
]


# ---------------------------------------------------------------------------
# Original fixtures (preserved for backward compatibility)
# ---------------------------------------------------------------------------


@pytest.fixture
def utm33_to_4326_setup():
    """Create a synthetic 64x64 UTM 33N raster and compute EPSG:4326 destination grid."""
    return make_reprojection_setup("EPSG:32633", "EPSG:4326", size=64)


@pytest.fixture
def downscale_setup():
    """Create a 128x128 source raster and 32x32 destination for average downsampling."""
    src_rows, src_cols = 128, 128
    crs = "EPSG:32633"
    pixel_size = 100.0
    origin_x, origin_y = 500000.0, 6600000.0

    src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

    rows, cols = np.meshgrid(np.arange(src_rows), np.arange(src_cols), indexing="ij")
    src = (rows * src_cols + cols).astype(np.float64)

    # 4x downscale: 128->32
    dst_rows, dst_cols = 32, 32
    dst_pixel_size = pixel_size * (src_rows / dst_rows)
    dst_transform = (dst_pixel_size, 0.0, origin_x, 0.0, -dst_pixel_size, origin_y)

    return {
        "src": src,
        "src_crs": crs,
        "src_transform": src_transform,
        "dst_crs": crs,
        "dst_transform": dst_transform,
        "dst_shape": (dst_rows, dst_cols),
    }


@pytest.fixture
def utm33_to_3857_setup():
    """Create a synthetic 64x64 UTM 33N raster and compute EPSG:3857 destination grid."""
    return make_reprojection_setup("EPSG:32633", "EPSG:3857", size=64)
