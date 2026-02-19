"""Shared test fixtures for rust-warp."""

import numpy as np
import pytest
import rasterio.warp
from rasterio.crs import CRS


@pytest.fixture
def utm33_to_4326_setup():
    """Create a synthetic 64x64 UTM 33N raster and compute EPSG:4326 destination grid."""
    src_rows, src_cols = 64, 64
    src_crs = "EPSG:32633"
    pixel_size = 100.0  # 100m pixels
    origin_x, origin_y = 500000.0, 6600000.0

    # Affine: (a=pixel_width, b=row_rot, c=x_origin, d=col_rot, e=-pixel_height, f=y_origin)
    src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

    # Synthetic data: gradient pattern
    rows, cols = np.meshgrid(np.arange(src_rows), np.arange(src_cols), indexing="ij")
    src = (rows * src_cols + cols).astype(np.float64)

    # Compute destination grid using rasterio
    dst_crs = "EPSG:4326"
    dst_transform_affine, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        CRS.from_user_input(src_crs),
        CRS.from_user_input(dst_crs),
        src_cols,
        src_rows,
        left=origin_x,
        bottom=origin_y - src_rows * pixel_size,
        right=origin_x + src_cols * pixel_size,
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

    # 4× downscale: 128→32
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
    src_rows, src_cols = 64, 64
    src_crs = "EPSG:32633"
    pixel_size = 100.0
    origin_x, origin_y = 500000.0, 6600000.0

    src_transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

    rows, cols = np.meshgrid(np.arange(src_rows), np.arange(src_cols), indexing="ij")
    src = (rows * src_cols + cols).astype(np.float64)

    dst_crs = "EPSG:3857"
    dst_transform_affine, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        CRS.from_user_input(src_crs),
        CRS.from_user_input(dst_crs),
        src_cols,
        src_rows,
        left=origin_x,
        bottom=origin_y - src_rows * pixel_size,
        right=origin_x + src_cols * pixel_size,
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
