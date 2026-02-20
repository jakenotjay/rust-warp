"""rust-warp: High-performance raster reprojection engine."""

from rust_warp._rust import (
    hello,
    plan_reproject,
    reproject_array,
    reproject_with_grid,
    transform_grid,
    transform_points,
)
from rust_warp.geobox import GeoBox
from rust_warp.reproject import reproject

__all__ = [
    "GeoBox",
    "hello",
    "plan_reproject",
    "reproject",
    "reproject_array",
    "reproject_with_grid",
    "transform_grid",
    "transform_points",
]

import contextlib

with contextlib.suppress(ImportError):
    import rust_warp.xarray_accessor  # noqa: F401
