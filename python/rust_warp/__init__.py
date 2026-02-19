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
    "hello",
    "reproject_array",
    "reproject_with_grid",
    "transform_grid",
    "transform_points",
    "plan_reproject",
    "GeoBox",
    "reproject",
]

try:
    import rust_warp.xarray_accessor  # noqa: F401
except ImportError:
    pass  # xarray not installed
