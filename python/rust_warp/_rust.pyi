import numpy as np
from numpy.typing import NDArray

def hello() -> str: ...

def reproject_array(
    src: NDArray[np.float64],
    src_crs: str,
    src_transform: tuple[float, float, float, float, float, float],
    dst_crs: str,
    dst_transform: tuple[float, float, float, float, float, float],
    dst_shape: tuple[int, int],
    resampling: str = "nearest",
    nodata: float | None = None,
) -> NDArray[np.float64]:
    """Reproject a 2D f64 array from one CRS to another.

    Args:
        src: Input 2D array (f64).
        src_crs: Source CRS string (e.g. "EPSG:32633").
        src_transform: Source affine transform as 6-element tuple.
        dst_crs: Destination CRS string.
        dst_transform: Destination affine transform as 6-element tuple.
        dst_shape: Output shape as (rows, cols) tuple.
        resampling: Resampling method — one of "nearest", "bilinear",
            "cubic", "lanczos", or "average".
        nodata: Optional nodata value.

    Returns:
        Reprojected 2D array (f64).
    """
    ...
