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
) -> NDArray[np.float64]: ...
