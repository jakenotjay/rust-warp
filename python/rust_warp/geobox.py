"""GeoBox: a georeferenced bounding box with CRS, affine transform, and shape."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeoBox:
    """A georeferenced grid definition.

    Combines a CRS string, an affine transform, and a pixel shape to fully
    describe a regular grid in projected or geographic coordinates.

    Attributes:
        crs: CRS string (e.g. "EPSG:32633").
        shape: Grid shape as (rows, cols).
        affine: Affine transform as (a, b, c, d, e, f) where
            x = a * col + b * row + c, y = d * col + e * row + f.
    """

    crs: str
    shape: tuple[int, int]
    affine: tuple[float, float, float, float, float, float]

    @classmethod
    def from_bbox(
        cls,
        bbox: tuple[float, float, float, float],
        crs: str,
        resolution: float | tuple[float, float] | None = None,
        shape: tuple[int, int] | None = None,
    ) -> GeoBox:
        """Create a GeoBox from a bounding box.

        Args:
            bbox: (left, bottom, right, top) in CRS units.
            crs: CRS string.
            resolution: Pixel size as scalar or (res_x, res_y). Required if
                shape is not provided.
            shape: Grid shape as (rows, cols). Required if resolution is not
                provided.

        Returns:
            A new GeoBox instance.
        """
        left, bottom, right, top = bbox
        width = right - left
        height = top - bottom

        if shape is not None:
            rows, cols = shape
            res_x = width / cols
            res_y = height / rows
        elif resolution is not None:
            if isinstance(resolution, (int, float)):
                res_x = float(resolution)
                res_y = float(resolution)
            else:
                res_x, res_y = float(resolution[0]), float(resolution[1])
            cols = max(1, int(round(width / res_x)))
            rows = max(1, int(round(height / res_y)))
        else:
            raise ValueError("Either resolution or shape must be provided")

        # North-up convention: origin at top-left, e is negative
        affine = (res_x, 0.0, left, 0.0, -res_y, top)
        return cls(crs=crs, shape=(rows, cols), affine=affine)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Bounding box as (left, bottom, right, top) in CRS units."""
        a, b, c, d, e, f = self.affine
        rows, cols = self.shape
        # Compute all 4 corners
        corners_col = [0, cols, 0, cols]
        corners_row = [0, 0, rows, rows]
        xs = [a * cc + b * cr + c for cc, cr in zip(corners_col, corners_row)]
        ys = [d * cc + e * cr + f for cc, cr in zip(corners_col, corners_row)]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def resolution(self) -> tuple[float, float]:
        """Pixel resolution as (res_x, res_y), both positive."""
        return (abs(self.affine[0]), abs(self.affine[4]))

    def xr_coords(self) -> dict[str, np.ndarray]:
        """Coordinate arrays for xarray DataArray construction.

        Returns:
            Dict with "x" and "y" keys, each a 1D array of pixel-center
            coordinates.
        """
        a, b, c, d, e, f = self.affine
        rows, cols = self.shape
        x = c + a * (np.arange(cols) + 0.5) + b * 0.5
        y = f + e * (np.arange(rows) + 0.5) + d * 0.5
        return {"x": x, "y": y}
