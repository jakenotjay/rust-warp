"""GeoBox: a georeferenced bounding box with CRS, affine transform, and shape."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SPATIAL_Y_NAMES = ("y", "latitude", "lat")
_SPATIAL_X_NAMES = ("x", "longitude", "lon")


def _find_spatial_dims(obj) -> tuple[str, str]:
    """Detect spatial (y, x) dimension names from an xarray object.

    Checks dim names against known spatial names. Falls back to the last two
    dimensions if no known names match.

    Returns:
        Tuple of (y_dim, x_dim).

    Raises:
        ValueError: If spatial dimensions cannot be determined.
    """
    dims = list(obj.dims)
    if len(dims) < 2:
        raise ValueError(
            f"Need at least 2 dimensions, got {len(dims)}: {dims}"
        )

    # Check known spatial names
    for y_name in _SPATIAL_Y_NAMES:
        for x_name in _SPATIAL_X_NAMES:
            if y_name in dims and x_name in dims:
                return (y_name, x_name)

    # Fall back to last two dims
    return (dims[-2], dims[-1])


def _extract_crs(obj) -> str:
    """Extract CRS string from an xarray object.

    Priority:
    1. coords["spatial_ref"].attrs["crs_wkt"]
    2. obj.attrs["crs"]

    Returns:
        CRS string.

    Raises:
        ValueError: If no CRS can be found.
    """
    # Check spatial_ref coordinate
    if "spatial_ref" in obj.coords:
        sr = obj.coords["spatial_ref"]
        # Prefer EPSG code for compact, round-trippable representation
        if "epsg" in sr.attrs:
            return f"EPSG:{sr.attrs['epsg']}"
        if "crs_wkt" in sr.attrs:
            # Try to recover the EPSG code from WKT via pyproj, since the
            # Rust proj4rs backend cannot parse raw WKT strings.
            wkt = str(sr.attrs["crs_wkt"])
            try:
                from pyproj import CRS as PyprojCRS  # noqa: N811

                epsg = PyprojCRS.from_wkt(wkt).to_epsg()
                if epsg is not None:
                    return f"EPSG:{epsg}"
            except Exception:
                pass
            return wkt

    # Check attrs
    if "crs" in obj.attrs:
        crs_val = str(obj.attrs["crs"])
        # If it looks like WKT, try to extract EPSG via pyproj
        if crs_val.startswith(("PROJCRS[", "GEOGCRS[", "COMPOUNDCRS[")):
            try:
                from pyproj import CRS as PyprojCRS  # noqa: N811

                epsg = PyprojCRS.from_wkt(crs_val).to_epsg()
                if epsg is not None:
                    return f"EPSG:{epsg}"
            except Exception:
                pass
        return crs_val

    raise ValueError(
        "Cannot determine CRS. Set a 'spatial_ref' coordinate with 'crs_wkt' attr, "
        "or set attrs['crs'] on the object."
    )


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

    @classmethod
    def from_xarray(cls, obj) -> GeoBox:
        """Create a GeoBox from an xarray DataArray or Dataset.

        Extracts CRS, affine transform, and shape from coordinate metadata.

        Args:
            obj: An xarray DataArray or Dataset with spatial coordinates.

        Returns:
            A new GeoBox instance.

        Raises:
            ValueError: If CRS or spatial dimensions cannot be determined.
        """
        y_dim, x_dim = _find_spatial_dims(obj)
        crs = _extract_crs(obj)

        x = obj.coords[x_dim].values
        y = obj.coords[y_dim].values

        if len(x) < 2 or len(y) < 2:
            raise ValueError("Need at least 2 points in each spatial dimension")

        res_x = float(x[1] - x[0])
        res_y = float(y[1] - y[0])
        origin_x = float(x[0]) - res_x / 2  # pixel-center → top-left corner
        origin_y = float(y[0]) - res_y / 2

        affine = (res_x, 0.0, origin_x, 0.0, res_y, origin_y)
        shape = (len(y), len(x))
        return cls(crs=crs, shape=shape, affine=affine)

    @classmethod
    def from_odc(cls, gbox) -> GeoBox:
        """Create from an odc-geo GeoBox.

        Args:
            gbox: An odc.geo.geobox.GeoBox instance.

        Returns:
            A new GeoBox instance.
        """
        crs = str(gbox.crs)
        shape = (gbox.shape.y, gbox.shape.x)
        a = gbox.affine
        affine = (a.a, a.b, a.c, a.d, a.e, a.f)
        return cls(crs=crs, shape=shape, affine=affine)

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
