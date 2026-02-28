"""Xarray accessors for raster reprojection via rust-warp.

Registers ``.warp`` on both ``xr.DataArray`` and ``xr.Dataset``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from rust_warp._backend import _derive_batch_size, detect_backend, stack_arrays
from rust_warp._rust import transform_points
from rust_warp.geobox import GeoBox, _extract_crs, _find_spatial_dims
from rust_warp.reproject import reproject

if TYPE_CHECKING:
    pass


def _as_geobox(obj) -> GeoBox:
    """Coerce *obj* to a rust-warp GeoBox.

    Accepts a rust-warp GeoBox, an odc-geo GeoBox (duck-typed), or an xarray
    DataArray/Dataset.
    """
    if isinstance(obj, GeoBox):
        return obj
    # Duck-type: odc-geo GeoBox has .affine, .crs, and .shape
    if hasattr(obj, "affine") and hasattr(obj, "crs") and hasattr(obj, "shape"):
        return GeoBox.from_odc(obj)
    return GeoBox.from_xarray(obj)


def _compute_dst_geobox(
    src_geobox: GeoBox,
    dst_crs: str,
    resolution: float | tuple[float, float] | None = None,
    shape: tuple[int, int] | None = None,
) -> GeoBox:
    """Compute a destination GeoBox by projecting the source boundary.

    Projects 21 points per edge of the source bounding box into *dst_crs* and
    computes the enclosing bounding box.
    """
    left, bottom, right, top = src_geobox.bounds

    # Build boundary points: 21 per edge (top, bottom, left, right)
    n = 21
    ts = np.linspace(0, 1, n)
    bx = np.concatenate(
        [
            left + ts * (right - left),  # top edge
            left + ts * (right - left),  # bottom edge
            np.full(n, left),  # left edge
            np.full(n, right),  # right edge
        ]
    )
    by = np.concatenate(
        [
            np.full(n, top),  # top edge
            np.full(n, bottom),  # bottom edge
            bottom + ts * (top - bottom),  # left edge
            bottom + ts * (top - bottom),  # right edge
        ]
    )

    dx, dy = transform_points(bx, by, src_geobox.crs, dst_crs)

    # Filter out non-finite points (projection failures at the edges)
    valid = np.isfinite(dx) & np.isfinite(dy)
    if not valid.any():
        raise ValueError(f"No valid projected points from {src_geobox.crs} to {dst_crs}")
    dx, dy = dx[valid], dy[valid]

    bbox = (float(dx.min()), float(dy.min()), float(dx.max()), float(dy.max()))

    if shape is not None:
        return GeoBox.from_bbox(bbox, dst_crs, shape=shape)
    if resolution is not None:
        return GeoBox.from_bbox(bbox, dst_crs, resolution=resolution)

    # Auto-resolve: preserve total pixel count
    src_rows, src_cols = src_geobox.shape
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    aspect = width / height if height != 0 else 1.0
    total_pixels = src_rows * src_cols
    dst_cols = max(1, round(np.sqrt(total_pixels * aspect)))
    dst_rows = max(1, round(total_pixels / dst_cols))
    return GeoBox.from_bbox(bbox, dst_crs, shape=(dst_rows, dst_cols))


def _make_spatial_ref_coord(crs_str: str) -> xr.DataArray:
    """Create a scalar ``spatial_ref`` coordinate with CRS metadata."""
    attrs: dict = {"crs_wkt": crs_str}
    try:
        from pyproj import CRS

        crs_obj = CRS.from_user_input(crs_str)
        attrs["crs_wkt"] = crs_obj.to_wkt()
        # Store EPSG code if available, for exact round-tripping
        epsg = crs_obj.to_epsg()
        if epsg is not None:
            attrs["epsg"] = epsg
    except Exception:
        pass
    return xr.DataArray(0, attrs=attrs)


def _reproject_dataarray(
    da: xr.DataArray,
    src_geobox: GeoBox,
    dst_geobox: GeoBox,
    resampling: str = "bilinear",
    nodata: float | None = None,
    dst_chunks: tuple[int, int] | None = None,
    batch_size: int | None = None,
    max_task_bytes: int = 256 * 1024 * 1024,
) -> xr.DataArray:
    """Reproject a DataArray, handling both 2D and N-D cases."""
    y_dim, x_dim = _find_spatial_dims(da)
    spatial_axes = (da.dims.index(y_dim), da.dims.index(x_dim))

    data = da.data  # numpy or dask array

    if data.ndim == 2:
        result_data = reproject(
            data,
            src_geobox,
            dst_geobox,
            resampling=resampling,
            nodata=nodata,
            dst_chunks=dst_chunks,
        )
    else:
        result_data = _reproject_nd(
            data,
            src_geobox,
            dst_geobox,
            spatial_axes=spatial_axes,
            resampling=resampling,
            nodata=nodata,
            dst_chunks=dst_chunks,
            batch_size=batch_size,
            max_task_bytes=max_task_bytes,
        )

    # Build new coordinates
    dst_coords = dst_geobox.xr_coords()
    new_coords: dict = {}

    # Preserve non-spatial coordinates
    for name, coord in da.coords.items():
        if name in (y_dim, x_dim, "spatial_ref"):
            continue
        # Only keep coords whose dims don't include the spatial dims
        coord_dims = set(coord.dims)
        if not coord_dims & {y_dim, x_dim}:
            new_coords[name] = coord

    # Spatial coordinates using the original dim names
    new_coords[y_dim] = (y_dim, dst_coords["y"])
    new_coords[x_dim] = (x_dim, dst_coords["x"])
    new_coords["spatial_ref"] = _make_spatial_ref_coord(dst_geobox.crs)

    # Build dims list — same order as input but with new spatial sizes
    dims = list(da.dims)

    # Filter attrs: remove stale geo metadata
    attrs = {k: v for k, v in da.attrs.items() if k not in ("crs", "transform")}

    result = xr.DataArray(
        data=result_data,
        dims=dims,
        coords=new_coords,
        name=da.name,
        attrs=attrs,
    )
    return result


def _reproject_nd(
    data,
    src_geobox: GeoBox,
    dst_geobox: GeoBox,
    spatial_axes: tuple[int, int],
    resampling: str = "bilinear",
    nodata: float | None = None,
    dst_chunks: tuple[int, int] | None = None,
    batch_size: int | None = None,
    max_task_bytes: int = 256 * 1024 * 1024,
):
    """Reproject an N-D array by iterating over non-spatial dimensions."""
    backend = detect_backend(data)

    ndim = data.ndim
    y_ax, x_ax = spatial_axes
    dst_rows, dst_cols = dst_geobox.shape

    # Build output shape
    out_shape = list(data.shape)
    out_shape[y_ax] = dst_rows
    out_shape[x_ax] = dst_cols

    # Non-spatial axis indices
    non_spatial = [i for i in range(ndim) if i not in (y_ax, x_ax)]

    if backend is not None:
        if backend == "dask":
            from rust_warp.dask_graph import reproject_dask as reproject_chunked
        else:
            from rust_warp.cubed_graph import reproject_cubed as reproject_chunked

        # Use dst_chunks if explicit; otherwise proxy with src chunk size (may
        # underestimate tile bytes for cross-CRS warps with large scale changes).
        sp_chunks = dst_chunks or (data.chunks[y_ax][0], data.chunks[x_ax][0])
        actual_batch = _derive_batch_size(data.dtype, sp_chunks, batch_size, max_task_bytes)

        # Collect selectors for all non-spatial slices
        all_sels = []
        for idx in np.ndindex(*[data.shape[i] for i in non_spatial]):
            sel = [slice(None)] * ndim
            for ax, ix in zip(non_spatial, idx, strict=True):
                sel[ax] = ix
            all_sels.append(tuple(sel))

        slices_2d = [data[sel] for sel in all_sels]  # list of 2D chunked arrays

        # Stack → (n_slices, y, x); rechunk batch dimension
        stacked = stack_arrays(slices_2d, axis=0, backend=backend)
        actual_batch = min(actual_batch, len(slices_2d))
        stacked = stacked.rechunk({0: actual_batch})

        # Single chunked reproject call on the 3D stack
        reprojected_3d = reproject_chunked(
            stacked,
            src_geobox,
            dst_geobox,
            resampling=resampling,
            dst_chunks=dst_chunks,
            nodata=nodata,
        )
        # reprojected_3d: (n_slices, dst_rows, dst_cols) → reshape to out_shape
        return reprojected_3d.reshape(out_shape)

    # Numpy path
    out = np.empty(out_shape, dtype=data.dtype)
    for idx in np.ndindex(*[data.shape[i] for i in non_spatial]):
        sel = [slice(None)] * ndim
        for ax, ix in zip(non_spatial, idx, strict=True):
            sel[ax] = ix
        slice_2d = np.ascontiguousarray(data[tuple(sel)])
        reprojected = reproject(
            slice_2d,
            src_geobox,
            dst_geobox,
            resampling=resampling,
            nodata=nodata,
        )
        out[tuple(sel)] = reprojected

    return out


@xr.register_dataarray_accessor("warp")
class WarpDataArrayAccessor:
    """Raster reprojection accessor for xarray DataArray."""

    def __init__(self, da: xr.DataArray):
        self._da = da

    @property
    def geobox(self) -> GeoBox:
        """GeoBox derived from this DataArray's coordinates and CRS."""
        return GeoBox.from_xarray(self._da)

    @property
    def crs(self) -> str | None:
        """CRS string, or None if not set."""
        try:
            return _extract_crs(self._da)
        except ValueError:
            return None

    def reproject(
        self,
        dst_crs: str,
        resolution: float | tuple[float, float] | None = None,
        shape: tuple[int, int] | None = None,
        resampling: str = "bilinear",
        nodata: float | None = None,
        dst_chunks: tuple[int, int] | None = None,
        batch_size: int | None = None,
        max_task_bytes: int = 256 * 1024 * 1024,
    ) -> xr.DataArray:
        """Reproject to a new CRS.

        Args:
            dst_crs: Target CRS string (e.g. ``"EPSG:4326"``).
            resolution: Target pixel size. If None, auto-computed to preserve
                pixel count.
            shape: Target grid shape as ``(rows, cols)``. Mutually exclusive
                with *resolution*.
            resampling: Resampling method (``"nearest"``, ``"bilinear"``,
                ``"cubic"``, ``"lanczos"``, ``"average"``).
            nodata: Nodata value. Defaults to NaN for floats.
            dst_chunks: Chunk size for dask output.
            batch_size: Number of N-D slices to reproject per dask task.
                If None, derived from *max_task_bytes*.
            max_task_bytes: Maximum bytes per dask task when auto-computing
                *batch_size* (default 256 MiB).

        Returns:
            Reprojected DataArray with updated coordinates and CRS.
        """
        src_geobox = self.geobox
        dst_geobox = _compute_dst_geobox(src_geobox, dst_crs, resolution, shape)
        return _reproject_dataarray(
            self._da,
            src_geobox,
            dst_geobox,
            resampling=resampling,
            nodata=nodata,
            dst_chunks=dst_chunks,
            batch_size=batch_size,
            max_task_bytes=max_task_bytes,
        )

    def reproject_match(
        self,
        other,
        resampling: str = "bilinear",
        nodata: float | None = None,
        dst_chunks: tuple[int, int] | None = None,
        batch_size: int | None = None,
        max_task_bytes: int = 256 * 1024 * 1024,
    ) -> xr.DataArray:
        """Reproject to match another grid.

        Args:
            other: Target grid — a rust-warp GeoBox, odc-geo GeoBox, or
                xarray DataArray/Dataset.
            resampling: Resampling method.
            nodata: Nodata value.
            dst_chunks: Chunk size for dask output.
            batch_size: Number of N-D slices to reproject per dask task.
            max_task_bytes: Maximum bytes per dask task (default 256 MiB).

        Returns:
            Reprojected DataArray aligned to the target grid.
        """
        src_geobox = self.geobox
        dst_geobox = _as_geobox(other)
        return _reproject_dataarray(
            self._da,
            src_geobox,
            dst_geobox,
            resampling=resampling,
            nodata=nodata,
            dst_chunks=dst_chunks,
            batch_size=batch_size,
            max_task_bytes=max_task_bytes,
        )


@xr.register_dataset_accessor("warp")
class WarpDatasetAccessor:
    """Raster reprojection accessor for xarray Dataset."""

    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    @property
    def geobox(self) -> GeoBox:
        """GeoBox derived from this Dataset's coordinates and CRS."""
        return GeoBox.from_xarray(self._ds)

    @property
    def crs(self) -> str | None:
        """CRS string, or None if not set."""
        try:
            return _extract_crs(self._ds)
        except ValueError:
            return None

    def reproject(
        self,
        dst_crs: str,
        resolution: float | tuple[float, float] | None = None,
        shape: tuple[int, int] | None = None,
        resampling: str = "bilinear",
        nodata: float | None = None,
        dst_chunks: tuple[int, int] | None = None,
        batch_size: int | None = None,
        max_task_bytes: int = 256 * 1024 * 1024,
    ) -> xr.Dataset:
        """Reproject all spatial variables to a new CRS.

        Non-spatial variables are passed through unchanged.

        Args:
            dst_crs: Target CRS string.
            resolution: Target pixel size.
            shape: Target grid shape.
            resampling: Resampling method.
            nodata: Nodata value.
            dst_chunks: Chunk size for dask output.
            batch_size: Number of same-dtype 2-D variables to reproject per
                dask task. If None, derived from *max_task_bytes*. N-D
                variables (e.g. with a time dimension) use *batch_size* as
                the number of non-spatial slices per task instead.
            max_task_bytes: Maximum bytes per dask task (default 256 MiB).

        Returns:
            Reprojected Dataset.
        """
        src_geobox = self.geobox
        dst_geobox = _compute_dst_geobox(src_geobox, dst_crs, resolution, shape)
        return self._reproject_with_geobox(
            src_geobox,
            dst_geobox,
            resampling,
            nodata,
            dst_chunks,
            batch_size,
            max_task_bytes,
        )

    def reproject_match(
        self,
        other,
        resampling: str = "bilinear",
        nodata: float | None = None,
        dst_chunks: tuple[int, int] | None = None,
        batch_size: int | None = None,
        max_task_bytes: int = 256 * 1024 * 1024,
    ) -> xr.Dataset:
        """Reproject to match another grid.

        Args:
            other: Target grid — a rust-warp GeoBox, odc-geo GeoBox, or
                xarray DataArray/Dataset.
            resampling: Resampling method.
            nodata: Nodata value.
            dst_chunks: Chunk size for dask output.
            batch_size: Number of same-dtype 2-D variables to reproject per
                dask task.
            max_task_bytes: Maximum bytes per dask task (default 256 MiB).

        Returns:
            Reprojected Dataset aligned to the target grid.
        """
        src_geobox = self.geobox
        dst_geobox = _as_geobox(other)
        return self._reproject_with_geobox(
            src_geobox,
            dst_geobox,
            resampling,
            nodata,
            dst_chunks,
            batch_size,
            max_task_bytes,
        )

    def _reproject_with_geobox(
        self,
        src_geobox: GeoBox,
        dst_geobox: GeoBox,
        resampling: str,
        nodata: float | None,
        dst_chunks: tuple[int, int] | None,
        batch_size: int | None,
        max_task_bytes: int,
    ) -> xr.Dataset:
        """Reproject all spatial variables using the given geoboxes."""
        try:
            y_dim, x_dim = _find_spatial_dims(self._ds)
        except ValueError:
            raise ValueError("Cannot determine spatial dimensions in Dataset") from None

        # Separate spatial variables from pass-through variables
        spatial_vars = {
            n: v for n, v in self._ds.data_vars.items() if y_dim in v.dims and x_dim in v.dims
        }
        new_vars = {n: v for n, v in self._ds.data_vars.items() if n not in spatial_vars}

        # Detect chunked backend from spatial variables
        backends = {detect_backend(v.data) for v in spatial_vars.values()}
        backends.discard(None)
        backend = backends.pop() if backends else None

        if backend is None:
            # Numpy path: process each variable individually
            for vname, var in spatial_vars.items():
                new_vars[vname] = _reproject_dataarray(
                    var,
                    src_geobox,
                    dst_geobox,
                    resampling=resampling,
                    nodata=nodata,
                    dst_chunks=dst_chunks,
                    batch_size=batch_size,
                    max_task_bytes=max_task_bytes,
                )
        else:
            if backend == "dask":
                from rust_warp.dask_graph import reproject_dask as reproject_chunked
            else:
                from rust_warp.cubed_graph import reproject_cubed as reproject_chunked

            # Group pure-2D, same-dtype chunked variables for batch processing.
            # N-D variables are handled individually via _reproject_dataarray.
            pure_2d_by_dtype: dict[np.dtype, list[str]] = {}
            nd_var_names = []

            for vname, var in spatial_vars.items():
                non_sp = [d for d in var.dims if d not in (y_dim, x_dim)]
                if not non_sp and detect_backend(var.data) is not None:
                    pure_2d_by_dtype.setdefault(var.dtype, []).append(vname)
                else:
                    nd_var_names.append(vname)

            # Process N-D variables individually (they use the improved _reproject_nd)
            for vname in nd_var_names:
                new_vars[vname] = _reproject_dataarray(
                    spatial_vars[vname],
                    src_geobox,
                    dst_geobox,
                    resampling=resampling,
                    nodata=nodata,
                    dst_chunks=dst_chunks,
                    batch_size=batch_size,
                    max_task_bytes=max_task_bytes,
                )

            # Process each dtype group as a single batched 3D reproject
            for grp_dtype, var_names in pure_2d_by_dtype.items():
                if len(var_names) == 1:
                    new_vars[var_names[0]] = _reproject_dataarray(
                        spatial_vars[var_names[0]],
                        src_geobox,
                        dst_geobox,
                        resampling=resampling,
                        nodata=nodata,
                        dst_chunks=dst_chunks,
                        batch_size=batch_size,
                        max_task_bytes=max_task_bytes,
                    )
                    continue

                # Use dst_chunks if explicit; otherwise proxy with src chunk size
                first_var = spatial_vars[var_names[0]]
                y_pos = first_var.dims.index(y_dim)
                x_pos = first_var.dims.index(x_dim)
                if dst_chunks is not None:
                    sp_chunks = dst_chunks
                else:
                    sp_chunks = (
                        first_var.data.chunks[y_pos][0],
                        first_var.data.chunks[x_pos][0],
                    )

                actual_batch = _derive_batch_size(
                    np.dtype(grp_dtype), sp_chunks, batch_size, max_task_bytes
                )
                actual_batch = min(actual_batch, len(var_names))

                # Stack variables into (n_vars, y, x) and rechunk batch dim
                stacked = stack_arrays(
                    [spatial_vars[n].data for n in var_names], axis=0, backend=backend
                )
                stacked = stacked.rechunk({0: actual_batch})

                # One reproject call covers all variables in this dtype group
                reprojected_3d = reproject_chunked(
                    stacked,
                    src_geobox,
                    dst_geobox,
                    resampling=resampling,
                    dst_chunks=dst_chunks,
                    nodata=nodata,
                )

                dst_coords = dst_geobox.xr_coords()

                for i, vname in enumerate(var_names):
                    orig = spatial_vars[vname]
                    result_2d = reprojected_3d[i]  # lazy 2D chunked array

                    new_coords: dict = {
                        y_dim: (y_dim, dst_coords["y"]),
                        x_dim: (x_dim, dst_coords["x"]),
                        "spatial_ref": _make_spatial_ref_coord(dst_geobox.crs),
                    }
                    for cname, coord in orig.coords.items():
                        if cname not in (y_dim, x_dim, "spatial_ref") and not set(coord.dims) & {
                            y_dim,
                            x_dim,
                        }:
                            new_coords[cname] = coord

                    attrs = {k: v for k, v in orig.attrs.items() if k not in ("crs", "transform")}
                    new_vars[vname] = xr.DataArray(
                        data=result_2d,
                        dims=[y_dim, x_dim],
                        coords=new_coords,
                        name=vname,
                        attrs=attrs,
                    )

        # Build the dataset from reprojected variables
        ds = xr.Dataset(new_vars)

        # Ensure spatial_ref is set
        if "spatial_ref" not in ds.coords:
            ds = ds.assign_coords(spatial_ref=_make_spatial_ref_coord(dst_geobox.crs))

        # Preserve dataset-level attrs, filtering stale geo metadata
        attrs = {k: v for k, v in self._ds.attrs.items() if k not in ("crs", "transform")}
        ds.attrs.update(attrs)

        return ds
