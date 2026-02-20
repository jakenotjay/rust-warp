"""Xarray accessor and GeoBox.from_xarray() tests."""

from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from rust_warp import GeoBox  # noqa: E402
from rust_warp.geobox import _extract_crs, _find_spatial_dims  # noqa: E402
from rust_warp.xarray_accessor import (  # noqa: E402
    WarpDataArrayAccessor,
    WarpDatasetAccessor,
    _as_geobox,
    _compute_dst_geobox,
    _make_spatial_ref_coord,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SRC_CRS = "EPSG:32633"
SRC_TRANSFORM = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0)
SRC_SHAPE = (64, 64)
SRC_GEOBOX = GeoBox(crs=SRC_CRS, shape=SRC_SHAPE, affine=SRC_TRANSFORM)


def _make_dataarray(
    geobox: GeoBox,
    y_name: str = "y",
    x_name: str = "x",
    crs_mode: str = "spatial_ref",
    extra_dims: dict | None = None,
    dtype=np.float64,
) -> xr.DataArray:
    """Build a synthetic DataArray from a GeoBox.

    Args:
        crs_mode: "spatial_ref" | "attrs" | "none"
        extra_dims: e.g. {"band": 3} to prepend a band dim.
    """
    coords_dict = geobox.xr_coords()
    y_vals = coords_dict["y"]
    x_vals = coords_dict["x"]

    rows, cols = geobox.shape
    shape = (rows, cols)
    dims = [y_name, x_name]

    if extra_dims:
        for dname, dsize in reversed(extra_dims.items()):
            shape = (dsize, *shape)
            dims = [dname, *dims]

    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

    coords: dict = {
        y_name: (y_name, y_vals),
        x_name: (x_name, x_vals),
    }
    if extra_dims:
        for dname, dsize in extra_dims.items():
            coords[dname] = (dname, np.arange(dsize))

    if crs_mode == "spatial_ref":
        coords["spatial_ref"] = _make_spatial_ref_coord(geobox.crs)
    da = xr.DataArray(data, dims=dims, coords=coords, name="test_var")
    if crs_mode == "attrs":
        da.attrs["crs"] = geobox.crs
    return da


def _make_dataset(geobox: GeoBox, n_vars: int = 2) -> xr.Dataset:
    """Build a synthetic Dataset with *n_vars* spatial variables."""
    coords_dict = geobox.xr_coords()
    rows, cols = geobox.shape
    ds_vars = {}
    for i in range(n_vars):
        data = np.random.default_rng(i).random((rows, cols))
        ds_vars[f"var{i}"] = (["y", "x"], data)
    coords = {
        "y": ("y", coords_dict["y"]),
        "x": ("x", coords_dict["x"]),
        "spatial_ref": _make_spatial_ref_coord(geobox.crs),
    }
    return xr.Dataset(ds_vars, coords=coords)


# ===========================================================================
# TestGeoBoxFromXarray
# ===========================================================================


class TestGeoBoxFromXarray:
    """GeoBox.from_xarray() classmethod."""

    def test_round_trip(self):
        """geobox → xr coords → from_xarray should round-trip."""
        da = _make_dataarray(SRC_GEOBOX)
        recovered = GeoBox.from_xarray(da)
        assert recovered.shape == SRC_GEOBOX.shape
        assert recovered.crs == SRC_GEOBOX.crs
        for a, b in zip(recovered.affine, SRC_GEOBOX.affine, strict=True):
            assert a == pytest.approx(b, abs=1e-10)

    def test_spatial_ref_coord(self):
        """CRS is read from spatial_ref coordinate."""
        da = _make_dataarray(SRC_GEOBOX, crs_mode="spatial_ref")
        assert GeoBox.from_xarray(da).crs == SRC_CRS

    def test_attrs_crs(self):
        """CRS is read from attrs['crs'] fallback."""
        da = _make_dataarray(SRC_GEOBOX, crs_mode="attrs")
        assert GeoBox.from_xarray(da).crs == SRC_CRS

    def test_lat_lon_dims(self):
        """Works with latitude/longitude dim names."""
        da = _make_dataarray(SRC_GEOBOX, y_name="latitude", x_name="longitude")
        gbox = GeoBox.from_xarray(da)
        assert gbox.shape == SRC_GEOBOX.shape

    def test_lat_lon_short_dims(self):
        """Works with lat/lon dim names."""
        da = _make_dataarray(SRC_GEOBOX, y_name="lat", x_name="lon")
        gbox = GeoBox.from_xarray(da)
        assert gbox.shape == SRC_GEOBOX.shape

    def test_missing_crs_raises(self):
        """Raises ValueError when no CRS can be determined."""
        da = _make_dataarray(SRC_GEOBOX, crs_mode="none")
        with pytest.raises(ValueError, match="Cannot determine CRS"):
            GeoBox.from_xarray(da)

    def test_missing_spatial_dims_raises(self):
        """Raises ValueError for a 1D DataArray."""
        da = xr.DataArray(np.arange(10), dims=["z"])
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            GeoBox.from_xarray(da)

    def test_from_dataset(self):
        """from_xarray works with a Dataset too."""
        ds = _make_dataset(SRC_GEOBOX)
        gbox = GeoBox.from_xarray(ds)
        assert gbox.shape == SRC_GEOBOX.shape
        assert gbox.crs == SRC_CRS


# ===========================================================================
# TestComputeDstGeobox
# ===========================================================================


class TestComputeDstGeobox:
    """_compute_dst_geobox helper."""

    def test_explicit_resolution(self):
        dst = _compute_dst_geobox(SRC_GEOBOX, "EPSG:4326", resolution=0.001)
        assert dst.crs == "EPSG:4326"
        assert dst.resolution[0] == pytest.approx(0.001, abs=1e-6)
        assert dst.shape[0] > 0
        assert dst.shape[1] > 0

    def test_explicit_shape(self):
        dst = _compute_dst_geobox(SRC_GEOBOX, "EPSG:4326", shape=(100, 100))
        assert dst.crs == "EPSG:4326"
        assert dst.shape == (100, 100)

    def test_auto_resolution(self):
        """Without explicit resolution/shape, pixel count is preserved."""
        dst = _compute_dst_geobox(SRC_GEOBOX, "EPSG:4326")
        assert dst.crs == "EPSG:4326"
        src_pixels = SRC_GEOBOX.shape[0] * SRC_GEOBOX.shape[1]
        dst_pixels = dst.shape[0] * dst.shape[1]
        # Should be within ~20% of source pixel count
        assert abs(dst_pixels - src_pixels) / src_pixels < 0.2

    def test_same_crs(self):
        """Same CRS produces a similar bounding box."""
        dst = _compute_dst_geobox(SRC_GEOBOX, SRC_CRS)
        src_left, _src_bot, _src_right, src_top = SRC_GEOBOX.bounds
        dst_left, _dst_bot, _dst_right, dst_top = dst.bounds
        assert dst_left == pytest.approx(src_left, rel=0.01)
        assert dst_top == pytest.approx(src_top, rel=0.01)


# ===========================================================================
# TestDataArrayAccessor
# ===========================================================================


class TestDataArrayAccessor:
    """WarpDataArrayAccessor (.warp on DataArray)."""

    def test_accessor_exists(self):
        da = _make_dataarray(SRC_GEOBOX)
        assert hasattr(da, "warp")
        assert isinstance(da.warp, WarpDataArrayAccessor)

    def test_geobox_property(self):
        da = _make_dataarray(SRC_GEOBOX)
        gbox = da.warp.geobox
        assert gbox.shape == SRC_GEOBOX.shape
        assert gbox.crs == SRC_CRS

    def test_crs_property(self):
        da = _make_dataarray(SRC_GEOBOX)
        assert da.warp.crs == SRC_CRS

    def test_crs_none_when_missing(self):
        da = _make_dataarray(SRC_GEOBOX, crs_mode="none")
        assert da.warp.crs is None

    def test_same_crs_reproject(self):
        """Reprojecting to the same CRS should return a valid result."""
        da = _make_dataarray(SRC_GEOBOX)
        result = da.warp.reproject(SRC_CRS, resampling="nearest")
        assert isinstance(result, xr.DataArray)
        assert result.warp.crs is not None

    def test_cross_crs_reproject(self):
        da = _make_dataarray(SRC_GEOBOX)
        result = da.warp.reproject("EPSG:4326", resolution=0.001, resampling="nearest")
        assert isinstance(result, xr.DataArray)
        assert result.shape[0] > 0
        assert result.shape[1] > 0
        # Should have spatial_ref coord
        assert "spatial_ref" in result.coords

    def test_output_coords_match_geobox(self):
        da = _make_dataarray(SRC_GEOBOX)
        result = da.warp.reproject("EPSG:4326", shape=(32, 32), resampling="nearest")
        gbox = result.warp.geobox
        assert gbox.shape == (32, 32)
        assert gbox.crs is not None

    def test_name_preserved(self):
        da = _make_dataarray(SRC_GEOBOX)
        da.name = "elevation"
        result = da.warp.reproject(SRC_CRS, resampling="nearest")
        assert result.name == "elevation"

    def test_attrs_preserved(self):
        da = _make_dataarray(SRC_GEOBOX)
        da.attrs["units"] = "meters"
        da.attrs["crs"] = "stale"
        result = da.warp.reproject(SRC_CRS, resampling="nearest")
        assert result.attrs["units"] == "meters"
        assert "crs" not in result.attrs  # stale geo attr removed

    def test_reproject_match(self):
        src = _make_dataarray(SRC_GEOBOX)
        dst_geobox = GeoBox.from_bbox(
            (500000.0, 6593600.0, 506400.0, 6600000.0),
            crs="EPSG:32633",
            shape=(32, 32),
        )
        dst = _make_dataarray(dst_geobox)
        result = src.warp.reproject_match(dst, resampling="nearest")
        assert result.shape == (32, 32)


# ===========================================================================
# TestDataArrayAccessorMultiDim
# ===========================================================================


class TestDataArrayAccessorMultiDim:
    """Multi-dimensional DataArray reprojection."""

    def test_band_y_x(self):
        da = _make_dataarray(SRC_GEOBOX, extra_dims={"band": 3})
        assert da.shape == (3, 64, 64)
        result = da.warp.reproject(SRC_CRS, resampling="nearest")
        assert result.ndim == 3
        assert result.dims[0] == "band"
        assert result.shape[0] == 3

    def test_time_y_x(self):
        da = _make_dataarray(SRC_GEOBOX, extra_dims={"time": 4})
        result = da.warp.reproject(SRC_CRS, resampling="nearest")
        assert result.ndim == 3
        assert result.dims[0] == "time"
        assert result.shape[0] == 4

    def test_time_band_y_x(self):
        da = _make_dataarray(SRC_GEOBOX, extra_dims={"time": 2, "band": 3})
        assert da.shape == (2, 3, 64, 64)
        result = da.warp.reproject(SRC_CRS, resampling="nearest")
        assert result.ndim == 4
        assert result.shape[0] == 2
        assert result.shape[1] == 3

    def test_non_spatial_coords_preserved(self):
        da = _make_dataarray(SRC_GEOBOX, extra_dims={"time": 3})
        result = da.warp.reproject(SRC_CRS, resampling="nearest")
        assert "time" in result.coords
        np.testing.assert_array_equal(result.coords["time"].values, [0, 1, 2])


# ===========================================================================
# TestDataArrayAccessorDask
# ===========================================================================


class TestDataArrayAccessorDask:
    """Dask-backed DataArray reprojection."""

    @pytest.fixture(autouse=True)
    def _skip_no_dask(self):
        pytest.importorskip("dask.array")

    def test_dask_in_dask_out(self):
        da = _make_dataarray(SRC_GEOBOX)
        da_chunked = da.chunk({"y": 32, "x": 32})
        result = da_chunked.warp.reproject(SRC_CRS, resampling="nearest")
        assert hasattr(result.data, "dask"), "Expected dask output"

    def test_dask_values_match_numpy(self):
        da = _make_dataarray(SRC_GEOBOX)
        da_chunked = da.chunk({"y": 32, "x": 32})
        result_dask = da_chunked.warp.reproject(SRC_CRS, resampling="nearest")
        result_numpy = da.warp.reproject(SRC_CRS, resampling="nearest")

        dask_vals = result_dask.values
        numpy_vals = result_numpy.values

        # NaN patterns may differ slightly at chunk boundaries
        dask_nan = np.isnan(dask_vals)
        numpy_nan = np.isnan(numpy_vals)
        both_valid = ~dask_nan & ~numpy_nan
        if both_valid.any():
            np.testing.assert_allclose(
                dask_vals[both_valid],
                numpy_vals[both_valid],
                atol=1e-10,
            )

    def test_coords_correct(self):
        da = _make_dataarray(SRC_GEOBOX)
        da_chunked = da.chunk({"y": 32, "x": 32})
        result = da_chunked.warp.reproject(SRC_CRS, resampling="nearest")
        assert "y" in result.coords
        assert "x" in result.coords
        assert "spatial_ref" in result.coords

    def test_dst_chunks_param(self):
        da = _make_dataarray(SRC_GEOBOX)
        da_chunked = da.chunk({"y": 32, "x": 32})
        result = da_chunked.warp.reproject(
            SRC_CRS,
            resampling="nearest",
            dst_chunks=(16, 16),
        )
        assert hasattr(result.data, "dask")


# ===========================================================================
# TestDatasetAccessor
# ===========================================================================


class TestDatasetAccessor:
    """WarpDatasetAccessor (.warp on Dataset)."""

    def test_accessor_exists(self):
        ds = _make_dataset(SRC_GEOBOX)
        assert hasattr(ds, "warp")
        assert isinstance(ds.warp, WarpDatasetAccessor)

    def test_all_vars_reprojected(self):
        ds = _make_dataset(SRC_GEOBOX, n_vars=2)
        result = ds.warp.reproject(SRC_CRS, resampling="nearest")
        assert isinstance(result, xr.Dataset)
        assert "var0" in result.data_vars
        assert "var1" in result.data_vars

    def test_non_spatial_vars_passthrough(self):
        ds = _make_dataset(SRC_GEOBOX, n_vars=1)
        ds["meta"] = xr.DataArray(42)  # scalar, non-spatial
        result = ds.warp.reproject(SRC_CRS, resampling="nearest")
        assert "meta" in result.data_vars
        assert int(result["meta"]) == 42

    def test_reproject_match(self):
        ds = _make_dataset(SRC_GEOBOX, n_vars=1)
        dst_geobox = GeoBox.from_bbox(
            (500000.0, 6593600.0, 506400.0, 6600000.0),
            crs="EPSG:32633",
            shape=(32, 32),
        )
        dst = _make_dataarray(dst_geobox)
        result = ds.warp.reproject_match(dst, resampling="nearest")
        assert result["var0"].shape == (32, 32)

    def test_geobox_property(self):
        ds = _make_dataset(SRC_GEOBOX)
        gbox = ds.warp.geobox
        assert gbox.shape == SRC_GEOBOX.shape

    def test_crs_property(self):
        ds = _make_dataset(SRC_GEOBOX)
        assert ds.warp.crs == SRC_CRS


# ===========================================================================
# TestOdcGeoCompat
# ===========================================================================


class TestOdcGeoCompat:
    """odc-geo GeoBox compatibility via from_odc() and _as_geobox()."""

    def _make_mock_odc_geobox(self, geobox: GeoBox):
        """Create a mock odc.geo GeoBox-like object."""
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.crs = geobox.crs
        shape_mock = MagicMock()
        shape_mock.y = geobox.shape[0]
        shape_mock.x = geobox.shape[1]
        mock.shape = shape_mock
        affine_mock = MagicMock()
        a, b, c, d, e, f = geobox.affine
        affine_mock.a = a
        affine_mock.b = b
        affine_mock.c = c
        affine_mock.d = d
        affine_mock.e = e
        affine_mock.f = f
        mock.affine = affine_mock
        return mock

    def test_from_odc_round_trip(self):
        mock_odc = self._make_mock_odc_geobox(SRC_GEOBOX)
        result = GeoBox.from_odc(mock_odc)
        assert result.crs == SRC_CRS
        assert result.shape == SRC_GEOBOX.shape
        for a, b in zip(result.affine, SRC_GEOBOX.affine, strict=True):
            assert a == pytest.approx(b)

    def test_as_geobox_with_geobox(self):
        result = _as_geobox(SRC_GEOBOX)
        assert result is SRC_GEOBOX

    def test_as_geobox_duck_type(self):
        mock_odc = self._make_mock_odc_geobox(SRC_GEOBOX)
        result = _as_geobox(mock_odc)
        assert isinstance(result, GeoBox)
        assert result.shape == SRC_GEOBOX.shape

    def test_as_geobox_xarray(self):
        da = _make_dataarray(SRC_GEOBOX)
        result = _as_geobox(da)
        assert isinstance(result, GeoBox)
        assert result.shape == SRC_GEOBOX.shape

    def test_reproject_match_with_odc_geobox(self):
        src = _make_dataarray(SRC_GEOBOX)
        mock_odc = self._make_mock_odc_geobox(
            GeoBox.from_bbox(
                (500000.0, 6593600.0, 506400.0, 6600000.0),
                crs="EPSG:32633",
                shape=(32, 32),
            )
        )
        result = src.warp.reproject_match(mock_odc, resampling="nearest")
        assert result.shape == (32, 32)


# ===========================================================================
# TestHelpers
# ===========================================================================


class TestHelpers:
    """Standalone helper function tests."""

    def test_find_spatial_dims_y_x(self):
        da = xr.DataArray(np.zeros((4, 4)), dims=["y", "x"])
        assert _find_spatial_dims(da) == ("y", "x")

    def test_find_spatial_dims_lat_lon(self):
        da = xr.DataArray(np.zeros((4, 4)), dims=["latitude", "longitude"])
        assert _find_spatial_dims(da) == ("latitude", "longitude")

    def test_find_spatial_dims_fallback(self):
        da = xr.DataArray(np.zeros((3, 4, 5)), dims=["time", "row", "col"])
        assert _find_spatial_dims(da) == ("row", "col")

    def test_extract_crs_spatial_ref(self):
        da = xr.DataArray(np.zeros((4, 4)), dims=["y", "x"])
        da = da.assign_coords(spatial_ref=_make_spatial_ref_coord("EPSG:4326"))
        crs = _extract_crs(da)
        assert crs is not None

    def test_extract_crs_attrs(self):
        da = xr.DataArray(np.zeros((4, 4)), dims=["y", "x"])
        da.attrs["crs"] = "EPSG:4326"
        assert _extract_crs(da) == "EPSG:4326"

    def test_extract_crs_raises(self):
        da = xr.DataArray(np.zeros((4, 4)), dims=["y", "x"])
        with pytest.raises(ValueError, match="Cannot determine CRS"):
            _extract_crs(da)

    def test_make_spatial_ref_coord(self):
        coord = _make_spatial_ref_coord("EPSG:4326")
        assert "crs_wkt" in coord.attrs
