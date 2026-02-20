"""GeoBox operations tests.

Tests GeoBox construction, property computation, coordinate generation,
slicing, resolution handling, and various geometric operations.
"""

import numpy as np
import pytest
from rust_warp.geobox import GeoBox


class TestGeoBoxFromBbox:
    """GeoBox.from_bbox construction edge cases."""

    def test_from_bbox_with_scalar_resolution(self):
        """Scalar resolution creates square pixels."""
        gbox = GeoBox.from_bbox((0, 0, 100, 100), "EPSG:32633", resolution=10.0)
        assert gbox.shape == (10, 10)
        assert gbox.resolution == (10.0, 10.0)

    def test_from_bbox_with_tuple_resolution(self):
        """Tuple resolution creates non-square pixels."""
        gbox = GeoBox.from_bbox((0, 0, 100, 200), "EPSG:32633", resolution=(10.0, 20.0))
        assert gbox.shape == (10, 10)
        assert gbox.resolution == (10.0, 20.0)

    def test_from_bbox_with_shape(self):
        """Shape-based construction should compute resolution."""
        gbox = GeoBox.from_bbox((0, 0, 100, 200), "EPSG:32633", shape=(20, 10))
        assert gbox.shape == (20, 10)
        assert gbox.resolution == (10.0, 10.0)

    def test_from_bbox_no_resolution_or_shape_raises(self):
        """Must provide either resolution or shape."""
        with pytest.raises(ValueError, match="resolution or shape"):
            GeoBox.from_bbox((0, 0, 100, 100), "EPSG:32633")

    def test_from_bbox_affine_origin(self):
        """Affine origin should be at top-left of bbox."""
        gbox = GeoBox.from_bbox((100, 200, 300, 400), "EPSG:32633", resolution=10.0)
        # c = left = 100, f = top = 400
        assert gbox.affine[2] == 100.0  # origin x = left
        assert gbox.affine[5] == 400.0  # origin y = top

    def test_from_bbox_negative_coords(self):
        """Negative coordinates should work fine."""
        gbox = GeoBox.from_bbox((-180, -90, 180, 90), "EPSG:4326", resolution=1.0)
        assert gbox.shape == (180, 360)
        assert gbox.affine[2] == -180.0
        assert gbox.affine[5] == 90.0

    def test_from_bbox_tiny_bbox(self):
        """Very small bbox should produce at least 1x1 grid."""
        gbox = GeoBox.from_bbox((0, 0, 0.001, 0.001), "EPSG:4326", resolution=0.01)
        assert gbox.shape[0] >= 1
        assert gbox.shape[1] >= 1

    def test_from_bbox_large_bbox(self):
        """Large bbox with fine resolution."""
        gbox = GeoBox.from_bbox(
            (100000, 5000000, 900000, 6000000),
            "EPSG:32633",
            resolution=100.0,
        )
        assert gbox.shape == (10000, 8000)


class TestGeoBoxBounds:
    """GeoBox.bounds property."""

    def test_bounds_roundtrip(self):
        """from_bbox bounds should match input bbox."""
        bbox = (500000.0, 6500000.0, 600000.0, 6600000.0)
        gbox = GeoBox.from_bbox(bbox, "EPSG:32633", resolution=100.0)
        bounds = gbox.bounds
        np.testing.assert_allclose(bounds[0], bbox[0], atol=1.0)  # left
        np.testing.assert_allclose(bounds[2], bbox[2], atol=1.0)  # right
        np.testing.assert_allclose(bounds[1], bbox[1], atol=1.0)  # bottom
        np.testing.assert_allclose(bounds[3], bbox[3], atol=1.0)  # top

    def test_bounds_with_nonsquare_pixels(self):
        """Bounds should be correct with non-square pixels."""
        gbox = GeoBox.from_bbox((0, 0, 200, 100), "EPSG:32633", shape=(10, 20))
        bounds = gbox.bounds
        np.testing.assert_allclose(bounds, (0.0, 0.0, 200.0, 100.0), atol=1e-10)

    def test_bounds_single_pixel(self):
        """1x1 GeoBox should have correct bounds."""
        gbox = GeoBox(
            crs="EPSG:32633",
            shape=(1, 1),
            affine=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600100.0),
        )
        left, bottom, right, top = gbox.bounds
        assert left == 500000.0
        assert right == 500100.0
        assert top == 6600100.0
        assert bottom == 6600000.0


class TestGeoBoxResolution:
    """GeoBox.resolution property."""

    def test_resolution_always_positive(self):
        """Resolution should be positive even with negative affine[4]."""
        gbox = GeoBox(
            crs="EPSG:32633",
            shape=(10, 10),
            affine=(100.0, 0.0, 0.0, 0.0, -100.0, 1000.0),
        )
        res_x, res_y = gbox.resolution
        assert res_x > 0
        assert res_y > 0

    def test_resolution_nonsquare(self):
        """Non-square resolution should report different x and y."""
        gbox = GeoBox(
            crs="EPSG:32633",
            shape=(10, 10),
            affine=(50.0, 0.0, 0.0, 0.0, -100.0, 1000.0),
        )
        assert gbox.resolution == (50.0, 100.0)


class TestGeoBoxXrCoords:
    """GeoBox.xr_coords() coordinate generation."""

    def test_coords_shape(self):
        """x and y coord arrays should match shape dimensions."""
        gbox = GeoBox.from_bbox((0, 0, 100, 200), "EPSG:32633", shape=(20, 10))
        coords = gbox.xr_coords()
        assert len(coords["x"]) == 10  # cols
        assert len(coords["y"]) == 20  # rows

    def test_coords_are_pixel_centers(self):
        """Coordinates should be at pixel centers (half-pixel offset from origin)."""
        gbox = GeoBox(
            crs="EPSG:32633",
            shape=(4, 4),
            affine=(10.0, 0.0, 100.0, 0.0, -10.0, 140.0),
        )
        coords = gbox.xr_coords()

        # x: origin=100, res=10 → centers at 105, 115, 125, 135
        np.testing.assert_allclose(coords["x"], [105.0, 115.0, 125.0, 135.0])
        # y: origin=140, res=-10 → centers at 135, 125, 115, 105
        np.testing.assert_allclose(coords["y"], [135.0, 125.0, 115.0, 105.0])

    def test_coords_monotonic(self):
        """x should be increasing, y should be decreasing (north-up)."""
        gbox = GeoBox.from_bbox((0, 0, 100, 100), "EPSG:32633", resolution=10.0)
        coords = gbox.xr_coords()
        assert np.all(np.diff(coords["x"]) > 0), "x not monotonically increasing"
        assert np.all(np.diff(coords["y"]) < 0), "y not monotonically decreasing"

    def test_coords_roundtrip_with_from_xarray(self):
        """GeoBox → xr_coords → from_xarray should round-trip."""
        import xarray as xr

        original = GeoBox.from_bbox(
            (500000, 6500000, 510000, 6510000),
            "EPSG:32633",
            resolution=100.0,
        )
        coords = original.xr_coords()

        # Build a DataArray with these coords
        da = xr.DataArray(
            np.zeros(original.shape),
            dims=["y", "x"],
            coords={
                "y": coords["y"],
                "x": coords["x"],
                "spatial_ref": xr.Variable((), 0, attrs={"epsg": 32633}),
            },
        )

        recovered = GeoBox.from_xarray(da)
        assert recovered.shape == original.shape
        assert recovered.crs == original.crs
        np.testing.assert_allclose(recovered.affine[0], original.affine[0], atol=1e-6)
        np.testing.assert_allclose(recovered.affine[4], original.affine[4], atol=1e-6)


class TestGeoBoxFromXarrayEdgeCases:
    """Edge cases for GeoBox.from_xarray."""

    def test_missing_crs_raises(self):
        """from_xarray with no CRS should raise ValueError."""
        import xarray as xr

        da = xr.DataArray(
            np.zeros((10, 10)),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        with pytest.raises(ValueError, match=r"[Cc]RS"):
            GeoBox.from_xarray(da)

    def test_1d_array_raises(self):
        """1D array should raise ValueError."""
        import xarray as xr

        da = xr.DataArray(np.arange(10), dims=["x"])
        with pytest.raises(ValueError, match="at least 2"):
            GeoBox.from_xarray(da)

    def test_lat_lon_dim_names(self):
        """latitude/longitude dimension names should be detected."""
        import xarray as xr

        da = xr.DataArray(
            np.zeros((10, 20)),
            dims=["latitude", "longitude"],
            coords={
                "latitude": np.linspace(60, 50, 10),
                "longitude": np.linspace(10, 12, 20),
                "spatial_ref": xr.Variable((), 0, attrs={"epsg": 4326}),
            },
        )
        gbox = GeoBox.from_xarray(da)
        assert gbox.shape == (10, 20)
        assert gbox.crs == "EPSG:4326"

    def test_lat_lon_short_names(self):
        """lat/lon dimension names should be detected."""
        import xarray as xr

        da = xr.DataArray(
            np.zeros((10, 20)),
            dims=["lat", "lon"],
            coords={
                "lat": np.linspace(60, 50, 10),
                "lon": np.linspace(10, 12, 20),
                "spatial_ref": xr.Variable((), 0, attrs={"epsg": 4326}),
            },
        )
        gbox = GeoBox.from_xarray(da)
        assert gbox.shape == (10, 20)


class TestGeoBoxEquality:
    """GeoBox equality and hashing (frozen dataclass)."""

    def test_identical_geoboxes_equal(self):
        """Two identical GeoBoxes should be equal."""
        a = GeoBox("EPSG:32633", (10, 10), (100.0, 0.0, 0.0, 0.0, -100.0, 1000.0))
        b = GeoBox("EPSG:32633", (10, 10), (100.0, 0.0, 0.0, 0.0, -100.0, 1000.0))
        assert a == b

    def test_different_crs_not_equal(self):
        """Different CRS should not be equal."""
        a = GeoBox("EPSG:32633", (10, 10), (100.0, 0.0, 0.0, 0.0, -100.0, 1000.0))
        b = GeoBox("EPSG:4326", (10, 10), (100.0, 0.0, 0.0, 0.0, -100.0, 1000.0))
        assert a != b

    def test_different_shape_not_equal(self):
        """Different shape should not be equal."""
        a = GeoBox("EPSG:32633", (10, 10), (100.0, 0.0, 0.0, 0.0, -100.0, 1000.0))
        b = GeoBox("EPSG:32633", (20, 10), (100.0, 0.0, 0.0, 0.0, -100.0, 1000.0))
        assert a != b

    def test_hashable(self):
        """Frozen GeoBox should be hashable (usable as dict key)."""
        gbox = GeoBox("EPSG:32633", (10, 10), (100.0, 0.0, 0.0, 0.0, -100.0, 1000.0))
        d = {gbox: "value"}
        assert d[gbox] == "value"

    def test_frozen(self):
        """GeoBox attributes should be immutable."""
        gbox = GeoBox("EPSG:32633", (10, 10), (100.0, 0.0, 0.0, 0.0, -100.0, 1000.0))
        with pytest.raises(AttributeError):
            gbox.crs = "EPSG:4326"


class TestGeoBoxFromOdc:
    """GeoBox.from_odc compatibility."""

    def test_from_odc_basic(self):
        """Convert from odc-geo GeoBox."""
        pytest.importorskip("odc.geo")
        from affine import Affine
        from odc.geo import CRS as OdcCRS  # noqa: N811
        from odc.geo.geobox import GeoBox as OdcGeoBox

        odc_gbox = OdcGeoBox(
            shape=(100, 200),
            affine=Affine(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
            crs=OdcCRS("EPSG:32633"),
        )

        gbox = GeoBox.from_odc(odc_gbox)
        assert gbox.shape == (100, 200)
        assert "32633" in gbox.crs
        assert gbox.affine[0] == 100.0
        assert gbox.affine[4] == -100.0
