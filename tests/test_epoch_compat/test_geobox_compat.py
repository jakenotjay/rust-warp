"""GeoBox compatibility tests with odc-geo.

Verifies that rust-warp's GeoBox can interop with odc-geo's GeoBox,
and that coordinate output matches.
"""

import numpy as np
import pytest
from rust_warp.geobox import GeoBox

odc_geo = pytest.importorskip("odc.geo")
from odc.geo.geobox import GeoBox as OdcGeoBox  # noqa: E402


class TestGeoBoxFromOdc:
    """Test GeoBox.from_odc() roundtrip."""

    def test_utm_geobox_roundtrip(self):
        """Create odc-geo GeoBox, convert to rust-warp, verify fields match."""
        from affine import Affine
        from odc.geo import CRS

        crs = CRS("EPSG:32633")
        transform = Affine(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0)
        odc_gbox = OdcGeoBox((64, 64), crs=crs, affine=transform)

        rw_gbox = GeoBox.from_odc(odc_gbox)

        assert rw_gbox.crs == "EPSG:32633"
        assert rw_gbox.shape == (64, 64)
        assert rw_gbox.affine[0] == pytest.approx(100.0)
        assert rw_gbox.affine[4] == pytest.approx(-100.0)

    def test_4326_geobox_roundtrip(self):
        """Geographic CRS GeoBox roundtrip."""
        from affine import Affine
        from odc.geo import CRS

        crs = CRS("EPSG:4326")
        transform = Affine(0.01, 0.0, 10.0, 0.0, -0.01, 60.0)
        odc_gbox = OdcGeoBox((100, 200), crs=crs, affine=transform)

        rw_gbox = GeoBox.from_odc(odc_gbox)

        assert rw_gbox.crs == "EPSG:4326"
        assert rw_gbox.shape == (100, 200)


class TestXrCoordsCompat:
    """Compare xr_coords() output format with odc-geo."""

    def test_coords_shape_and_monotonicity(self):
        """xr_coords() should produce monotonic 1D arrays of correct length."""
        gbox = GeoBox(
            crs="EPSG:32633",
            shape=(64, 128),
            affine=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
        )

        coords = gbox.xr_coords()
        assert "x" in coords
        assert "y" in coords
        assert len(coords["x"]) == 128
        assert len(coords["y"]) == 64

        # x should be increasing (positive pixel width)
        assert np.all(np.diff(coords["x"]) > 0)
        # y should be decreasing (negative pixel height)
        assert np.all(np.diff(coords["y"]) < 0)

    def test_pixel_centers_at_half_pixel(self):
        """First coordinate should be at half-pixel offset from origin."""
        gbox = GeoBox(
            crs="EPSG:32633",
            shape=(10, 10),
            affine=(100.0, 0.0, 500000.0, 0.0, -100.0, 6000000.0),
        )

        coords = gbox.xr_coords()
        # First x center: origin + 0.5 * pixel_width = 500000 + 50 = 500050
        assert coords["x"][0] == pytest.approx(500050.0)
        # First y center: origin + 0.5 * pixel_height = 6000000 + 0.5*(-100) = 5999950
        assert coords["y"][0] == pytest.approx(5999950.0)


class TestBoundsCompat:
    """Test GeoBox.bounds property."""

    def test_bounds_basic(self):
        """Bounds should match expected bounding box."""
        gbox = GeoBox(
            crs="EPSG:32633",
            shape=(100, 200),
            affine=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
        )

        left, bottom, right, top = gbox.bounds
        assert left == pytest.approx(500000.0)
        assert right == pytest.approx(520000.0)  # 500000 + 200*100
        assert top == pytest.approx(6600000.0)
        assert bottom == pytest.approx(6590000.0)  # 6600000 - 100*100

    def test_resolution_property(self):
        """Resolution should return positive values."""
        gbox = GeoBox(
            crs="EPSG:32633",
            shape=(100, 200),
            affine=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
        )

        res_x, res_y = gbox.resolution
        assert res_x == pytest.approx(100.0)
        assert res_y == pytest.approx(100.0)


class TestFromBbox:
    """Test GeoBox.from_bbox()."""

    def test_from_bbox_with_resolution(self):
        gbox = GeoBox.from_bbox(
            bbox=(500000.0, 6590000.0, 520000.0, 6600000.0),
            crs="EPSG:32633",
            resolution=100.0,
        )

        assert gbox.shape == (100, 200)
        assert gbox.affine[0] == pytest.approx(100.0)

    def test_from_bbox_with_shape(self):
        gbox = GeoBox.from_bbox(
            bbox=(500000.0, 6590000.0, 520000.0, 6600000.0),
            crs="EPSG:32633",
            shape=(50, 100),
        )

        assert gbox.shape == (50, 100)
        assert gbox.affine[0] == pytest.approx(200.0)  # 20000/100
