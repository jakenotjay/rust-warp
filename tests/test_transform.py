"""Tests for transform_points coordinate transformation."""

import numpy as np
import pytest
from rust_warp import transform_points


class TestTransformRoundTrip:
    """Round-trip transforms should recover original coordinates."""

    def test_4326_to_utm33_roundtrip(self):
        x = np.array([15.0, 15.5, 16.0], dtype=np.float64)
        y = np.array([52.0, 52.5, 53.0], dtype=np.float64)

        # Forward: 4326 → UTM33
        x_utm, y_utm = transform_points(x, y, "EPSG:4326", "EPSG:32633")
        assert x_utm.dtype == np.float64
        assert y_utm.dtype == np.float64

        # UTM33 values should be in reasonable range
        assert np.all(x_utm > 400_000)
        assert np.all(x_utm < 700_000)
        assert np.all(y_utm > 5_700_000)
        assert np.all(y_utm < 5_900_000)

        # Inverse: UTM33 → 4326
        x_back, y_back = transform_points(x_utm, y_utm, "EPSG:32633", "EPSG:4326")

        np.testing.assert_allclose(x_back, x, atol=1e-6)
        np.testing.assert_allclose(y_back, y, atol=1e-6)

    def test_4326_to_3857_roundtrip(self):
        x = np.array([0.0, 10.0, -10.0], dtype=np.float64)
        y = np.array([0.0, 45.0, -45.0], dtype=np.float64)

        x_merc, y_merc = transform_points(x, y, "EPSG:4326", "EPSG:3857")
        x_back, y_back = transform_points(x_merc, y_merc, "EPSG:3857", "EPSG:4326")

        np.testing.assert_allclose(x_back, x, atol=1e-6)
        np.testing.assert_allclose(y_back, y, atol=1e-6)


class TestTransformVsPyproj:
    """Compare against pyproj for correctness."""

    def test_4326_to_utm33(self):
        pyproj = pytest.importorskip("pyproj")

        x = np.array([14.0, 15.0, 16.0], dtype=np.float64)
        y = np.array([51.0, 52.0, 53.0], dtype=np.float64)

        # rust-warp
        x_rust, y_rust = transform_points(x, y, "EPSG:4326", "EPSG:32633")

        # pyproj
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
        x_ref, y_ref = transformer.transform(x, y)

        np.testing.assert_allclose(x_rust, x_ref, atol=1.0)  # 1m tolerance
        np.testing.assert_allclose(y_rust, y_ref, atol=1.0)


class TestTransformEdgeCases:
    """Edge cases and error handling."""

    def test_empty_arrays(self):
        x = np.array([], dtype=np.float64)
        y = np.array([], dtype=np.float64)

        x_out, y_out = transform_points(x, y, "EPSG:4326", "EPSG:32633")
        assert len(x_out) == 0
        assert len(y_out) == 0

    def test_single_point(self):
        x = np.array([15.0], dtype=np.float64)
        y = np.array([52.0], dtype=np.float64)

        x_out, y_out = transform_points(x, y, "EPSG:4326", "EPSG:32633")
        assert len(x_out) == 1
        assert len(y_out) == 1
        assert x_out[0] > 400_000

    def test_mismatched_lengths(self):
        x = np.array([1.0, 2.0], dtype=np.float64)
        y = np.array([1.0], dtype=np.float64)

        with pytest.raises(ValueError, match="same length"):
            transform_points(x, y, "EPSG:4326", "EPSG:32633")

    def test_invalid_crs(self):
        x = np.array([15.0], dtype=np.float64)
        y = np.array([52.0], dtype=np.float64)

        with pytest.raises(ValueError, match="EPSG:99999"):
            transform_points(x, y, "EPSG:99999", "EPSG:32633")

    def test_same_crs_identity(self):
        x = np.array([500000.0, 510000.0], dtype=np.float64)
        y = np.array([5760000.0, 5770000.0], dtype=np.float64)

        x_out, y_out = transform_points(x, y, "EPSG:32633", "EPSG:32633")

        np.testing.assert_allclose(x_out, x, atol=0.01)
        np.testing.assert_allclose(y_out, y, atol=0.01)
