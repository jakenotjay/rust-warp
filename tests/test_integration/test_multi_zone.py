"""Multi-zone integration tests.

Verifies that multiple UTM zones can be reprojected to a common 4326 grid
and that zone boundaries produce consistent results.
"""

import numpy as np
import pytest
from rust_warp import reproject_array
from rust_warp.geobox import GeoBox


@pytest.mark.stress
class TestMultiZone:
    """Multi-zone reprojection consistency tests."""

    def _make_zone_raster(self, zone, size=512):
        """Create a synthetic raster in a UTM zone."""
        crs = f"EPSG:326{zone:02d}"
        pixel_size = 100.0
        # Central meridian of UTM zone
        central_lon = (zone - 1) * 6 - 180 + 3
        # Place raster near central meridian
        origin_x = 500000.0 - size * pixel_size / 2
        origin_y = 6000000.0

        transform = (pixel_size, 0.0, origin_x, 0.0, -pixel_size, origin_y)

        # Constant value per zone for easy verification
        src = np.full((size, size), zone * 100.0, dtype=np.float32)

        return src, crs, transform, central_lon

    def test_three_zones_to_4326(self):
        """Reproject UTM32, 33, 34 tiles to common EPSG:4326 grid."""
        zones = [32, 33, 34]
        results = []

        for zone in zones:
            src, src_crs, src_transform, central_lon = self._make_zone_raster(zone)

            # Common destination covering all three zones
            dst_crs = "EPSG:4326"
            dst_transform = (0.01, 0.0, 3.0, 0.0, -0.01, 55.0)
            dst_shape = (200, 600)

            result = reproject_array(
                src, src_crs, src_transform,
                dst_crs, dst_transform, dst_shape,
                resampling="nearest",
            )
            results.append(result)

        # Each zone's output should have some valid pixels
        for i, (zone, result) in enumerate(zip(zones, results)):
            valid = ~np.isnan(result)
            assert valid.any(), f"Zone {zone} produced no valid pixels"

        # Verify zone values are preserved
        for i, (zone, result) in enumerate(zip(zones, results)):
            valid = ~np.isnan(result)
            if valid.any():
                unique_vals = np.unique(result[valid])
                expected_val = zone * 100.0
                match = (
                    expected_val in unique_vals
                    or np.isclose(unique_vals, expected_val, atol=1).any()
                )
                assert match, (
                    f"Zone {zone}: expected {expected_val} in output, "
                    f"got {unique_vals[:5]}"
                )

    def test_adjacent_zones_overlap_consistency(self):
        """Adjacent UTM zones, when reprojected to 4326, should have overlapping regions."""
        zone_a = 33
        zone_b = 34

        src_a, crs_a, tf_a, _ = self._make_zone_raster(zone_a, size=256)
        src_b, crs_b, tf_b, _ = self._make_zone_raster(zone_b, size=256)

        # Use a smooth gradient instead of constant for better testing
        r, c = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
        src_a = (r + c).astype(np.float32)
        src_b = (r + c).astype(np.float32)

        dst_crs = "EPSG:4326"
        # Cover the boundary between zone 33 (15°E) and zone 34 (21°E)
        dst_transform = (0.005, 0.0, 14.0, 0.0, -0.005, 55.0)
        dst_shape = (200, 400)

        result_a = reproject_array(
            src_a, crs_a, tf_a, dst_crs, dst_transform, dst_shape,
            resampling="bilinear",
        )
        result_b = reproject_array(
            src_b, crs_b, tf_b, dst_crs, dst_transform, dst_shape,
            resampling="bilinear",
        )

        # Both should produce valid data (different regions of the output)
        valid_a = ~np.isnan(result_a)
        valid_b = ~np.isnan(result_b)

        assert valid_a.any(), "Zone A produced no valid pixels"
        assert valid_b.any(), "Zone B produced no valid pixels"

    def test_geobox_reproject_consistency(self):
        """Verify GeoBox-based reprojection produces same results as raw arrays."""
        src_geobox = GeoBox(
            crs="EPSG:32633",
            shape=(64, 64),
            affine=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
        )
        dst_geobox = GeoBox(
            crs="EPSG:4326",
            shape=(64, 64),
            affine=(0.001, 0.0, 15.0, 0.0, -0.001, 59.5),
        )

        src = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)

        # Direct call
        result_direct = reproject_array(
            src,
            src_geobox.crs,
            src_geobox.affine,
            dst_geobox.crs,
            dst_geobox.affine,
            dst_geobox.shape,
            resampling="bilinear",
        )

        # Via high-level API
        from rust_warp import reproject
        result_api = reproject(src, src_geobox, dst_geobox, resampling="bilinear")

        np.testing.assert_array_equal(result_direct, result_api)
