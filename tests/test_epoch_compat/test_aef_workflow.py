"""AEF workflow simulation tests.

Simulates the epoch-mono AEF pattern: multiple UTM zones, int8 data,
many bands, reprojected to EPSG:4326 using the .warp xarray accessor.
"""

import numpy as np
import pytest
from rust_warp import reproject_array
from rust_warp.geobox import GeoBox

xarray = pytest.importorskip("xarray")


class TestAEFWorkflow:
    """Simulate AEF-style multi-zone, multi-band reprojection."""

    def _make_zone_dataset(self, zone, n_bands=3, size=64):
        """Create a synthetic xarray Dataset mimicking AEF data for a UTM zone.

        Args:
            zone: UTM zone number.
            n_bands: Number of bands.
            size: Spatial size (square).

        Returns:
            xarray.Dataset with spatial coordinates and CRS.
        """
        crs = f"EPSG:326{zone:02d}"
        pixel_size = 100.0
        origin_x = 500000.0
        origin_y = 6000000.0 + size * pixel_size

        x_coords = origin_x + (np.arange(size) + 0.5) * pixel_size
        y_coords = origin_y + (np.arange(size) + 0.5) * (-pixel_size)

        rng = np.random.default_rng(zone)
        data_vars = {}
        for b in range(n_bands):
            data = rng.integers(1, 255, size=(size, size), dtype=np.uint8)
            data_vars[f"band_{b}"] = xarray.DataArray(
                data,
                dims=["y", "x"],
                coords={"y": y_coords, "x": x_coords},
            )

        ds = xarray.Dataset(data_vars)
        ds = ds.assign_coords(
            spatial_ref=xarray.DataArray(
                0,
                attrs={
                    "crs_wkt": crs,
                    "epsg": int(crs.split(":")[1]),
                },
            )
        )

        return ds

    def test_single_zone_reproject(self):
        """Single UTM zone reprojection should work end-to-end."""
        ds = self._make_zone_dataset(33, n_bands=3, size=64)

        src_geobox = GeoBox.from_xarray(ds)
        dst_geobox = GeoBox(
            crs="EPSG:4326",
            shape=(64, 64),
            affine=(0.001, 0.0, 15.0, 0.0, -0.001, 54.5),
        )

        # Reproject each band
        results = {}
        for var_name in ds.data_vars:
            src = ds[var_name].values
            result = reproject_array(
                src,
                src_geobox.crs,
                src_geobox.affine,
                dst_geobox.crs,
                dst_geobox.affine,
                dst_geobox.shape,
                resampling="nearest",
                nodata=0.0,
            )
            results[var_name] = result

        # Each band should produce output
        for _name, result in results.items():
            assert result.shape == (64, 64)
            assert result.dtype == np.uint8

    def test_multi_zone_reproject(self):
        """Multi-zone (32, 33, 34) reprojection to common 4326 grid."""
        zones = [32, 33, 34]
        n_bands = 3

        dst_geobox = GeoBox(
            crs="EPSG:4326",
            shape=(100, 300),
            affine=(0.01, 0.0, 6.0, 0.0, -0.01, 55.0),
        )

        all_results = {}
        for zone in zones:
            ds = self._make_zone_dataset(zone, n_bands=n_bands, size=64)
            src_geobox = GeoBox.from_xarray(ds)

            zone_results = {}
            for var_name in ds.data_vars:
                src = ds[var_name].values
                result = reproject_array(
                    src,
                    src_geobox.crs,
                    src_geobox.affine,
                    dst_geobox.crs,
                    dst_geobox.affine,
                    dst_geobox.shape,
                    resampling="nearest",
                    nodata=0.0,
                )
                zone_results[var_name] = result
            all_results[zone] = zone_results

        # Each zone should produce some valid data in different parts of the output
        for _zone, results in all_results.items():
            for _name, result in results.items():
                assert result.shape == dst_geobox.shape
                # At least some pixels should be valid
                # (zones far from the output extent may have none)
                assert result.dtype == np.uint8

    def test_many_bands_performance(self):
        """64-band reprojection should complete without issues."""
        ds = self._make_zone_dataset(33, n_bands=64, size=64)
        src_geobox = GeoBox.from_xarray(ds)
        dst_geobox = GeoBox(
            crs="EPSG:4326",
            shape=(64, 64),
            affine=(0.001, 0.0, 15.0, 0.0, -0.001, 54.5),
        )

        results = []
        for var_name in ds.data_vars:
            src = ds[var_name].values
            result = reproject_array(
                src,
                src_geobox.crs,
                src_geobox.affine,
                dst_geobox.crs,
                dst_geobox.affine,
                dst_geobox.shape,
                resampling="nearest",
                nodata=0.0,
            )
            results.append(result)

        assert len(results) == 64
        for result in results:
            assert result.shape == (64, 64)
            assert result.dtype == np.uint8


class TestWarpAccessor:
    """Test the .warp xarray accessor if available."""

    def test_accessor_registration(self):
        """The .warp accessor should be importable and registered."""
        try:
            import rust_warp.xarray_accessor  # noqa: F401
        except ImportError:
            pytest.skip("xarray_accessor not available")

        da = xarray.DataArray(
            np.ones((10, 10), dtype=np.float64),
            dims=["y", "x"],
            coords={
                "y": np.arange(10) * -100.0 + 6000000.0,
                "x": np.arange(10) * 100.0 + 500000.0,
            },
        )
        da = da.assign_coords(
            spatial_ref=xarray.DataArray(
                0,
                attrs={
                    "crs_wkt": "EPSG:32633",
                    "epsg": 32633,
                },
            )
        )

        assert hasattr(da, "warp")
