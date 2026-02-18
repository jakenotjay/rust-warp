"""Integration test: reproject real ERA5 data from GCS, compare rust-warp vs odc-geo."""

import time

import numpy as np
import pytest
import rasterio.warp
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.warp import rio_reproject
from rasterio.crs import CRS

from rust_warp import reproject_array


@pytest.fixture(scope="module")
def era5_europe():
    """Fetch a single ERA5 timestep over Europe from GCS.

    Returns source array, CRS, transform, and destination grid parameters.
    """
    from obstore.store import GCSStore
    from zarr.storage import ObjectStore
    import xarray as xr

    gcs = GCSStore(
        "gcp-public-data-arco-era5",
        prefix="ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        skip_signature=True,
    )
    store = ObjectStore(gcs, read_only=True)
    ds = xr.open_zarr(store, consolidated=True)

    t2m = (
        ds["2m_temperature"]
        .sel(time="2023-07-15T12:00", method="nearest")
        .sel(latitude=slice(72, 35), longitude=slice(-25, 45))
    )
    data = t2m.values.astype(np.float64)

    lats = t2m.latitude.values  # descending (72 -> 35)
    lons = t2m.longitude.values  # ascending (-25 -> 45)
    dx = float(lons[1] - lons[0])
    dy = float(lats[1] - lats[0])

    src_crs = "EPSG:4326"
    src_transform = (dx, 0.0, float(lons[0]) - dx / 2, 0.0, dy, float(lats[0]) - dy / 2)

    dst_crs = "EPSG:3857"
    rows, cols = data.shape
    left = float(lons[0]) - dx / 2
    right = float(lons[-1]) + dx / 2
    top = float(lats[0]) - dy / 2
    bottom = float(lats[-1]) + dy / 2

    dst_affine, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        CRS.from_user_input(src_crs),
        CRS.from_user_input(dst_crs),
        cols,
        rows,
        left=left,
        bottom=bottom,
        right=right,
        top=top,
    )
    dst_transform = tuple(dst_affine)[:6]
    dst_shape = (dst_height, dst_width)

    return {
        "data": data,
        "src_crs": src_crs,
        "src_transform": src_transform,
        "dst_crs": dst_crs,
        "dst_transform": dst_transform,
        "dst_affine": dst_affine,
        "dst_shape": dst_shape,
    }


@pytest.mark.slow
class TestERA5Reprojection:
    """Reproject real ERA5 Europe subset: EPSG:4326 -> EPSG:3857."""

    def test_rust_warp_produces_valid_output(self, era5_europe):
        s = era5_europe
        result = reproject_array(
            s["data"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling="bilinear",
        )
        assert result.shape == s["dst_shape"]
        valid_pct = np.count_nonzero(~np.isnan(result)) / result.size * 100
        assert valid_pct > 90, f"Only {valid_pct:.1f}% valid pixels"

    def test_rust_warp_vs_odc_geo(self, era5_europe):
        """Compare rust-warp against odc-geo (rasterio/GDAL) on real ERA5 data."""
        s = era5_europe

        # --- rust-warp ---
        t0 = time.perf_counter()
        rust_result = reproject_array(
            s["data"], s["src_crs"], s["src_transform"],
            s["dst_crs"], s["dst_transform"], s["dst_shape"],
            resampling="bilinear",
        )
        rust_time = time.perf_counter() - t0

        # --- odc-geo (wraps rasterio/GDAL) ---
        src_gbox = GeoBox(
            shape=s["data"].shape,
            affine=Affine(*s["src_transform"]),
            crs=s["src_crs"],
        )
        dst_gbox = GeoBox(
            shape=s["dst_shape"],
            affine=s["dst_affine"],
            crs=s["dst_crs"],
        )
        odc_result = np.full(s["dst_shape"], np.nan, dtype=np.float64)
        t0 = time.perf_counter()
        rio_reproject(
            s["data"], odc_result, src_gbox, dst_gbox,
            resampling="bilinear", src_nodata=np.nan, dst_nodata=np.nan,
        )
        odc_time = time.perf_counter() - t0

        # --- Report timing ---
        print(f"\n  rust-warp: {rust_time:.3f}s")
        print(f"  odc-geo:   {odc_time:.3f}s")
        print(f"  speedup:   {odc_time / rust_time:.1f}x")

        # --- Compare ---
        both_valid = ~np.isnan(rust_result) & ~np.isnan(odc_result)
        assert both_valid.any(), "No overlapping valid pixels"

        diff = np.abs(rust_result[both_valid] - odc_result[both_valid])
        print(f"  max diff:  {diff.max():.4f} K")
        print(f"  mean diff: {diff.mean():.4f} K")
        print(f"  median:    {np.median(diff):.4f} K")

        # Temperature values should agree within 1K for bilinear on a smooth field
        assert np.median(diff) < 0.5, f"Median diff {np.median(diff):.4f} K too large"
        assert diff.max() < 5.0, f"Max diff {diff.max():.4f} K too large"
