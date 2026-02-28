"""Integration tests: reproject real AEF tiles via dask and cubed, compare results.

Uses a full 64-band AEF tile (64 x 8192 x 8192, int8, ~4GB) from Source Cooperative.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

da = pytest.importorskip("dask.array")
cubed = pytest.importorskip("cubed")
pytest.importorskip("aef_loader")

import rust_warp  # noqa: F401, E402  — registers .warp accessor
from aef_loader import AEFIndex, DataSource, VirtualTiffReader  # noqa: E402

SPATIAL_CHUNKS = {"time": 1, "band": 1, "y": 1024, "x": 1024}
DST_CHUNKS = (512, 512)
NODATA = -128  # AEF int8 nodata value


@pytest.fixture(scope="module")
def aef_tile():
    """Load a single AEF tile from Source Cooperative."""

    async def _load():
        index = AEFIndex(source=DataSource.SOURCE_COOP)
        await index.download()
        tiles = await index.query(
            bbox=(-122.5, 37.5, -122.0, 38.0),
            years=(2020, 2020),
            limit=1,
        )
        async with VirtualTiffReader() as reader:
            tree = await reader.open_tiles_by_zone(tiles)
        zone = next(iter(tree.children))
        return tree[zone].ds

    return asyncio.run(_load())


@pytest.fixture(scope="module")
def aef_allbands_dask(aef_tile):
    """All 64 bands (64 x 8192 x 8192) as a dask-backed DataArray."""
    ds = aef_tile.chunk(chunks=SPATIAL_CHUNKS)
    var_name = next(iter(ds.data_vars))
    return ds[var_name].isel(time=0)


@pytest.fixture(scope="module")
def aef_allbands_cubed(aef_tile):
    """All 64 bands (64 x 8192 x 8192) as a cubed-backed DataArray."""
    ds = aef_tile.chunk(
        chunks=SPATIAL_CHUNKS,
        chunked_array_type="cubed",
        from_array_kwargs={"spec": cubed.Spec(allowed_mem="2GB")},
    )
    var_name = next(iter(ds.data_vars))
    return ds[var_name].isel(time=0)


def _assert_valid_reproject(result_np, nodata=NODATA):
    """Sanity-check a reprojected int8 AEF result."""
    assert result_np.shape[-2] > 0
    assert result_np.shape[-1] > 0
    # Must contain real data, not just nodata/zeros
    valid = result_np[result_np != nodata]
    assert valid.size > 0, "No valid (non-nodata) pixels in output"
    assert valid.size > result_np.size * 0.5, (
        f"Less than 50% valid pixels: {valid.size}/{result_np.size}"
    )


@pytest.mark.slow
class TestAEFDask:
    """Reproject all 64 bands of a real AEF tile with dask."""

    def test_reproject_aef_tile_dask(self, aef_allbands_dask):
        assert hasattr(aef_allbands_dask.data, "dask")
        assert aef_allbands_dask.shape == (64, 8192, 8192)

        result = aef_allbands_dask.warp.reproject(
            "EPSG:4326", dst_chunks=DST_CHUNKS, resampling="nearest", nodata=NODATA
        )
        result_np = result.values

        assert result_np.ndim == 3
        assert result_np.shape[0] == 64
        _assert_valid_reproject(result_np)


@pytest.mark.slow
class TestAEFCubed:
    """Reproject all 64 bands of a real AEF tile with cubed."""

    def test_reproject_aef_tile_cubed(self, aef_allbands_cubed):
        assert isinstance(aef_allbands_cubed.data, cubed.Array)
        assert aef_allbands_cubed.shape == (64, 8192, 8192)

        result = aef_allbands_cubed.warp.reproject(
            "EPSG:4326", dst_chunks=DST_CHUNKS, resampling="nearest", nodata=NODATA
        )
        result_np = result.values

        assert result_np.ndim == 3
        assert result_np.shape[0] == 64
        _assert_valid_reproject(result_np)


@pytest.mark.slow
class TestAEFCrossBackend:
    """Compare dask and cubed reprojection of all 64 bands of a real AEF tile."""

    def test_dask_and_cubed_match(self, aef_allbands_dask, aef_allbands_cubed):
        """Dask and cubed must produce pixel-identical output on real data."""
        result_dask = aef_allbands_dask.warp.reproject(
            "EPSG:4326", dst_chunks=DST_CHUNKS, resampling="nearest", nodata=NODATA
        ).values
        result_cubed = aef_allbands_cubed.warp.reproject(
            "EPSG:4326", dst_chunks=DST_CHUNKS, resampling="nearest", nodata=NODATA
        ).values

        assert result_dask.shape == result_cubed.shape
        assert result_dask.shape[0] == 64

        # For int8 data, nodata is -128 (not NaN). Compare all pixels directly.
        np.testing.assert_array_equal(
            result_dask,
            result_cubed,
            err_msg="Dask and cubed produced different pixel values",
        )

        # Verify both results contain meaningful data
        _assert_valid_reproject(result_dask)
