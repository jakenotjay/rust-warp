"""Integration tests: reproject real AEF tiles via dask and cubed, compare results."""

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
def aef_2d_dask(aef_tile):
    """Single 2D spatial slice as a dask-backed DataArray."""
    ds = aef_tile.chunk(chunks=SPATIAL_CHUNKS)
    var_name = next(iter(ds.data_vars))
    return ds[var_name].isel(time=0, band=0)


@pytest.fixture(scope="module")
def aef_2d_cubed(aef_tile):
    """Single 2D spatial slice as a cubed-backed DataArray."""
    ds = aef_tile.chunk(
        chunks=SPATIAL_CHUNKS,
        chunked_array_type="cubed",
        from_array_kwargs={"spec": cubed.Spec(allowed_mem="2GB")},
    )
    var_name = next(iter(ds.data_vars))
    return ds[var_name].isel(time=0, band=0)


@pytest.mark.slow
class TestAEFDask:
    """Reproject a real AEF tile with dask."""

    def test_reproject_aef_tile_dask(self, aef_2d_dask):
        assert hasattr(aef_2d_dask.data, "dask")

        result = aef_2d_dask.warp.reproject("EPSG:4326", dst_chunks=DST_CHUNKS)
        result_np = result.values

        assert result_np.ndim == 2
        assert result_np.shape[0] > 0
        assert result_np.shape[1] > 0
        assert not np.all(np.isnan(result_np))


@pytest.mark.slow
class TestAEFCubed:
    """Reproject a real AEF tile with cubed."""

    def test_reproject_aef_tile_cubed(self, aef_2d_cubed):
        assert isinstance(aef_2d_cubed.data, cubed.Array)

        result = aef_2d_cubed.warp.reproject("EPSG:4326", dst_chunks=DST_CHUNKS)
        result_np = result.values

        assert result_np.ndim == 2
        assert result_np.shape[0] > 0
        assert result_np.shape[1] > 0
        assert not np.all(np.isnan(result_np))


@pytest.mark.slow
class TestAEFCrossBackend:
    """Compare dask and cubed reprojection of the same real AEF tile."""

    def test_dask_and_cubed_match(self, aef_2d_dask, aef_2d_cubed):
        """Dask and cubed must produce identical output on real data."""
        result_dask = aef_2d_dask.warp.reproject(
            "EPSG:4326", dst_chunks=DST_CHUNKS, resampling="nearest"
        ).values
        result_cubed = aef_2d_cubed.warp.reproject(
            "EPSG:4326", dst_chunks=DST_CHUNKS, resampling="nearest"
        ).values

        assert result_dask.shape == result_cubed.shape

        dask_nan = np.isnan(result_dask)
        cubed_nan = np.isnan(result_cubed)
        np.testing.assert_array_equal(dask_nan, cubed_nan, err_msg="NaN pattern mismatch")

        both_valid = ~dask_nan & ~cubed_nan
        if both_valid.any():
            np.testing.assert_array_equal(
                result_dask[both_valid],
                result_cubed[both_valid],
                err_msg="Pixel values differ between dask and cubed",
            )
