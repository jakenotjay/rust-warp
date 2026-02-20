"""Tests for plan_reproject chunk planner."""

import numpy as np
from rust_warp import plan_reproject

EXPECTED_KEYS = {
    "dst_slice",
    "src_slice",
    "src_transform",
    "dst_transform",
    "dst_tile_shape",
    "has_data",
}

# Common test parameters: 64x64 UTM 33N source
SRC_CRS = "EPSG:32633"
SRC_TRANSFORM = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0)
SRC_SHAPE = (64, 64)


class TestPlanReprojectBasic:
    """plan_reproject returns well-formed tile dicts."""

    def test_returns_nonempty_list_with_correct_keys(self):
        result = plan_reproject(
            src_crs=SRC_CRS,
            src_transform=SRC_TRANSFORM,
            src_shape=SRC_SHAPE,
            dst_crs="EPSG:4326",
            dst_transform=(0.001, 0.0, 14.0, 0.0, -0.001, 60.0),
            dst_shape=(64, 64),
            dst_chunks=(32, 32),
        )
        assert len(result) > 0
        for tile in result:
            assert set(tile.keys()) == EXPECTED_KEYS

    def test_tile_shape_matches_dst_slice(self):
        result = plan_reproject(
            src_crs=SRC_CRS,
            src_transform=SRC_TRANSFORM,
            src_shape=SRC_SHAPE,
            dst_crs=SRC_CRS,
            dst_transform=SRC_TRANSFORM,
            dst_shape=SRC_SHAPE,
            dst_chunks=(32, 32),
        )
        for tile in result:
            r0, r1, c0, c1 = tile["dst_slice"]
            assert tile["dst_tile_shape"] == (r1 - r0, c1 - c0)


class TestPlanReprojectCoverage:
    """Tile dst_slices should cover the full destination extent."""

    def test_same_crs_2x2_tiles(self):
        result = plan_reproject(
            src_crs=SRC_CRS,
            src_transform=SRC_TRANSFORM,
            src_shape=SRC_SHAPE,
            dst_crs=SRC_CRS,
            dst_transform=SRC_TRANSFORM,
            dst_shape=SRC_SHAPE,
            dst_chunks=(32, 32),
        )
        assert len(result) == 4
        covered = np.zeros(SRC_SHAPE, dtype=bool)
        for tile in result:
            r0, r1, c0, c1 = tile["dst_slice"]
            assert not covered[r0:r1, c0:c1].any(), "Overlapping tiles"
            covered[r0:r1, c0:c1] = True
        assert covered.all(), "Tiles do not cover full extent"

    def test_uneven_chunks_full_coverage(self):
        """100x100 with 64x64 tiles should still cover everything."""
        shape = (100, 100)
        transform = (100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0)
        result = plan_reproject(
            src_crs=SRC_CRS,
            src_transform=transform,
            src_shape=shape,
            dst_crs=SRC_CRS,
            dst_transform=transform,
            dst_shape=shape,
            dst_chunks=(64, 64),
        )
        covered = np.zeros(shape, dtype=bool)
        for tile in result:
            r0, r1, c0, c1 = tile["dst_slice"]
            covered[r0:r1, c0:c1] = True
        assert covered.all()


class TestPlanReprojectBounds:
    """Source slices should be within source bounds."""

    def test_cross_crs_src_within_bounds(self):
        result = plan_reproject(
            src_crs=SRC_CRS,
            src_transform=SRC_TRANSFORM,
            src_shape=SRC_SHAPE,
            dst_crs="EPSG:4326",
            dst_transform=(0.001, 0.0, 14.0, 0.0, -0.001, 60.0),
            dst_shape=(64, 64),
            dst_chunks=(32, 32),
        )
        for tile in result:
            if tile["has_data"]:
                sr0, sr1, sc0, sc1 = tile["src_slice"]
                assert sr0 >= 0
                assert sr1 <= SRC_SHAPE[0]
                assert sc0 >= 0
                assert sc1 <= SRC_SHAPE[1]
                assert sr1 > sr0
                assert sc1 > sc0


class TestPlanReprojectNoChunks:
    """Without dst_chunks, should return a single tile."""

    def test_single_tile(self):
        result = plan_reproject(
            src_crs=SRC_CRS,
            src_transform=SRC_TRANSFORM,
            src_shape=SRC_SHAPE,
            dst_crs=SRC_CRS,
            dst_transform=SRC_TRANSFORM,
            dst_shape=SRC_SHAPE,
        )
        assert len(result) == 1
        assert result[0]["dst_slice"] == (0, 64, 0, 64)
        assert result[0]["dst_tile_shape"] == (64, 64)
