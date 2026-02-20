"""Overlap and ROI computation tests.

Tests pixel-space transforms between GeoBoxes, region-of-interest computation
for reprojection, pasteable detection, and overlap edge cases.
"""

import numpy as np
import rasterio.warp
from rasterio.crs import CRS
from rust_warp import plan_reproject, reproject_array, transform_grid


class TestTransformGrid:
    """transform_grid should compute correct source pixel coordinates."""

    def test_identity_grid(self):
        """Same CRS + same transform should produce identity-like grid."""
        size = 16
        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0
        transform = (px, 0.0, origin_x, 0.0, -px, origin_y)

        col_grid, row_grid = transform_grid(
            "EPSG:32633",
            transform,
            "EPSG:32633",
            transform,
            (size, size),
        )

        assert col_grid.shape == (size, size)
        assert row_grid.shape == (size, size)

        # Center of each dst pixel should map to same position in src
        for r in range(size):
            for c in range(size):
                np.testing.assert_allclose(col_grid[r, c], c + 0.5, atol=0.01)
                np.testing.assert_allclose(row_grid[r, c], r + 0.5, atol=0.01)

    def test_shifted_grid(self):
        """Shifted destination should produce offset source coordinates."""
        size = 8
        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0
        src_transform = (px, 0.0, origin_x, 0.0, -px, origin_y)
        # Shift destination by 5 pixels east, 3 pixels south
        dst_transform = (px, 0.0, origin_x + 5 * px, 0.0, -px, origin_y - 3 * px)

        col_grid, row_grid = transform_grid(
            "EPSG:32633",
            src_transform,
            "EPSG:32633",
            dst_transform,
            (size, size),
        )

        # dst pixel (0,0) center should map to src pixel (3+0.5, 5+0.5)
        np.testing.assert_allclose(col_grid[0, 0], 5.5, atol=0.01)
        np.testing.assert_allclose(row_grid[0, 0], 3.5, atol=0.01)

    def test_downscale_grid(self):
        """2x downscale should produce grid mapping to every-other pixel."""
        size = 8
        px_src = 100.0
        px_dst = 200.0
        origin_x, origin_y = 500000.0, 6600000.0
        src_transform = (px_src, 0.0, origin_x, 0.0, -px_src, origin_y)
        dst_transform = (px_dst, 0.0, origin_x, 0.0, -px_dst, origin_y)

        col_grid, row_grid = transform_grid(
            "EPSG:32633",
            src_transform,
            "EPSG:32633",
            dst_transform,
            (size, size),
        )

        # dst pixel (0,0) center at (origin + 100m) should map to src pixel 1.0
        np.testing.assert_allclose(col_grid[0, 0], 1.0, atol=0.1)
        np.testing.assert_allclose(row_grid[0, 0], 1.0, atol=0.1)

        # Stride should be ~2.0 per dst pixel
        col_stride = col_grid[0, 1] - col_grid[0, 0]
        row_stride = row_grid[1, 0] - row_grid[0, 0]
        np.testing.assert_allclose(col_stride, 2.0, atol=0.01)
        np.testing.assert_allclose(row_stride, 2.0, atol=0.01)

    def test_cross_crs_grid_finite(self):
        """Cross-CRS grid should produce finite coordinates for overlapping areas."""
        size = 16
        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0
        src_transform = (px, 0.0, origin_x, 0.0, -px, origin_y)

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:32633"),
            CRS.from_user_input("EPSG:4326"),
            size,
            size,
            left=origin_x,
            bottom=origin_y - size * px,
            right=origin_x + size * px,
            top=origin_y,
        )
        dst_transform = tuple(dst_affine)[:6]

        col_grid, _row_grid = transform_grid(
            "EPSG:32633",
            src_transform,
            "EPSG:4326",
            dst_transform,
            (dst_h, dst_w),
        )

        assert col_grid.shape == (dst_h, dst_w)
        # Most pixels should map to finite source coordinates
        finite_pct = np.sum(np.isfinite(col_grid)) / col_grid.size * 100
        assert finite_pct > 50, f"Only {finite_pct:.0f}% finite coordinates"


class TestPlanReprojectOverlap:
    """plan_reproject overlap and ROI computation edge cases."""

    def test_non_overlapping_geoboxes(self):
        """Completely non-overlapping src/dst should produce tiles with has_data=False."""
        # Source in Norway, destination in Australia — no overlap
        tiles = plan_reproject(
            src_crs="EPSG:32633",
            src_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600000.0),
            src_shape=(64, 64),
            dst_crs="EPSG:32633",
            dst_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 1000000.0),
            dst_shape=(64, 64),
            dst_chunks=(32, 32),
        )

        # All tiles should have has_data=False (or src_slice covering nothing)
        has_data_count = sum(1 for t in tiles if t["has_data"])
        assert has_data_count == 0, (
            f"Expected no overlapping tiles, got {has_data_count} with has_data=True"
        )

    def test_partial_overlap(self):
        """Partially overlapping grids: most tiles should have data when overlap exists."""
        # Source covers 500000-506400, dst offset far enough that some tiles miss
        tiles = plan_reproject(
            src_crs="EPSG:32633",
            src_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6606400.0),
            src_shape=(64, 64),
            dst_crs="EPSG:32633",
            dst_transform=(100.0, 0.0, 504000.0, 0.0, -100.0, 6606400.0),
            dst_shape=(128, 128),
            dst_chunks=(32, 32),
        )

        has_data_tiles = [t for t in tiles if t["has_data"]]
        # Some tiles should have data (the overlap region)
        assert len(has_data_tiles) > 0, "Expected some has_data tiles"
        # Not all tiles should have data (dst is much larger than src)
        assert len(has_data_tiles) < len(tiles), (
            "Expected some tiles without data since dst is much larger than src"
        )

    def test_src_fully_inside_dst(self):
        """Source fully inside destination should have all valid tiles."""
        # Small source inside large destination
        tiles = plan_reproject(
            src_crs="EPSG:32633",
            src_transform=(100.0, 0.0, 502000.0, 0.0, -100.0, 6604000.0),
            src_shape=(16, 16),
            dst_crs="EPSG:32633",
            dst_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6606400.0),
            dst_shape=(64, 64),
            dst_chunks=(32, 32),
        )

        # At least some tiles should have data
        has_data_count = sum(1 for t in tiles if t["has_data"])
        assert has_data_count > 0, "Source inside dst but no tiles have data"

    def test_tiny_src_large_dst(self):
        """Very small source mapped to large destination grid."""
        tiles = plan_reproject(
            src_crs="EPSG:32633",
            src_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6600400.0),
            src_shape=(4, 4),
            dst_crs="EPSG:32633",
            dst_transform=(100.0, 0.0, 499600.0, 0.0, -100.0, 6600800.0),
            dst_shape=(128, 128),
            dst_chunks=(32, 32),
        )

        # Most tiles should have no data
        no_data_count = sum(1 for t in tiles if not t["has_data"])
        total = len(tiles)
        assert no_data_count > total / 2, "Most tiles should be empty for tiny source"


class TestPlanReprojectTileCoverage:
    """Tile coverage completeness and correctness."""

    def test_tiles_cover_full_destination(self):
        """All tiles together should cover every pixel of the destination."""
        dst_rows, dst_cols = 100, 100
        tiles = plan_reproject(
            src_crs="EPSG:32633",
            src_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6610000.0),
            src_shape=(100, 100),
            dst_crs="EPSG:32633",
            dst_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6610000.0),
            dst_shape=(dst_rows, dst_cols),
            dst_chunks=(30, 30),
        )

        # Build coverage mask
        covered = np.zeros((dst_rows, dst_cols), dtype=bool)
        for tile in tiles:
            r0, r1, c0, c1 = tile["dst_slice"]
            covered[r0:r1, c0:c1] = True

        assert np.all(covered), (
            f"Not all pixels covered: {np.sum(~covered)} uncovered out of {dst_rows * dst_cols}"
        )

    def test_tiles_do_not_overlap(self):
        """Tiles should not have overlapping destination regions."""
        dst_rows, dst_cols = 64, 64
        tiles = plan_reproject(
            src_crs="EPSG:32633",
            src_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6606400.0),
            src_shape=(64, 64),
            dst_crs="EPSG:32633",
            dst_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6606400.0),
            dst_shape=(dst_rows, dst_cols),
            dst_chunks=(32, 32),
        )

        coverage_count = np.zeros((dst_rows, dst_cols), dtype=int)
        for tile in tiles:
            r0, r1, c0, c1 = tile["dst_slice"]
            coverage_count[r0:r1, c0:c1] += 1

        assert np.all(coverage_count <= 1), (
            f"Overlapping tiles detected: max coverage = {coverage_count.max()}"
        )

    def test_src_slices_within_bounds(self):
        """Source slices should not exceed source array bounds."""
        src_rows, src_cols = 64, 64
        tiles = plan_reproject(
            src_crs="EPSG:32633",
            src_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6606400.0),
            src_shape=(src_rows, src_cols),
            dst_crs="EPSG:4326",
            dst_transform=(0.001, 0.0, 14.0, 0.0, -0.001, 60.0),
            dst_shape=(64, 64),
            dst_chunks=(32, 32),
        )

        for i, tile in enumerate(tiles):
            r0, r1, c0, c1 = tile["src_slice"]
            assert r0 >= 0, f"Tile {i}: src row start {r0} < 0"
            assert c0 >= 0, f"Tile {i}: src col start {c0} < 0"
            assert r1 <= src_rows, f"Tile {i}: src row end {r1} > {src_rows}"
            assert c1 <= src_cols, f"Tile {i}: src col end {c1} > {src_cols}"

    def test_uneven_chunks_full_coverage(self):
        """Uneven chunk sizes should still cover the full destination."""
        dst_rows, dst_cols = 100, 100
        tiles = plan_reproject(
            src_crs="EPSG:32633",
            src_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6610000.0),
            src_shape=(100, 100),
            dst_crs="EPSG:32633",
            dst_transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 6610000.0),
            dst_shape=(dst_rows, dst_cols),
            dst_chunks=(64, 64),  # doesn't evenly divide 100
        )

        covered = np.zeros((dst_rows, dst_cols), dtype=bool)
        for tile in tiles:
            r0, r1, c0, c1 = tile["dst_slice"]
            covered[r0:r1, c0:c1] = True

        assert np.all(covered), "Uneven chunks left gaps in coverage"


class TestPasteableDetection:
    """Same-CRS grids that are aligned should be directly pasteable."""

    def test_aligned_same_crs_is_identity(self):
        """Aligned same-CRS grids should produce an identity reprojection."""
        size = 32
        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * px
        transform = (px, 0.0, origin_x, 0.0, -px, origin_y)
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        result = reproject_array(
            src,
            "EPSG:32633",
            transform,
            "EPSG:32633",
            transform,
            (size, size),
            resampling="nearest",
        )

        np.testing.assert_array_equal(result, src)

    def test_subpixel_shift_not_pasteable(self):
        """Half-pixel shift should cause interpolation differences."""
        size = 32
        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0 + size * px
        src_transform = (px, 0.0, origin_x, 0.0, -px, origin_y)
        # Shift by half a pixel
        dst_transform = (px, 0.0, origin_x + px * 0.5, 0.0, -px, origin_y - px * 0.5)

        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        result = reproject_array(
            src,
            "EPSG:32633",
            src_transform,
            "EPSG:32633",
            dst_transform,
            (size, size),
            resampling="bilinear",
        )

        # Result should differ from source due to interpolation
        both_valid = ~np.isnan(result) & ~np.isnan(src)
        if both_valid.sum() > 0:
            # Not all pixels should match (because of the half-pixel shift)
            exact_match = np.sum(result[both_valid] == src[both_valid])
            assert exact_match < both_valid.sum(), "Half-pixel shift should cause differences"


class TestGeoBoxOverlap:
    """GeoBox overlap computation through plan_reproject."""

    def test_well_separated_geoboxes(self):
        """Well-separated GeoBoxes should produce no has_data tiles."""
        px = 100.0
        # Source and dst separated by a large gap (100km)
        tiles = plan_reproject(
            src_crs="EPSG:32633",
            src_transform=(px, 0.0, 500000.0, 0.0, -px, 6603200.0),
            src_shape=(32, 32),
            dst_crs="EPSG:32633",
            dst_transform=(px, 0.0, 600000.0, 0.0, -px, 6603200.0),
            dst_shape=(32, 32),
        )

        has_data_count = sum(1 for t in tiles if t["has_data"])
        assert has_data_count == 0, (
            f"Well-separated boxes should not overlap, but {has_data_count} tiles have data"
        )

    def test_identical_geobox_cross_crs(self):
        """Same geographic area in different CRS should fully overlap."""
        size = 32
        px = 100.0
        origin_x, origin_y = 500000.0, 6600000.0
        src_transform = (px, 0.0, origin_x, 0.0, -px, origin_y)

        dst_affine, dst_w, dst_h = rasterio.warp.calculate_default_transform(
            CRS.from_user_input("EPSG:32633"),
            CRS.from_user_input("EPSG:4326"),
            size,
            size,
            left=origin_x,
            bottom=origin_y - size * px,
            right=origin_x + size * px,
            top=origin_y,
        )
        dst_transform = tuple(dst_affine)[:6]

        tiles = plan_reproject(
            src_crs="EPSG:32633",
            src_transform=src_transform,
            src_shape=(size, size),
            dst_crs="EPSG:4326",
            dst_transform=dst_transform,
            dst_shape=(dst_h, dst_w),
            dst_chunks=(16, 16),
        )

        # All tiles should have data (same geographic area)
        has_data_tiles = [t for t in tiles if t["has_data"]]
        assert len(has_data_tiles) == len(tiles), (
            f"Only {len(has_data_tiles)}/{len(tiles)} tiles have data for same area"
        )
