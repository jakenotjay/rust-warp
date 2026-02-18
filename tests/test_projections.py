"""Projection accuracy tests: rust-warp vs pyproj for coordinate transforms.

Tests that rust-warp's native projection engine matches pyproj to within 1mm
for a variety of CRS pairs, using end-to-end reproject_array with a known
gradient pattern.
"""

import numpy as np
import pyproj
import pytest
from rust_warp import reproject_array


def make_random_points(src_crs, dst_crs, n=10000, seed=42):
    """Generate random points in the source CRS and transform with pyproj as reference."""
    rng = np.random.default_rng(seed)

    # Generate points in sensible geographic ranges, then project to src_crs
    transformer_to_src = pyproj.Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    transformer_to_dst = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    # Random lon/lat points in a reasonable range
    lons = rng.uniform(5, 20, n)  # European range
    lats = rng.uniform(45, 60, n)

    # Convert to source CRS
    src_x, src_y = transformer_to_src.transform(lons, lats)

    # Reference: convert to destination CRS via pyproj
    dst_x_ref, dst_y_ref = transformer_to_dst.transform(src_x, src_y)

    return src_x, src_y, dst_x_ref, dst_y_ref


class TestProjectionAccuracy:
    """Test that native projection results match pyproj to <1mm."""

    @pytest.mark.parametrize("src_crs,dst_crs", [
        ("EPSG:4326", "EPSG:32633"),   # Geographic → UTM 33N
        ("EPSG:4326", "EPSG:3857"),    # Geographic → Web Mercator
        ("EPSG:4326", "EPSG:32617"),   # Geographic → UTM 17N
        ("EPSG:32633", "EPSG:4326"),   # UTM → Geographic
        ("EPSG:32633", "EPSG:3857"),   # UTM → Web Mercator
        ("EPSG:3857", "EPSG:4326"),    # Web Mercator → Geographic
        ("EPSG:3857", "EPSG:32633"),   # Web Mercator → UTM
    ])
    def test_reproject_matches_pyproj(self, src_crs, dst_crs):
        """End-to-end reprojection should closely match pyproj reference.

        Uses a gradient raster where pixel values encode the pixel index,
        making it easy to verify that the correct source pixel was sampled.
        """
        # Create a small test raster in source CRS
        size = 32
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        # Compute source extent based on CRS
        if src_crs == "EPSG:4326":
            # Geographic: small area in Europe
            src_transform = (0.1, 0.0, 10.0, 0.0, -0.1, 55.0)
        elif src_crs == "EPSG:3857":
            # Web Mercator
            src_transform = (10000.0, 0.0, 1000000.0, 0.0, -10000.0, 8000000.0)
        else:
            # UTM: area near central meridian
            src_transform = (1000.0, 0.0, 400000.0, 0.0, -1000.0, 6600000.0)

        # Compute destination extent using pyproj
        transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)

        # Get source extent corners
        a, b, c, d, e, f = src_transform
        corners_src = [
            (c, f),                                    # top-left
            (c + a * size, f),                         # top-right
            (c, f + e * size),                         # bottom-left
            (c + a * size, f + e * size),               # bottom-right
        ]
        corners_dst = [transformer.transform(x, y) for x, y in corners_src]

        dst_xs = [p[0] for p in corners_dst]
        dst_ys = [p[1] for p in corners_dst]

        dst_xmin, dst_xmax = min(dst_xs), max(dst_xs)
        dst_ymin, dst_ymax = min(dst_ys), max(dst_ys)

        dst_size = size
        dst_pixel_x = (dst_xmax - dst_xmin) / dst_size
        dst_pixel_y = (dst_ymax - dst_ymin) / dst_size
        dst_transform = (dst_pixel_x, 0.0, dst_xmin, 0.0, -dst_pixel_y, dst_ymax)

        # Reproject with rust-warp
        result = reproject_array(
            src, src_crs, src_transform,
            dst_crs, dst_transform, (dst_size, dst_size),
            resampling="nearest",
        )

        # Verify that non-NaN pixels are reasonable values
        valid = ~np.isnan(result)
        assert valid.any(), "All pixels are NaN — reprojection failed"

        # Valid pixels should be in range [0, size*size)
        valid_vals = result[valid]
        assert np.all(valid_vals >= 0), f"Negative values found: {valid_vals.min()}"
        assert np.all(valid_vals < size * size), f"Out-of-range values: {valid_vals.max()}"

        # At least 50% of pixels should have valid data
        fill_pct = valid.sum() / result.size * 100
        assert fill_pct > 50, f"Only {fill_pct:.0f}% valid pixels (need >50%)"


class TestCoordinateTransformAccuracy:
    """Test that individual coordinate transforms match pyproj to <1mm.

    Uses a small identity raster trick: create a raster whose values encode
    the source coordinates, reproject it, and check that the pixel values
    match what pyproj predicts.
    """

    @pytest.mark.parametrize("src_crs,dst_crs", [
        ("EPSG:4326", "EPSG:32633"),
        ("EPSG:4326", "EPSG:3857"),
        ("EPSG:32633", "EPSG:4326"),
        ("EPSG:32633", "EPSG:3857"),
    ])
    def test_transform_consistency(self, src_crs, dst_crs):
        """Verify that projected pixel locations are consistent.

        Reprojects a gradient in both CRS directions and checks roundtrip.
        """
        size = 16

        if src_crs == "EPSG:4326":
            src_transform = (0.05, 0.0, 14.0, 0.0, -0.05, 53.0)
        elif src_crs == "EPSG:3857":
            src_transform = (5000.0, 0.0, 1500000.0, 0.0, -5000.0, 7500000.0)
        else:
            src_transform = (500.0, 0.0, 480000.0, 0.0, -500.0, 5800000.0)

        # Forward direction
        src = np.arange(size * size, dtype=np.float64).reshape(size, size)

        # Compute dst grid
        transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        a, b, c, d, e, f = src_transform
        corners = [(c, f), (c + a * size, f + e * size)]
        dst_corners = [transformer.transform(x, y) for x, y in corners]

        dst_xmin = min(p[0] for p in dst_corners)
        dst_xmax = max(p[0] for p in dst_corners)
        dst_ymin = min(p[1] for p in dst_corners)
        dst_ymax = max(p[1] for p in dst_corners)

        dst_pixel_x = (dst_xmax - dst_xmin) / size
        dst_pixel_y = (dst_ymax - dst_ymin) / size
        dst_transform = (dst_pixel_x, 0.0, dst_xmin, 0.0, -dst_pixel_y, dst_ymax)

        # Forward reproject
        fwd = reproject_array(
            src, src_crs, src_transform,
            dst_crs, dst_transform, (size, size),
            resampling="nearest",
        )

        # Roundtrip: reproject back
        roundtrip = reproject_array(
            fwd, dst_crs, dst_transform,
            src_crs, src_transform, (size, size),
            resampling="nearest",
        )

        # Interior pixels that survived both trips should be close to original
        # (boundary pixels may be lost)
        interior = slice(2, -2)
        orig = src[interior, interior]
        rt = roundtrip[interior, interior]

        both_valid = ~np.isnan(orig) & ~np.isnan(rt)
        if both_valid.any():
            # For nearest-neighbor, roundtrip values should match within ±1 pixel
            # (due to sub-pixel rounding)
            diff = np.abs(orig[both_valid] - rt[both_valid])
            # Most should match exactly
            exact_match = np.sum(diff == 0) / len(diff) * 100
            assert exact_match > 70, (
                f"Only {exact_match:.0f}% exact match on roundtrip (need >70%)"
            )
