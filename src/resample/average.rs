//! Area-weighted average resampling kernel.
//!
//! Computes a weighted average over the source pixel footprint, skipping
//! nodata/NaN pixels (matching GDAL `GRA_Average` behavior).

use ndarray::ArrayView2;
use num_traits::NumCast;

use super::is_nodata_value;

/// Sample a 2D array using area-weighted averaging.
///
/// The `scale` parameter is the source-to-destination pixel ratio `(sx, sy)`,
/// computed from the affine transforms. The averaging window half-width is
/// `max(scale/2, 0.5)` in each axis.
///
/// Unlike cubic/lanczos, nodata/NaN pixels are **skipped** rather than
/// propagating — the average is computed over remaining valid pixels.
/// Returns `None` only if ALL pixels in the footprint are nodata.
pub fn sample<T>(
    src: &ArrayView2<'_, T>,
    x: f64,
    y: f64,
    nodata: Option<T>,
    scale: (f64, f64),
) -> Option<T>
where
    T: Copy + NumCast + PartialEq,
{
    // Use corner-based coordinates directly. The mapped source coordinate
    // (x, y) already represents the destination pixel center in source pixel
    // space. The footprint is centered at this point.
    let cx = x;
    let cy = y;

    // Window half-width: at least 0.5 (single pixel) in each axis
    let hx = (scale.0 / 2.0).max(0.5);
    let hy = (scale.1 / 2.0).max(0.5);

    let (rows, cols) = (src.nrows() as isize, src.ncols() as isize);

    // Integer range of source pixels covered by the footprint
    let x_min = (cx - hx).floor() as isize;
    let x_max = (cx + hx).ceil() as isize;
    let y_min = (cy - hy).floor() as isize;
    let y_max = (cy + hy).ceil() as isize;

    // Clamp to source bounds
    let x_min = x_min.max(0);
    let x_max = x_max.min(cols);
    let y_min = y_min.max(0);
    let y_max = y_max.min(rows);

    if x_min >= x_max || y_min >= y_max {
        return None;
    }

    // Precompute y-overlap weights to avoid redundant max/min in inner loop
    let oy_weights: Vec<f64> = (y_min..y_max)
        .map(|iy| {
            let lo = (iy as f64).max(cy - hy);
            let hi = ((iy + 1) as f64).min(cy + hy);
            (hi - lo).max(0.0)
        })
        .collect();

    let mut weighted_sum = 0.0_f64;
    let mut total_weight = 0.0_f64;

    for (yi, iy) in (y_min..y_max).enumerate() {
        let oy = oy_weights[yi];

        for ix in x_min..x_max {
            let val = src[(iy as usize, ix as usize)];
            if is_nodata_value(val, nodata) {
                continue; // skip nodata — don't propagate
            }

            let fval: f64 = NumCast::from(val)?;

            // Overlap in x: intersection of [ix, ix+1] with [cx-hx, cx+hx]
            let ox_lo = (ix as f64).max(cx - hx);
            let ox_hi = ((ix + 1) as f64).min(cx + hx);
            let ox = (ox_hi - ox_lo).max(0.0);

            let w = ox * oy;
            weighted_sum += w * fval;
            total_weight += w;
        }
    }

    if total_weight < 1e-15 {
        return None; // all pixels in footprint were nodata
    }

    NumCast::from(weighted_sum / total_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    #[test]
    fn test_uniform_downscale_2x() {
        // 4×4 source of all 10.0, scale=(2,2) → should average to 10.0
        let arr = Array2::from_elem((4, 4), 10.0_f64);
        let view = arr.view();

        let val = sample(&view, 1.5, 1.5, None, (2.0, 2.0)).unwrap();
        assert_relative_eq!(val, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gradient_downscale() {
        // 4×4 source with horizontal gradient: col values 0,1,2,3
        let mut arr = Array2::zeros((4, 4));
        for r in 0..4 {
            for c in 0..4 {
                arr[(r, c)] = c as f64;
            }
        }
        let view = arr.view();

        // Center of the array, scale=1 → single pixel average
        let val = sample(&view, 2.5, 2.5, None, (1.0, 1.0)).unwrap();
        assert_relative_eq!(val, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nodata_skipping() {
        let mut arr = Array2::from_elem((4, 4), 10.0);
        arr[(1, 1)] = -9999.0;
        let view = arr.view();

        // Average with scale=2 centered on area including nodata pixel
        let val = sample(&view, 1.5, 1.5, Some(-9999.0), (2.0, 2.0)).unwrap();
        // nodata pixel skipped, rest are 10.0
        assert_relative_eq!(val, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nan_skipping() {
        let mut arr = Array2::from_elem((4, 4), 10.0_f64);
        arr[(1, 1)] = f64::NAN;
        let view = arr.view();

        let val = sample(&view, 1.5, 1.5, None, (2.0, 2.0)).unwrap();
        assert_relative_eq!(val, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_all_nodata() {
        let arr = Array2::from_elem((4, 4), -9999.0);
        let view = arr.view();

        let val = sample(&view, 1.5, 1.5, Some(-9999.0), (2.0, 2.0));
        assert!(val.is_none());
    }

    #[test]
    fn test_all_nan() {
        let arr = Array2::from_elem((4, 4), f64::NAN);
        let view = arr.view();

        let val = sample::<f64>(&view, 1.5, 1.5, None, (2.0, 2.0));
        assert!(val.is_none());
    }

    #[test]
    fn test_scale_1_at_pixel_center() {
        // Scale=1 at a pixel center should return that pixel's value
        let mut arr = Array2::zeros((6, 6));
        for r in 0..6 {
            for c in 0..6 {
                arr[(r, c)] = (r * 6 + c) as f64;
            }
        }
        let view = arr.view();

        let val = sample(&view, 3.5, 3.5, None, (1.0, 1.0)).unwrap();
        assert_relative_eq!(val, arr[(3, 3)], epsilon = 1e-10);
    }

    #[test]
    fn test_area_weighting() {
        // 2×2 source: [[0, 100], [0, 100]]
        // Scale=2 centered at (1.0, 1.0) should average all 4 pixels
        let arr = ndarray::array![[0.0, 100.0], [0.0, 100.0]];
        let view = arr.view();

        let val = sample(&view, 1.0, 1.0, None, (2.0, 2.0)).unwrap();
        assert_relative_eq!(val, 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_out_of_bounds_returns_none() {
        let arr = Array2::from_elem((4, 4), 1.0_f64);
        let view = arr.view();

        // Completely outside
        assert!(sample::<f64>(&view, -5.0, -5.0, None, (1.0, 1.0)).is_none());
        assert!(sample::<f64>(&view, 10.0, 10.0, None, (1.0, 1.0)).is_none());
    }

    #[test]
    fn test_integer_type() {
        let arr = Array2::from_elem((4, 4), 42_i32);
        let view = arr.view();

        let val = sample(&view, 2.5, 2.5, None, (2.0, 2.0)).unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn test_known_2x_downscale_average() {
        // 4×4 with known values, 2× downscale: each 2×2 block averages exactly
        let arr = ndarray::array![
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0],
            [90.0, 100.0, 110.0, 120.0],
            [130.0, 140.0, 150.0, 160.0],
        ];
        let view = arr.view();

        // Center of 2×2 block (0,0)-(1,1) is at (1.0, 1.0), scale=2
        let val = sample(&view, 1.0, 1.0, None, (2.0, 2.0)).unwrap();
        let expected = (10.0 + 20.0 + 50.0 + 60.0) / 4.0; // = 35.0
        assert_relative_eq!(val, expected, epsilon = 1e-10);

        // Center of 2×2 block (2,2)-(3,3) is at (3.0, 3.0), scale=2
        let val = sample(&view, 3.0, 3.0, None, (2.0, 2.0)).unwrap();
        let expected = (110.0 + 120.0 + 150.0 + 160.0) / 4.0; // = 135.0
        assert_relative_eq!(val, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_3x_downscale_with_nodata() {
        // 6×6 with one nodata pixel in a 3×3 block
        let mut arr = Array2::from_elem((6, 6), 100.0_f64);
        arr[(1, 1)] = -9999.0; // nodata in first 3×3 block
        let view = arr.view();

        // 3× downscale, center of first 3×3 block
        let val = sample(&view, 1.5, 1.5, Some(-9999.0), (3.0, 3.0)).unwrap();
        // Should average remaining 8 pixels (all 100.0)
        assert_relative_eq!(val, 100.0, epsilon = 1e-10);
    }
}
