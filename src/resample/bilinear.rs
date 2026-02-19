//! Bilinear interpolation resampling kernel.

use ndarray::ArrayView2;
use num_traits::NumCast;

/// Sample a 2D array using bilinear interpolation.
///
/// Subtracts 0.5 from input coordinates to center on pixel centers
/// (GDAL convention: pixel center at col+0.5, row+0.5).
///
/// Performs 2×2 weighted interpolation. At edges, evaluates only the
/// available pixels and renormalizes weights (matching GDAL behavior).
/// Returns `None` if no valid pixels contribute.
pub fn sample<T>(src: &ArrayView2<'_, T>, x: f64, y: f64, nodata: Option<T>) -> Option<T>
where
    T: Copy + NumCast + PartialEq,
{
    // Convert from corner-based to center-based coordinates
    let cx = x - 0.5;
    let cy = y - 0.5;

    let x0 = cx.floor() as isize;
    let y0 = cy.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let (rows, cols) = (src.nrows() as isize, src.ncols() as isize);

    // Fast path: all 4 neighbors in bounds
    if x0 >= 0 && x1 < cols && y0 >= 0 && y1 < rows {
        let (x0u, y0u, x1u, y1u) = (x0 as usize, y0 as usize, x1 as usize, y1 as usize);
        let v00 = src[(y0u, x0u)];
        let v10 = src[(y0u, x1u)];
        let v01 = src[(y1u, x0u)];
        let v11 = src[(y1u, x1u)];

        if let Some(nd) = nodata {
            if v00 == nd || v10 == nd || v01 == nd || v11 == nd {
                return None;
            }
        }

        let f00: f64 = NumCast::from(v00)?;
        let f10: f64 = NumCast::from(v10)?;
        let f01: f64 = NumCast::from(v01)?;
        let f11: f64 = NumCast::from(v11)?;

        if f00.is_nan() || f10.is_nan() || f01.is_nan() || f11.is_nan() {
            return None;
        }

        let dx = cx - x0 as f64;
        let dy = cy - y0 as f64;

        let result = f00 * (1.0 - dx) * (1.0 - dy)
            + f10 * dx * (1.0 - dy)
            + f01 * (1.0 - dx) * dy
            + f11 * dx * dy;

        return NumCast::from(result);
    }

    // Edge path: partial kernel evaluation with weight renormalization.
    // Matches GDAL's GWKBilinearResample4Sample behavior.
    let dx = cx - x0 as f64;
    let dy = cy - y0 as f64;

    // Weights for the 4 pixels: (x0,y0), (x1,y0), (x0,y1), (x1,y1)
    let weights = [
        (1.0 - dx) * (1.0 - dy),
        dx * (1.0 - dy),
        (1.0 - dx) * dy,
        dx * dy,
    ];
    let coords: [(isize, isize); 4] = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];

    let mut accum = 0.0_f64;
    let mut weight_sum = 0.0_f64;

    for (k, &(cx_k, cy_k)) in coords.iter().enumerate() {
        if cx_k < 0 || cx_k >= cols || cy_k < 0 || cy_k >= rows {
            continue;
        }
        let val = src[(cy_k as usize, cx_k as usize)];

        if let Some(nd) = nodata {
            if val == nd {
                continue;
            }
        }

        let fval: f64 = match NumCast::from(val) {
            Some(f) => f,
            None => continue,
        };
        if fval.is_nan() {
            continue;
        }

        accum += weights[k] * fval;
        weight_sum += weights[k];
    }

    if weight_sum < 1e-10 {
        return None;
    }

    NumCast::from(accum / weight_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_pixel_center_exact() {
        // 3×3 grid, sampling exactly at pixel centers should return exact values
        let arr = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];
        let view = arr.view();

        // Pixel center of (1,1) is at corner-based (1.5, 1.5)
        let val = sample(&view, 1.5, 1.5, None).unwrap();
        assert_relative_eq!(val, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_midpoint_interpolation() {
        let arr = array![[0.0, 10.0], [0.0, 10.0],];
        let view = arr.view();

        // Midpoint between pixel centers (0.5,0.5) and (1.5,0.5) is at (1.0, 0.5)
        let val = sample(&view, 1.0, 0.5, None).unwrap();
        assert_relative_eq!(val, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_out_of_bounds() {
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let view = arr.view();

        // At edges, bilinear now does partial evaluation (matching GDAL).
        // x=0.0 is at the left edge — partial kernel uses available pixels.
        assert!(sample::<f64>(&view, 0.0, 0.5, None).is_some());
        // x=2.0 is at the right edge — partial kernel
        assert!(sample::<f64>(&view, 2.0, 0.5, None).is_some());
        // Well outside — None
        assert!(sample::<f64>(&view, -1.0, 0.5, None).is_none());
        assert!(sample::<f64>(&view, 3.0, 0.5, None).is_none());
    }

    #[test]
    fn test_nan_propagation() {
        let arr = array![[1.0, f64::NAN], [3.0, 4.0],];
        let view = arr.view();

        // Any NaN neighbor → None
        let val = sample::<f64>(&view, 1.0, 1.0, None);
        assert!(val.is_none());
    }

    #[test]
    fn test_nodata_propagation() {
        let arr = array![[-9999.0, 2.0], [3.0, 4.0],];
        let view = arr.view();

        let val = sample(&view, 1.0, 1.0, Some(-9999.0));
        assert!(val.is_none());

        // Without nodata flag, interpolation proceeds
        let val = sample(&view, 1.0, 1.0, None).unwrap();
        assert!(val < 0.0); // interpolation includes -9999.0
    }

    #[test]
    fn test_gradient() {
        // Linear gradient: bilinear should reproduce it exactly
        let arr = array![[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0],];
        let view = arr.view();

        // Quarter-way between pixel centers 0 and 1 (x direction)
        let val = sample(&view, 1.0, 1.5, None).unwrap();
        assert_relative_eq!(val, 0.5, epsilon = 1e-10);

        let val = sample(&view, 1.75, 1.5, None).unwrap();
        assert_relative_eq!(val, 1.25, epsilon = 1e-10);
    }

    #[test]
    fn test_linear_gradient_exact_preservation() {
        // Analytical test: bilinear interpolation should EXACTLY reproduce
        // any linear function f(x,y) = ax + by + c
        let a = 3.0_f64;
        let b = -2.0_f64;
        let c = 7.0_f64;

        let mut arr = ndarray::Array2::zeros((10, 10));
        for r in 0..10 {
            for col in 0..10 {
                arr[(r, col)] = a * col as f64 + b * r as f64 + c;
            }
        }
        let view = arr.view();

        // Sample at many sub-pixel positions — all should match f(x,y) exactly
        for row_f in [1.5, 2.0, 3.25, 4.75, 7.5] {
            for col_f in [1.5, 2.0, 3.25, 4.75, 7.5] {
                let expected = a * (col_f - 0.5) + b * (row_f - 0.5) + c;
                let val = sample(&view, col_f, row_f, None).unwrap();
                assert!(
                    (val - expected).abs() < 1e-10,
                    "At ({col_f}, {row_f}): expected {expected}, got {val}"
                );
            }
        }
    }
}
