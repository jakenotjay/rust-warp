//! Cubic convolution resampling kernel (Keys 1981, a = -0.5).
//!
//! Uses a 4×4 neighborhood with the classic Keys weight function.

use ndarray::ArrayView2;
use num_traits::NumCast;

use super::is_nodata_value;

/// Cubic convolution weight function (Keys 1981, a = -0.5).
///
/// ```text
/// W(t) = (a+2)|t|³ - (a+3)|t|² + 1       for 0 ≤ |t| ≤ 1
/// W(t) = a|t|³ - 5a|t|² + 8a|t| - 4a     for 1 < |t| ≤ 2
/// W(t) = 0                                 for |t| > 2
/// ```
fn cubic_weight(t: f64) -> f64 {
    const A: f64 = -0.5;
    let t = t.abs();
    if t <= 1.0 {
        (A + 2.0) * t * t * t - (A + 3.0) * t * t + 1.0
    } else if t <= 2.0 {
        A * t * t * t - 5.0 * A * t * t + 8.0 * A * t - 4.0 * A
    } else {
        0.0
    }
}

/// Sample a 2D array using cubic convolution interpolation.
///
/// Uses a 4×4 neighborhood centered on the sample point.
/// Corner-to-center conversion (-0.5 offset), anchor at `floor()`.
///
/// Returns `None` if any of the 16 neighbors is out of bounds, nodata, or NaN.
pub fn sample<T>(src: &ArrayView2<'_, T>, x: f64, y: f64, nodata: Option<T>) -> Option<T>
where
    T: Copy + NumCast + PartialEq,
{
    // Convert from corner-based to center-based coordinates
    let cx = x - 0.5;
    let cy = y - 0.5;

    let ix = cx.floor() as isize;
    let iy = cy.floor() as isize;

    let (rows, cols) = (src.nrows() as isize, src.ncols() as isize);

    // Check bounds for 4×4 neighborhood: offsets -1..+2
    if ix - 1 < 0 || ix + 2 >= cols || iy - 1 < 0 || iy + 2 >= rows {
        return None;
    }

    let dx = cx - ix as f64;
    let dy = cy - iy as f64;

    // Precompute 1D weight arrays to reduce cubic_weight calls from 16 to 8
    let wx: [f64; 4] = std::array::from_fn(|k| cubic_weight(dx - (k as f64 - 1.0)));
    let wy: [f64; 4] = std::array::from_fn(|k| cubic_weight(dy - (k as f64 - 1.0)));

    let mut result = 0.0;
    for (jk, j) in (-1..=2_isize).enumerate() {
        let w_row = wy[jk];
        for (ik, i) in (-1..=2_isize).enumerate() {
            let val = src[((iy + j) as usize, (ix + i) as usize)];
            if is_nodata_value(val, nodata) {
                return None;
            }

            let fval: f64 = NumCast::from(val)?;
            result += wx[ik] * w_row * fval;
        }
    }

    NumCast::from(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{array, Array2};

    #[test]
    fn test_weight_at_zero() {
        assert_relative_eq!(cubic_weight(0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_weight_at_one() {
        assert_relative_eq!(cubic_weight(1.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_weight_at_two() {
        assert_relative_eq!(cubic_weight(2.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_weight_symmetry() {
        for &t in &[0.3, 0.7, 1.2, 1.8] {
            assert_relative_eq!(cubic_weight(t), cubic_weight(-t), epsilon = 1e-12);
        }
    }

    #[test]
    fn test_partition_of_unity() {
        // For any fractional offset, weights over -1..+2 should sum to 1
        for &dx in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let sum: f64 = (-1..=2).map(|i| cubic_weight(dx - i as f64)).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_pixel_center_exact() {
        // 6×6 grid, sampling at pixel center (3,3) = corner-based (3.5, 3.5)
        let mut arr = Array2::zeros((6, 6));
        for r in 0..6 {
            for c in 0..6 {
                arr[(r, c)] = (r * 6 + c) as f64;
            }
        }
        let view = arr.view();

        let val = sample(&view, 3.5, 3.5, None).unwrap();
        assert_relative_eq!(val, arr[(3, 3)], epsilon = 1e-10);
    }

    #[test]
    fn test_linear_gradient_preservation() {
        // Cubic should reproduce a linear gradient exactly
        let mut arr = Array2::zeros((8, 8));
        for r in 0..8 {
            for c in 0..8 {
                arr[(r, c)] = c as f64; // horizontal gradient
            }
        }
        let view = arr.view();

        // Sample at various sub-pixel x positions, fixed y
        let val = sample(&view, 3.75, 3.5, None).unwrap();
        assert_relative_eq!(val, 3.25, epsilon = 1e-10);

        let val = sample(&view, 4.0, 3.5, None).unwrap();
        assert_relative_eq!(val, 3.5, epsilon = 1e-10);
    }

    #[test]
    fn test_nan_propagation() {
        let mut arr = Array2::from_elem((6, 6), 1.0_f64);
        arr[(3, 3)] = f64::NAN;
        let view = arr.view();

        // Sample near the NaN pixel — it's in the 4×4 neighborhood
        assert!(sample::<f64>(&view, 3.5, 3.5, None).is_none());
    }

    #[test]
    fn test_nodata_propagation() {
        let mut arr = Array2::from_elem((6, 6), 1.0);
        arr[(3, 3)] = -9999.0;
        let view = arr.view();

        assert!(sample(&view, 3.5, 3.5, Some(-9999.0)).is_none());
        // Without nodata, it's treated as a valid value
        assert!(sample(&view, 3.5, 3.5, None).is_some());
    }

    #[test]
    fn test_out_of_bounds() {
        let arr = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let view = arr.view();

        // 4×4 grid with cubic (radius 2) — no interior pixel can be sampled
        // because the 4×4 neighborhood always reaches out of bounds
        assert!(sample::<f64>(&view, 0.5, 0.5, None).is_none());
        assert!(sample::<f64>(&view, 2.5, 2.5, None).is_none());
    }

    #[test]
    fn test_integer_type() {
        let mut arr = Array2::zeros((6, 6));
        for r in 0..6_i32 {
            for c in 0..6_i32 {
                arr[(r as usize, c as usize)] = r * 6 + c;
            }
        }
        let view = arr.view();

        // Pixel center — should return the exact integer value
        let val = sample(&view, 3.5, 3.5, None::<i32>).unwrap();
        assert_eq!(val, 21); // 3*6+3 = 21
    }

    #[test]
    fn test_quadratic_surface_preservation() {
        // Cubic convolution (Keys, a=-0.5) exactly reproduces polynomials up to degree 3.
        // Test with quadratic: f(x,y) = 2x² + 3xy - y² + 5x - 2y + 10
        let mut arr = Array2::zeros((12, 12));
        for r in 0..12 {
            for c in 0..12 {
                let x = c as f64;
                let y = r as f64;
                arr[(r, c)] = 2.0 * x * x + 3.0 * x * y - y * y + 5.0 * x - 2.0 * y + 10.0;
            }
        }
        let view = arr.view();

        // Sample at sub-pixel positions (center-based coords)
        for &row_f in &[3.5, 4.25, 5.0, 6.75, 8.5] {
            for &col_f in &[3.5, 4.25, 5.0, 6.75, 8.5] {
                let x = col_f - 0.5;
                let y = row_f - 0.5;
                let expected = 2.0 * x * x + 3.0 * x * y - y * y + 5.0 * x - 2.0 * y + 10.0;
                let val = sample(&view, col_f, row_f, None).unwrap();
                assert!(
                    (val - expected).abs() < 1e-6,
                    "At ({col_f}, {row_f}): expected {expected}, got {val}"
                );
            }
        }
    }

    #[test]
    fn test_weight_partition_of_unity_2d() {
        // For the 2D case, product of 1D weights should still sum to 1
        for &dx in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9] {
            for &dy in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9] {
                let wx: [f64; 4] = std::array::from_fn(|k| cubic_weight(dx - (k as f64 - 1.0)));
                let wy: [f64; 4] = std::array::from_fn(|k| cubic_weight(dy - (k as f64 - 1.0)));
                let sum_2d: f64 = wx
                    .iter()
                    .flat_map(|&wx_i| wy.iter().map(move |&wy_j| wx_i * wy_j))
                    .sum();
                assert!(
                    (sum_2d - 1.0).abs() < 1e-12,
                    "2D weight sum at ({dx}, {dy}): {sum_2d}"
                );
            }
        }
    }
}
