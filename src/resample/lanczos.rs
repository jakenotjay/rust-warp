//! Lanczos sinc-windowed resampling kernel (a = 3).
//!
//! Uses a 6×6 neighborhood with normalized sinc weights.

use ndarray::ArrayView2;
use num_traits::NumCast;

use super::is_nodata_value;

/// Normalized sinc function: sinc(x) = sin(πx) / (πx), sinc(0) = 1.
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Lanczos weight function: L(t) = sinc(t) * sinc(t/a) for |t| < a, else 0.
fn lanczos_weight(t: f64) -> f64 {
    const A: f64 = 3.0;
    let t = t.abs();
    if t < A {
        sinc(t) * sinc(t / A)
    } else {
        0.0
    }
}

/// Sample a 2D array using Lanczos interpolation (a=3).
///
/// Uses a 6×6 neighborhood centered on the sample point.
/// Corner-to-center conversion (-0.5 offset), anchor at `floor()`.
/// Weights are normalized to sum to 1.0.
///
/// Returns `None` if any of the 36 neighbors is out of bounds, nodata, or NaN.
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

    // Check bounds for 6×6 neighborhood: offsets -2..+3
    if ix - 2 < 0 || ix + 3 >= cols || iy - 2 < 0 || iy + 3 >= rows {
        return None;
    }

    let dx = cx - ix as f64;
    let dy = cy - iy as f64;

    // Compute 1D weights and normalize
    let mut wx = [0.0_f64; 6];
    let mut wy = [0.0_f64; 6];
    for (k, offset) in (-2..=3_isize).enumerate() {
        wx[k] = lanczos_weight(dx - offset as f64);
        wy[k] = lanczos_weight(dy - offset as f64);
    }

    let sum_wx: f64 = wx.iter().sum();
    let sum_wy: f64 = wy.iter().sum();
    if sum_wx.abs() < 1e-15 || sum_wy.abs() < 1e-15 {
        return None;
    }
    for w in &mut wx {
        *w /= sum_wx;
    }
    for w in &mut wy {
        *w /= sum_wy;
    }

    let mut result = 0.0;
    for (jk, j) in (-2..=3_isize).enumerate() {
        for (ik, i) in (-2..=3_isize).enumerate() {
            let val = src[((iy + j) as usize, (ix + i) as usize)];
            if is_nodata_value(val, nodata) {
                return None;
            }

            let fval: f64 = NumCast::from(val)?;
            result += wx[ik] * wy[jk] * fval;
        }
    }

    NumCast::from(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    #[test]
    fn test_sinc_at_zero() {
        assert_relative_eq!(sinc(0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_sinc_at_integers() {
        // sinc(n) = 0 for non-zero integers
        for n in 1..=5 {
            assert_relative_eq!(sinc(n as f64), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_lanczos_weight_at_zero() {
        assert_relative_eq!(lanczos_weight(0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lanczos_weight_at_boundary() {
        assert_relative_eq!(lanczos_weight(3.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(lanczos_weight(-3.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lanczos_weight_beyond_boundary() {
        assert_relative_eq!(lanczos_weight(3.5), 0.0, epsilon = 1e-12);
        assert_relative_eq!(lanczos_weight(100.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lanczos_weight_symmetry() {
        for &t in &[0.3, 1.0, 1.5, 2.7] {
            assert_relative_eq!(lanczos_weight(t), lanczos_weight(-t), epsilon = 1e-12);
        }
    }

    #[test]
    fn test_pixel_center_exact() {
        // 10×10 grid, sampling at pixel center (5,5) = corner-based (5.5, 5.5)
        let mut arr = Array2::zeros((10, 10));
        for r in 0..10 {
            for c in 0..10 {
                arr[(r, c)] = (r * 10 + c) as f64;
            }
        }
        let view = arr.view();

        let val = sample(&view, 5.5, 5.5, None).unwrap();
        assert_relative_eq!(val, arr[(5, 5)], epsilon = 1e-10);
    }

    #[test]
    fn test_linear_gradient_close() {
        // Lanczos on a linear gradient should be very close (slight ringing
        // from asymmetric 6-point support is expected).
        let mut arr = Array2::zeros((10, 10));
        for r in 0..10 {
            for c in 0..10 {
                arr[(r, c)] = c as f64;
            }
        }
        let view = arr.view();

        let val = sample(&view, 4.75, 5.5, None).unwrap();
        assert_relative_eq!(val, 4.25, epsilon = 0.05);

        let val = sample(&view, 5.0, 5.5, None).unwrap();
        assert_relative_eq!(val, 4.5, epsilon = 0.05);
    }

    #[test]
    fn test_nan_propagation() {
        let mut arr = Array2::from_elem((10, 10), 1.0_f64);
        arr[(5, 5)] = f64::NAN;
        let view = arr.view();

        assert!(sample::<f64>(&view, 5.5, 5.5, None).is_none());
    }

    #[test]
    fn test_nodata_propagation() {
        let mut arr = Array2::from_elem((10, 10), 1.0);
        arr[(5, 5)] = -9999.0;
        let view = arr.view();

        assert!(sample(&view, 5.5, 5.5, Some(-9999.0)).is_none());
        assert!(sample(&view, 5.5, 5.5, None).is_some());
    }

    #[test]
    fn test_out_of_bounds() {
        // 6×6 grid — no valid interior with lanczos radius 3
        let arr = Array2::from_elem((6, 6), 1.0_f64);
        let view = arr.view();

        assert!(sample::<f64>(&view, 0.5, 0.5, None).is_none());
        assert!(sample::<f64>(&view, 3.5, 3.5, None).is_none());
    }

    #[test]
    fn test_valid_in_large_array() {
        // 12×12 grid — interior pixels should work
        let mut arr = Array2::zeros((12, 12));
        for r in 0..12 {
            for c in 0..12 {
                arr[(r, c)] = (r * 12 + c) as f64;
            }
        }
        let view = arr.view();

        // Center of the array
        let val = sample(&view, 6.5, 6.5, None);
        assert!(val.is_some());
    }

    #[test]
    fn test_integer_type() {
        let mut arr = Array2::zeros((10, 10));
        for r in 0..10_i32 {
            for c in 0..10_i32 {
                arr[(r as usize, c as usize)] = r * 10 + c;
            }
        }
        let view = arr.view();

        let val = sample(&view, 5.5, 5.5, None::<i32>).unwrap();
        assert_eq!(val, 55); // 5*10+5 = 55
    }
}
