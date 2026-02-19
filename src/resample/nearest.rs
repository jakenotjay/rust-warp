//! Nearest-neighbor resampling kernel.

use ndarray::ArrayView2;
use num_traits::NumCast;

/// Sample a 2D array using nearest-neighbor interpolation.
///
/// Uses GDAL corner-based pixel convention: pixel (0,0) has its upper-left
/// corner at coordinate (0.0, 0.0) and its center at (0.5, 0.5).
/// Nearest-neighbor simply uses `floor()` to find the containing pixel.
///
/// Returns `None` if the coordinate is outside the array bounds or if
/// the sampled value equals `nodata`.
pub fn sample<T>(src: &ArrayView2<'_, T>, x: f64, y: f64, nodata: Option<T>) -> Option<T>
where
    T: Copy + NumCast + PartialEq,
{
    let col = x.floor() as isize;
    let row = y.floor() as isize;

    let (rows, cols) = (src.nrows() as isize, src.ncols() as isize);
    if col < 0 || col >= cols || row < 0 || row >= rows {
        return None;
    }

    let val = src[(row as usize, col as usize)];

    if let Some(nd) = nodata {
        if val == nd {
            return None;
        }
    }

    Some(val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_center_of_pixel() {
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let view = arr.view();

        // Center of pixel (0,0) is at (0.5, 0.5) -> should get value 1.0
        assert_eq!(sample(&view, 0.5, 0.5, None), Some(1.0));
        // Center of pixel (1,0) is at (1.5, 0.5) -> should get value 2.0
        assert_eq!(sample(&view, 1.5, 0.5, None), Some(2.0));
        // Center of pixel (0,1) is at (0.5, 1.5) -> should get value 3.0
        assert_eq!(sample(&view, 0.5, 1.5, None), Some(3.0));
        // Center of pixel (1,1) is at (1.5, 1.5) -> should get value 4.0
        assert_eq!(sample(&view, 1.5, 1.5, None), Some(4.0));
    }

    #[test]
    fn test_upper_left_corner() {
        let arr = array![[10.0, 20.0], [30.0, 40.0]];
        let view = arr.view();

        // Upper-left corner of pixel (0,0) is at (0.0, 0.0)
        assert_eq!(sample(&view, 0.0, 0.0, None), Some(10.0));
    }

    #[test]
    fn test_out_of_bounds() {
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let view = arr.view();

        assert_eq!(sample::<f64>(&view, -0.1, 0.5, None), None);
        assert_eq!(sample::<f64>(&view, 0.5, -0.1, None), None);
        assert_eq!(sample::<f64>(&view, 2.0, 0.5, None), None);
        assert_eq!(sample::<f64>(&view, 0.5, 2.0, None), None);
    }

    #[test]
    fn test_nodata() {
        // Use a sentinel value for nodata (NaN != NaN so equality check won't match)
        let arr2 = array![[-9999.0, 2.0], [3.0, 4.0]];
        let view2 = arr2.view();
        assert_eq!(sample(&view2, 0.5, 0.5, Some(-9999.0)), None);
        assert_eq!(sample(&view2, 1.5, 0.5, Some(-9999.0)), Some(2.0));

        // Without nodata, the sentinel is returned as a valid value
        assert_eq!(sample(&view2, 0.5, 0.5, None), Some(-9999.0));
    }

    #[test]
    fn test_integer_type() {
        let arr = ndarray::array![[1i32, 2], [3, 4]];
        let view = arr.view();
        assert_eq!(sample(&view, 0.5, 0.5, None), Some(1));
        assert_eq!(sample(&view, 1.5, 1.5, None), Some(4));
        assert_eq!(sample(&view, 0.5, 0.5, Some(1)), None);
    }

    #[test]
    fn test_subpixel_offsets_select_correct_pixel() {
        // Analytical test: for a grid where pixel(r,c) = r*10 + c,
        // sub-pixel offsets within a pixel should always select that pixel.
        let mut arr = ndarray::Array2::zeros((8, 8));
        for r in 0..8 {
            for c in 0..8 {
                arr[(r, c)] = (r * 10 + c) as f64;
            }
        }
        let view = arr.view();

        // Offsets within pixel (3,4): corner-based coords in [4.0, 5.0) × [3.0, 4.0)
        for &dx in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.99] {
            for &dy in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.99] {
                let val = sample(&view, 4.0 + dx, 3.0 + dy, None).unwrap();
                assert_eq!(
                    val,
                    34.0,
                    "At ({}, {}), expected 34.0 but got {}",
                    4.0 + dx,
                    3.0 + dy,
                    val
                );
            }
        }
    }

    #[test]
    fn test_boundary_at_exact_pixel_edge() {
        // At exact pixel boundaries (integer coords), floor() determines which pixel
        let arr = array![[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]];
        let view = arr.view();

        // x=1.0 is the boundary between col 0 and col 1; floor(1.0) = 1
        assert_eq!(sample(&view, 1.0, 0.0, None), Some(20.0));
        // x=0.9999 still in col 0
        assert_eq!(sample(&view, 0.999, 0.0, None), Some(10.0));
    }
}
