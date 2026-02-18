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
}
