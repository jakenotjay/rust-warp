//! Inverse-mapping warp engine.
//!
//! For each output pixel, projects back to source CRS and samples the source array.
//! Uses LinearApprox for scanline optimization when available.

use ndarray::{Array2, ArrayView2};

use crate::affine::Affine;
use crate::error::WarpError;
use crate::proj::approx::LinearApprox;
use crate::proj::pipeline::Pipeline;
use crate::resample::{self, ResamplingMethod};

/// Reproject a 2D array from source CRS to destination CRS.
///
/// # Arguments
/// * `src` — source raster data (row-major, f64)
/// * `src_affine` — source geotransform (pixel → source CRS coordinates)
/// * `dst_affine` — destination geotransform (pixel → destination CRS coordinates)
/// * `dst_shape` — (rows, cols) of the output array
/// * `pipeline` — CRS transform pipeline (dst → src direction)
/// * `method` — resampling method
/// * `nodata` — optional nodata sentinel value
#[allow(clippy::too_many_arguments)]
pub fn warp(
    src: &ArrayView2<'_, f64>,
    src_affine: &Affine,
    dst_affine: &Affine,
    dst_shape: (usize, usize),
    pipeline: &Pipeline,
    method: ResamplingMethod,
    nodata: Option<f64>,
) -> Result<Array2<f64>, WarpError> {
    let src_affine_inv = src_affine.inverse()?;
    let (dst_rows, dst_cols) = dst_shape;

    let fill = nodata.unwrap_or(f64::NAN);
    let mut dst = Array2::from_elem(dst_shape, fill);

    let approx = LinearApprox::default();
    let mut src_cols_buf = vec![0.0; dst_cols];
    let mut src_rows_buf = vec![0.0; dst_cols];

    for row in 0..dst_rows {
        // Use LinearApprox for the scanline
        let scanline_ok = approx
            .transform_scanline(
                pipeline,
                dst_affine,
                &src_affine_inv,
                row,
                dst_cols,
                &mut src_cols_buf,
                &mut src_rows_buf,
            )
            .is_ok();

        if !scanline_ok {
            // If scanline transform fails entirely, leave as nodata
            continue;
        }

        for col in 0..dst_cols {
            let src_col = src_cols_buf[col];
            let src_row = src_rows_buf[col];

            let val = match method {
                ResamplingMethod::Nearest => {
                    resample::nearest::sample(src, src_col, src_row, nodata)
                }
                ResamplingMethod::Bilinear => {
                    resample::bilinear::sample(src, src_col, src_row, nodata)
                }
                _ => {
                    return Err(WarpError::Resampling(format!(
                        "{method:?} not yet implemented"
                    )));
                }
            };

            if let Some(v) = val {
                dst[(row, col)] = v;
            }
        }
    }

    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_identity_reprojection() {
        // Same CRS, same affine → output should equal input
        let src = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let affine = Affine::new(10.0, 0.0, 500000.0, 0.0, -10.0, 6000040.0);
        let pipeline = Pipeline::new("EPSG:32633", "EPSG:32633").unwrap();

        let result = warp(
            &src.view(),
            &affine,
            &affine,
            (4, 4),
            &pipeline,
            ResamplingMethod::Nearest,
            None,
        )
        .unwrap();

        for row in 0..4 {
            for col in 0..4 {
                assert_relative_eq!(result[(row, col)], src[(row, col)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_identity_bilinear() {
        // Bilinear on identity should also match (for interior pixels)
        let src = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let affine = Affine::new(10.0, 0.0, 500000.0, 0.0, -10.0, 6000040.0);
        let pipeline = Pipeline::new("EPSG:32633", "EPSG:32633").unwrap();

        let result = warp(
            &src.view(),
            &affine,
            &affine,
            (4, 4),
            &pipeline,
            ResamplingMethod::Bilinear,
            None,
        )
        .unwrap();

        // Interior pixels should match exactly
        for row in 1..3 {
            for col in 1..3 {
                assert_relative_eq!(result[(row, col)], src[(row, col)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_nodata_propagation() {
        let mut src = Array2::from_elem((4, 4), 42.0);
        src[(1, 1)] = f64::NAN;

        let affine = Affine::new(10.0, 0.0, 500000.0, 0.0, -10.0, 6000040.0);
        let pipeline = Pipeline::new("EPSG:32633", "EPSG:32633").unwrap();

        let result = warp(
            &src.view(),
            &affine,
            &affine,
            (4, 4),
            &pipeline,
            ResamplingMethod::Nearest,
            None,
        )
        .unwrap();

        // The NaN pixel should remain NaN
        assert!(result[(1, 1)].is_nan());
        // Other pixels should be 42.0
        assert_relative_eq!(result[(0, 0)], 42.0);
        assert_relative_eq!(result[(2, 2)], 42.0);
    }

    #[test]
    fn test_unsupported_method() {
        let src = array![[1.0]];
        let affine = Affine::new(10.0, 0.0, 500000.0, 0.0, -10.0, 6000010.0);
        let pipeline = Pipeline::new("EPSG:32633", "EPSG:32633").unwrap();

        let result = warp(
            &src.view(),
            &affine,
            &affine,
            (1, 1),
            &pipeline,
            ResamplingMethod::Cubic,
            None,
        );
        assert!(result.is_err());
    }
}
