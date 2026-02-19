//! Inverse-mapping warp engine.
//!
//! For each output pixel, projects back to source CRS and samples the source array.
//! Uses LinearApprox for scanline optimization when available.
//! Output rows are processed in parallel using Rayon.

use ndarray::{Array2, ArrayView2};
use num_traits::NumCast;
use rayon::prelude::*;

use crate::affine::Affine;
use crate::error::WarpError;
use crate::proj::approx::LinearApprox;
use crate::proj::pipeline::Pipeline;
use crate::resample::{self, ResamplingMethod};

/// Compute the source-to-destination pixel scale ratio from affine transforms.
///
/// Returns `(sx, sy)` where `sx` is the source pixel width in destination pixel
/// units. Used by the average kernel to determine the averaging footprint.
fn compute_scale(src_affine: &Affine, dst_affine: &Affine) -> (f64, f64) {
    let src_pixel_x = (src_affine.a * src_affine.a + src_affine.d * src_affine.d).sqrt();
    let src_pixel_y = (src_affine.b * src_affine.b + src_affine.e * src_affine.e).sqrt();
    let dst_pixel_x = (dst_affine.a * dst_affine.a + dst_affine.d * dst_affine.d).sqrt();
    let dst_pixel_y = (dst_affine.b * dst_affine.b + dst_affine.e * dst_affine.e).sqrt();

    let sx = dst_pixel_x / src_pixel_x.max(1e-15);
    let sy = dst_pixel_y / src_pixel_y.max(1e-15);
    (sx, sy)
}

/// Reproject a 2D array from source CRS to destination CRS (generic over element type).
///
/// # Arguments
/// * `src` — source raster data (row-major)
/// * `src_affine` — source geotransform (pixel → source CRS coordinates)
/// * `dst_affine` — destination geotransform (pixel → destination CRS coordinates)
/// * `dst_shape` — (rows, cols) of the output array
/// * `pipeline` — CRS transform pipeline (dst → src direction)
/// * `method` — resampling method
/// * `nodata` — optional nodata sentinel value
/// * `fill` — value to fill the output array (pixels that don't map to source)
#[allow(clippy::too_many_arguments)]
pub fn warp_generic<T>(
    src: &ArrayView2<'_, T>,
    src_affine: &Affine,
    dst_affine: &Affine,
    dst_shape: (usize, usize),
    pipeline: &Pipeline,
    method: ResamplingMethod,
    nodata: Option<T>,
    fill: T,
) -> Result<Array2<T>, WarpError>
where
    T: Copy + NumCast + PartialEq + Default + Send + Sync,
{
    let src_affine_inv = src_affine.inverse()?;
    let (_dst_rows, dst_cols) = dst_shape;

    let mut dst = Array2::from_elem(dst_shape, fill);

    let approx = LinearApprox::default();
    let scale = compute_scale(src_affine, dst_affine);
    let radius = method.kernel_radius();
    let src_rows_f = src.nrows() as f64;
    let src_cols_f = src.ncols() as f64;

    dst.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(row, mut dst_row)| {
            // Thread-local coordinate buffers (allocated once per row)
            let mut src_cols_buf = vec![0.0_f64; dst_cols];
            let mut src_rows_buf = vec![0.0_f64; dst_cols];

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
                return;
            }

            for col in 0..dst_cols {
                let src_col = src_cols_buf[col];
                let src_row = src_rows_buf[col];

                // Kernel-radius pre-check: skip if clearly outside source extent
                if src_col < -radius
                    || src_col > src_cols_f + radius
                    || src_row < -radius
                    || src_row > src_rows_f + radius
                {
                    continue;
                }

                let val = match method {
                    ResamplingMethod::Nearest => {
                        resample::nearest::sample(src, src_col, src_row, nodata)
                    }
                    ResamplingMethod::Bilinear => {
                        resample::bilinear::sample(src, src_col, src_row, nodata)
                    }
                    ResamplingMethod::Cubic => {
                        resample::cubic::sample(src, src_col, src_row, nodata)
                    }
                    ResamplingMethod::Lanczos => {
                        resample::lanczos::sample(src, src_col, src_row, nodata)
                    }
                    ResamplingMethod::Average => {
                        resample::average::sample(src, src_col, src_row, nodata, scale)
                    }
                };

                if let Some(v) = val {
                    dst_row[col] = v;
                }
            }
        });

    Ok(dst)
}

/// Reproject a 2D f64 array from source CRS to destination CRS.
///
/// Thin wrapper around `warp_generic::<f64>()` that uses NaN as the default
/// nodata/fill value.
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
    let fill = nodata.unwrap_or(f64::NAN);
    warp_generic(
        src, src_affine, dst_affine, dst_shape, pipeline, method, nodata, fill,
    )
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
    fn test_identity_cubic() {
        // Cubic needs larger array (4×4 neighborhood → radius 2)
        let mut src = Array2::zeros((8, 8));
        for r in 0..8 {
            for c in 0..8 {
                src[(r, c)] = (r * 8 + c) as f64;
            }
        }
        let affine = Affine::new(10.0, 0.0, 500000.0, 0.0, -10.0, 6000080.0);
        let pipeline = Pipeline::new("EPSG:32633", "EPSG:32633").unwrap();

        let result = warp(
            &src.view(),
            &affine,
            &affine,
            (8, 8),
            &pipeline,
            ResamplingMethod::Cubic,
            None,
        )
        .unwrap();

        // Interior pixels (2..6) should match
        for row in 2..6 {
            for col in 2..6 {
                assert_relative_eq!(result[(row, col)], src[(row, col)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_identity_lanczos() {
        // Lanczos needs larger array (6×6 neighborhood → radius 3)
        let mut src = Array2::zeros((12, 12));
        for r in 0..12 {
            for c in 0..12 {
                src[(r, c)] = (r * 12 + c) as f64;
            }
        }
        let affine = Affine::new(10.0, 0.0, 500000.0, 0.0, -10.0, 6000120.0);
        let pipeline = Pipeline::new("EPSG:32633", "EPSG:32633").unwrap();

        let result = warp(
            &src.view(),
            &affine,
            &affine,
            (12, 12),
            &pipeline,
            ResamplingMethod::Lanczos,
            None,
        )
        .unwrap();

        // Interior pixels (3..9) should match
        for row in 3..9 {
            for col in 3..9 {
                assert_relative_eq!(result[(row, col)], src[(row, col)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_identity_average() {
        // Average with same affine → scale=1, should reproduce input
        let mut src = Array2::zeros((8, 8));
        for r in 0..8 {
            for c in 0..8 {
                src[(r, c)] = (r * 8 + c) as f64;
            }
        }
        let affine = Affine::new(10.0, 0.0, 500000.0, 0.0, -10.0, 6000080.0);
        let pipeline = Pipeline::new("EPSG:32633", "EPSG:32633").unwrap();

        let result = warp(
            &src.view(),
            &affine,
            &affine,
            (8, 8),
            &pipeline,
            ResamplingMethod::Average,
            None,
        )
        .unwrap();

        // All pixels should match (average with scale=1 = single pixel)
        for row in 0..8 {
            for col in 0..8 {
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

        assert!(result[(1, 1)].is_nan());
        assert_relative_eq!(result[(0, 0)], 42.0);
        assert_relative_eq!(result[(2, 2)], 42.0);
    }

    #[test]
    fn test_warp_generic_i32() {
        // Test that warp_generic works with integer types
        let mut src = Array2::zeros((4, 4));
        for r in 0..4_i32 {
            for c in 0..4_i32 {
                src[(r as usize, c as usize)] = r * 4 + c;
            }
        }
        let affine = Affine::new(10.0, 0.0, 500000.0, 0.0, -10.0, 6000040.0);
        let pipeline = Pipeline::new("EPSG:32633", "EPSG:32633").unwrap();

        let result = warp_generic(
            &src.view(),
            &affine,
            &affine,
            (4, 4),
            &pipeline,
            ResamplingMethod::Nearest,
            None,
            0_i32,
        )
        .unwrap();

        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(result[(row, col)], src[(row, col)]);
            }
        }
    }

    #[test]
    fn test_compute_scale_identity() {
        let affine = Affine::new(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0);
        let (sx, sy) = compute_scale(&affine, &affine);
        assert_relative_eq!(sx, 1.0, epsilon = 1e-10);
        assert_relative_eq!(sy, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_scale_downscale() {
        let src_affine = Affine::new(10.0, 0.0, 0.0, 0.0, -10.0, 0.0);
        let dst_affine = Affine::new(20.0, 0.0, 0.0, 0.0, -20.0, 0.0);
        let (sx, sy) = compute_scale(&src_affine, &dst_affine);
        assert_relative_eq!(sx, 2.0, epsilon = 1e-10);
        assert_relative_eq!(sy, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pipeline_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Pipeline>();
    }

    #[test]
    fn test_parallel_matches_sequential() {
        // Compare Rayon output to a manual sequential reference on 128×128
        let mut src = Array2::zeros((128, 128));
        for r in 0..128 {
            for c in 0..128 {
                src[(r, c)] = (r * 128 + c) as f64;
            }
        }
        let affine = Affine::new(100.0, 0.0, 400000.0, 0.0, -100.0, 6012800.0);
        let pipeline = Pipeline::new("EPSG:32633", "EPSG:32633").unwrap();

        // Test with nearest (exact) and bilinear (interpolated)
        for method in [ResamplingMethod::Nearest, ResamplingMethod::Bilinear] {
            let result = warp(
                &src.view(),
                &affine,
                &affine,
                (128, 128),
                &pipeline,
                method,
                None,
            )
            .unwrap();

            // For identity reprojection, interior pixels should match
            let margin = method.kernel_radius().ceil() as usize;
            for row in margin..128 - margin {
                for col in margin..128 - margin {
                    assert_relative_eq!(result[(row, col)], src[(row, col)], epsilon = 1e-4,);
                }
            }
        }
    }
}
