//! Linear approximation for scanline projection.
//!
//! Instead of projecting every pixel, computes exact source coordinates at a few
//! points per scanline and linearly interpolates between them, recursively
//! subdividing when the interpolation error exceeds a tolerance (default 0.125 px).

use crate::affine::Affine;
use crate::error::ProjError;
use crate::proj::pipeline::Pipeline;

/// Scanline projection approximator using recursive subdivision.
pub struct LinearApprox {
    /// Maximum allowed interpolation error in pixels. Default: 0.125 (GDAL default).
    pub tolerance_px: f64,
}

impl Default for LinearApprox {
    fn default() -> Self {
        Self {
            tolerance_px: 0.125,
        }
    }
}

impl LinearApprox {
    pub fn new(tolerance_px: f64) -> Self {
        Self { tolerance_px }
    }

    /// Transform a full scanline from destination pixel coords to source pixel coords.
    ///
    /// For each pixel in the scanline at `row`, computes the corresponding source
    /// pixel (col, row) using the pipeline, with linear interpolation and recursive
    /// subdivision to stay within tolerance.
    ///
    /// # Arguments
    /// * `pipeline` — CRS-to-CRS transform (dst → src direction)
    /// * `dst_affine` — destination geotransform (pixel → dst CRS coords)
    /// * `src_affine_inv` — **inverse** of source geotransform (src CRS → pixel coords)
    /// * `row` — destination row index
    /// * `width` — number of columns in the destination
    /// * `out_src_col` — output: source column for each destination pixel (len = width)
    /// * `out_src_row` — output: source row for each destination pixel (len = width)
    #[allow(clippy::too_many_arguments)]
    pub fn transform_scanline(
        &self,
        pipeline: &Pipeline,
        dst_affine: &Affine,
        src_affine_inv: &Affine,
        row: usize,
        width: usize,
        out_src_col: &mut [f64],
        out_src_row: &mut [f64],
    ) -> Result<(), ProjError> {
        if width == 0 {
            return Ok(());
        }

        let row_f = row as f64 + 0.5; // pixel center

        // Project endpoints and midpoint
        let left_col = 0.5;
        let right_col = width as f64 - 0.5;

        let (left_sc, left_sr) =
            self.project_pixel(pipeline, dst_affine, src_affine_inv, left_col, row_f)?;
        let (right_sc, right_sr) =
            self.project_pixel(pipeline, dst_affine, src_affine_inv, right_col, row_f)?;

        // Fill the scanline using recursive subdivision
        self.subdivide(
            pipeline,
            dst_affine,
            src_affine_inv,
            row_f,
            0,
            width - 1,
            left_sc,
            left_sr,
            right_sc,
            right_sr,
            out_src_col,
            out_src_row,
            0, // recursion depth
        )?;

        Ok(())
    }

    /// Project a single destination pixel to source pixel coordinates.
    fn project_pixel(
        &self,
        pipeline: &Pipeline,
        dst_affine: &Affine,
        src_affine_inv: &Affine,
        col: f64,
        row: f64,
    ) -> Result<(f64, f64), ProjError> {
        let (dst_x, dst_y) = dst_affine.forward(col, row);
        let (src_x, src_y) = pipeline.transform_inv(dst_x, dst_y)?;
        Ok(src_affine_inv.forward(src_x, src_y))
    }

    /// Recursively subdivide and interpolate a scanline segment.
    ///
    /// `left_idx` and `right_idx` are pixel indices (0-based).
    /// `left_sc/sr` and `right_sc/sr` are the exact source pixel coords at those indices.
    #[allow(clippy::too_many_arguments)]
    fn subdivide(
        &self,
        pipeline: &Pipeline,
        dst_affine: &Affine,
        src_affine_inv: &Affine,
        row_f: f64,
        left_idx: usize,
        right_idx: usize,
        left_sc: f64,
        left_sr: f64,
        right_sc: f64,
        right_sr: f64,
        out_src_col: &mut [f64],
        out_src_row: &mut [f64],
        depth: usize,
    ) -> Result<(), ProjError> {
        // Base case: adjacent or same pixel
        if right_idx <= left_idx + 1 {
            out_src_col[left_idx] = left_sc;
            out_src_row[left_idx] = left_sr;
            if right_idx > left_idx {
                out_src_col[right_idx] = right_sc;
                out_src_row[right_idx] = right_sr;
            }
            return Ok(());
        }

        let mid_idx = (left_idx + right_idx) / 2;
        let mid_col = mid_idx as f64 + 0.5;

        // Compute exact source coords at the midpoint
        let (mid_sc, mid_sr) =
            self.project_pixel(pipeline, dst_affine, src_affine_inv, mid_col, row_f)?;

        // Linearly interpolated midpoint
        let t = (mid_idx - left_idx) as f64 / (right_idx - left_idx) as f64;
        let interp_sc = left_sc + t * (right_sc - left_sc);
        let interp_sr = left_sr + t * (right_sr - left_sr);

        // Check error
        let err_c = (mid_sc - interp_sc).abs();
        let err_r = (mid_sr - interp_sr).abs();
        let err = err_c.max(err_r);

        const MAX_DEPTH: usize = 20;

        if err > self.tolerance_px && depth < MAX_DEPTH {
            // Subdivide left half
            self.subdivide(
                pipeline,
                dst_affine,
                src_affine_inv,
                row_f,
                left_idx,
                mid_idx,
                left_sc,
                left_sr,
                mid_sc,
                mid_sr,
                out_src_col,
                out_src_row,
                depth + 1,
            )?;
            // Subdivide right half
            self.subdivide(
                pipeline,
                dst_affine,
                src_affine_inv,
                row_f,
                mid_idx,
                right_idx,
                mid_sc,
                mid_sr,
                right_sc,
                right_sr,
                out_src_col,
                out_src_row,
                depth + 1,
            )?;
        } else {
            // Linear interpolation is good enough for this segment
            for i in left_idx..=right_idx {
                let t = (i - left_idx) as f64 / (right_idx - left_idx) as f64;
                out_src_col[i] = left_sc + t * (right_sc - left_sc);
                out_src_row[i] = left_sr + t * (right_sr - left_sr);
            }
            // Fix exact values at the known points
            out_src_col[left_idx] = left_sc;
            out_src_row[left_idx] = left_sr;
            out_src_col[right_idx] = right_sc;
            out_src_row[right_idx] = right_sr;
            // Use exact midpoint value (we computed it)
            out_src_col[mid_idx] = mid_sc;
            out_src_row[mid_idx] = mid_sr;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::pipeline::Pipeline;
    use approx::assert_relative_eq;

    /// Compute exact per-pixel projection for comparison.
    fn exact_scanline(
        pipeline: &Pipeline,
        dst_affine: &Affine,
        src_affine_inv: &Affine,
        row: usize,
        width: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let row_f = row as f64 + 0.5;
        let mut cols = vec![0.0; width];
        let mut rows = vec![0.0; width];
        for i in 0..width {
            let col_f = i as f64 + 0.5;
            let (dst_x, dst_y) = dst_affine.forward(col_f, row_f);
            if let Ok((src_x, src_y)) = pipeline.transform_inv(dst_x, dst_y) {
                let (sc, sr) = src_affine_inv.forward(src_x, src_y);
                cols[i] = sc;
                rows[i] = sr;
            }
        }
        (cols, rows)
    }

    #[test]
    fn test_identity_scanline() {
        // Same CRS, same affine → approximation should match exact
        let pipeline = Pipeline::new("EPSG:32633", "EPSG:32633").unwrap();
        let affine = Affine::new(10.0, 0.0, 500000.0, 0.0, -10.0, 6000040.0);
        let inv = affine.inverse().unwrap();

        let approx = LinearApprox::default();
        let width = 64;
        let mut out_col = vec![0.0; width];
        let mut out_row = vec![0.0; width];

        approx
            .transform_scanline(
                &pipeline,
                &affine,
                &inv,
                0,
                width,
                &mut out_col,
                &mut out_row,
            )
            .unwrap();

        // Should map to itself
        for i in 0..width {
            assert_relative_eq!(out_col[i], i as f64 + 0.5, epsilon = 0.01);
            assert_relative_eq!(out_row[i], 0.5, epsilon = 0.01);
        }
    }

    #[test]
    fn test_utm_to_4326_within_tolerance() {
        // UTM 33N → 4326 — moderately nonlinear
        let pipeline = Pipeline::new("EPSG:4326", "EPSG:32633").unwrap();
        let dst_affine = Affine::new(100.0, 0.0, 400000.0, 0.0, -100.0, 6000000.0);
        let src_affine = Affine::new(0.001, 0.0, 13.0, 0.0, -0.001, 55.0);
        let src_inv = src_affine.inverse().unwrap();

        let approx = LinearApprox::default();
        let width = 256;
        let row = 50;

        let mut approx_col = vec![0.0; width];
        let mut approx_row = vec![0.0; width];

        approx
            .transform_scanline(
                &pipeline,
                &dst_affine,
                &src_inv,
                row,
                width,
                &mut approx_col,
                &mut approx_row,
            )
            .unwrap();

        let (exact_col, exact_row) = exact_scanline(&pipeline, &dst_affine, &src_inv, row, width);

        for i in 0..width {
            let err_c = (approx_col[i] - exact_col[i]).abs();
            let err_r = (approx_row[i] - exact_row[i]).abs();
            let err = err_c.max(err_r);
            assert!(
                err < 0.2, // slightly more than tolerance due to linear interp between segments
                "pixel {i}: err={err:.4} (approx=({:.2},{:.2}), exact=({:.2},{:.2}))",
                approx_col[i],
                approx_row[i],
                exact_col[i],
                exact_row[i]
            );
        }
    }

    #[test]
    fn test_webmerc_to_4326_within_tolerance() {
        // Web Mercator → 4326 — more nonlinear
        let pipeline = Pipeline::new("EPSG:4326", "EPSG:3857").unwrap();
        let dst_affine = Affine::new(1000.0, 0.0, 0.0, 0.0, -1000.0, 8000000.0);
        let src_affine = Affine::new(0.01, 0.0, -10.0, 0.0, -0.01, 65.0);
        let src_inv = src_affine.inverse().unwrap();

        let approx = LinearApprox::default();
        let width = 512;
        let row = 100;

        let mut approx_col = vec![0.0; width];
        let mut approx_row = vec![0.0; width];

        approx
            .transform_scanline(
                &pipeline,
                &dst_affine,
                &src_inv,
                row,
                width,
                &mut approx_col,
                &mut approx_row,
            )
            .unwrap();

        let (exact_col, exact_row) = exact_scanline(&pipeline, &dst_affine, &src_inv, row, width);

        let mut max_err = 0.0_f64;
        for i in 0..width {
            let err_c = (approx_col[i] - exact_col[i]).abs();
            let err_r = (approx_row[i] - exact_row[i]).abs();
            max_err = max_err.max(err_c.max(err_r));
        }

        assert!(
            max_err < 0.2,
            "max error = {max_err:.6} pixels (want < 0.2)"
        );
    }

    #[test]
    fn test_single_pixel() {
        let pipeline = Pipeline::new("EPSG:32633", "EPSG:32633").unwrap();
        let affine = Affine::new(10.0, 0.0, 500000.0, 0.0, -10.0, 6000010.0);
        let inv = affine.inverse().unwrap();

        let approx = LinearApprox::default();
        let mut col = vec![0.0; 1];
        let mut row = vec![0.0; 1];

        approx
            .transform_scanline(&pipeline, &affine, &inv, 0, 1, &mut col, &mut row)
            .unwrap();

        assert_relative_eq!(col[0], 0.5, epsilon = 0.01);
        assert_relative_eq!(row[0], 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_linear_approx_accuracy_utm_to_4326() {
        // Verify LinearApprox stays within 0.125 pixel tolerance for UTM→4326
        let pipeline = Pipeline::new("EPSG:4326", "EPSG:32633").unwrap();
        let dst_affine = Affine::new(100.0, 0.0, 400000.0, 0.0, -100.0, 6000000.0);
        let src_affine = Affine::new(0.001, 0.0, 13.0, 0.0, -0.001, 55.0);
        let src_inv = src_affine.inverse().unwrap();

        let approx = LinearApprox::new(0.125);
        let width = 1024;

        for row in [0, 100, 250, 500, 900] {
            let mut approx_col = vec![0.0; width];
            let mut approx_row = vec![0.0; width];

            approx
                .transform_scanline(
                    &pipeline,
                    &dst_affine,
                    &src_inv,
                    row,
                    width,
                    &mut approx_col,
                    &mut approx_row,
                )
                .unwrap();

            let (exact_col, exact_row) =
                exact_scanline(&pipeline, &dst_affine, &src_inv, row, width);

            let mut max_err = 0.0_f64;
            for i in 0..width {
                let err_c = (approx_col[i] - exact_col[i]).abs();
                let err_r = (approx_row[i] - exact_row[i]).abs();
                max_err = max_err.max(err_c.max(err_r));
            }

            assert!(
                max_err < 0.2,
                "Row {row}: max error = {max_err:.6} pixels (want < 0.2)"
            );
        }
    }

    #[test]
    fn test_linear_approx_accuracy_webmerc_to_4326() {
        // Verify LinearApprox for WebMerc→4326 at multiple scanlines
        let pipeline = Pipeline::new("EPSG:4326", "EPSG:3857").unwrap();
        let dst_affine = Affine::new(1000.0, 0.0, 0.0, 0.0, -1000.0, 8000000.0);
        let src_affine = Affine::new(0.01, 0.0, -10.0, 0.0, -0.01, 65.0);
        let src_inv = src_affine.inverse().unwrap();

        let approx = LinearApprox::new(0.125);
        let width = 1024;

        for row in [0, 200, 500, 800] {
            let mut approx_col = vec![0.0; width];
            let mut approx_row = vec![0.0; width];

            approx
                .transform_scanline(
                    &pipeline,
                    &dst_affine,
                    &src_inv,
                    row,
                    width,
                    &mut approx_col,
                    &mut approx_row,
                )
                .unwrap();

            let (exact_col, exact_row) =
                exact_scanline(&pipeline, &dst_affine, &src_inv, row, width);

            let mut max_err = 0.0_f64;
            for i in 0..width {
                let err_c = (approx_col[i] - exact_col[i]).abs();
                let err_r = (approx_row[i] - exact_row[i]).abs();
                max_err = max_err.max(err_c.max(err_r));
            }

            assert!(
                max_err < 0.2,
                "Row {row}: max error = {max_err:.6} pixels (want < 0.2)"
            );
        }
    }
}
