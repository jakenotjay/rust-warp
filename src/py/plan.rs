//! PyO3 binding for plan_reproject — chunk planner for dask integration.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::affine::Affine;
use crate::chunk::planner;
use crate::resample::ResamplingMethod;

/// Plan the chunk-level reprojection tasks for a raster dataset.
///
/// Divides the destination grid into tiles and computes the corresponding source
/// ROI (with halo padding) for each tile.
///
/// Args:
///     src_crs: Source CRS string (e.g. "EPSG:32633").
///     src_transform: Source affine transform as 6-element tuple.
///     src_shape: Source raster shape as (rows, cols).
///     dst_crs: Destination CRS string.
///     dst_transform: Destination affine transform as 6-element tuple.
///     dst_shape: Destination raster shape as (rows, cols).
///     dst_chunks: Optional chunk size as (rows, cols). Defaults to full image.
///     resampling: Resampling method name. Defaults to "bilinear".
///
/// Returns:
///     List of tile plan dicts, each with keys:
///     - dst_slice: (row_start, row_end, col_start, col_end)
///     - src_slice: (row_start, row_end, col_start, col_end)
///     - src_transform: (a, b, c, d, e, f) shifted to src_slice origin
///     - dst_transform: (a, b, c, d, e, f) shifted to dst_slice origin
///     - dst_tile_shape: (rows, cols)
///     - has_data: bool
#[pyfunction]
#[pyo3(signature = (src_crs, src_transform, src_shape, dst_crs, dst_transform, dst_shape, dst_chunks=None, resampling="bilinear"))]
#[allow(clippy::too_many_arguments)]
pub fn plan_reproject(
    py: Python<'_>,
    src_crs: &str,
    src_transform: [f64; 6],
    src_shape: (usize, usize),
    dst_crs: &str,
    dst_transform: [f64; 6],
    dst_shape: (usize, usize),
    dst_chunks: Option<(usize, usize)>,
    resampling: &str,
) -> PyResult<Vec<PyObject>> {
    let method = ResamplingMethod::from_name(resampling).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown resampling method: '{resampling}'"
        ))
    })?;
    let kernel_radius = method.kernel_radius().ceil() as usize;

    let src_aff = Affine::new(
        src_transform[0],
        src_transform[1],
        src_transform[2],
        src_transform[3],
        src_transform[4],
        src_transform[5],
    );
    let dst_aff = Affine::new(
        dst_transform[0],
        dst_transform[1],
        dst_transform[2],
        dst_transform[3],
        dst_transform[4],
        dst_transform[5],
    );

    let tile_size = dst_chunks.unwrap_or(dst_shape);

    // Capture CRS strings for the closure
    let src_crs = src_crs.to_string();
    let dst_crs = dst_crs.to_string();

    let plans = py
        .allow_threads(move || {
            planner::plan_tiles(
                &src_crs,
                &src_aff,
                src_shape,
                &dst_crs,
                &dst_aff,
                dst_shape,
                tile_size,
                kernel_radius,
                21,
            )
        })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let result: Vec<PyObject> = plans
        .iter()
        .map(|plan| -> PyResult<PyObject> {
            let dict = PyDict::new(py);
            dict.set_item("dst_slice", plan.dst_slice)?;
            dict.set_item("src_slice", plan.src_slice)?;
            dict.set_item("src_transform", plan.src_transform.to_tuple())?;
            dict.set_item("dst_transform", plan.dst_transform.to_tuple())?;
            dict.set_item("dst_tile_shape", plan.dst_tile_shape)?;
            dict.set_item("has_data", plan.has_data)?;
            Ok(dict.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(result)
}
