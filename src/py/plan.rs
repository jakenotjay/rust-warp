//! PyO3 binding for plan_reproject (stub for Phase 4).

use pyo3::prelude::*;

/// Plan the chunk-level reprojection tasks for a raster dataset.
///
/// This is a stub that returns an empty list. Phase 4 will implement the
/// real chunk planner that maps destination chunks to source ROIs.
///
/// Args:
///     src_crs: Source CRS string.
///     src_transform: Source affine transform as 6-element tuple.
///     src_shape: Source raster shape as (rows, cols).
///     dst_crs: Destination CRS string.
///     dst_transform: Destination affine transform as 6-element tuple.
///     dst_shape: Destination raster shape as (rows, cols).
///     dst_chunks: Optional chunk size as (rows, cols). Defaults to full image.
///     resampling: Resampling method name. Defaults to "bilinear".
///
/// Returns:
///     List of chunk task dicts (currently empty — Phase 4 stub).
#[pyfunction]
#[pyo3(signature = (src_crs, src_transform, src_shape, dst_crs, dst_transform, dst_shape, dst_chunks=None, resampling="bilinear"))]
#[allow(clippy::too_many_arguments, unused_variables)]
pub fn plan_reproject(
    src_crs: &str,
    src_transform: [f64; 6],
    src_shape: (usize, usize),
    dst_crs: &str,
    dst_transform: [f64; 6],
    dst_shape: (usize, usize),
    dst_chunks: Option<(usize, usize)>,
    resampling: &str,
) -> PyResult<Vec<PyObject>> {
    Ok(vec![])
}
