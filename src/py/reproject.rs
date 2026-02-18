//! PyO3 binding for reproject_array.

use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::affine::Affine;
use crate::proj::crs::CrsTransform;
use crate::resample::ResamplingMethod;
use crate::warp::engine;

/// Reproject a 2D f64 array from one CRS to another.
///
/// Args:
///     src: Input 2D array (f64). For f32 data, cast to f64 on the Python side.
///     src_crs: Source CRS string (e.g. "EPSG:32633" or PROJ string).
///     src_transform: Source affine transform as 6-element tuple (a, b, c, d, e, f)
///         in rasterio convention: (pixel_width, rot_x, x_origin, rot_y, -pixel_height, y_origin).
///     dst_crs: Destination CRS string.
///     dst_transform: Destination affine transform as 6-element tuple (same convention).
///     dst_shape: Output shape as (rows, cols) tuple.
///     resampling: Resampling method name ("nearest" or "bilinear").
///     nodata: Optional nodata value.
///
/// Returns:
///     Reprojected 2D array (f64).
#[pyfunction]
#[pyo3(signature = (src, src_crs, src_transform, dst_crs, dst_transform, dst_shape, resampling="nearest", nodata=None))]
#[allow(clippy::too_many_arguments)]
pub fn reproject_array<'py>(
    py: Python<'py>,
    src: PyReadonlyArray2<'py, f64>,
    src_crs: &str,
    src_transform: [f64; 6],
    dst_crs: &str,
    dst_transform: [f64; 6],
    dst_shape: (usize, usize),
    resampling: &str,
    nodata: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let method = ResamplingMethod::from_name(resampling)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown resampling method: {resampling}")))?;

    // Copy strings to owned before releasing GIL
    let src_crs = src_crs.to_string();
    let dst_crs = dst_crs.to_string();

    // Copy array to owned ndarray
    let src_array: Array2<f64> = src.as_array().to_owned();

    let result: Array2<f64> = py.allow_threads(move || {
        let src_affine = Affine::new(
            src_transform[0],
            src_transform[1],
            src_transform[2],
            src_transform[3],
            src_transform[4],
            src_transform[5],
        );
        let dst_affine = Affine::new(
            dst_transform[0],
            dst_transform[1],
            dst_transform[2],
            dst_transform[3],
            dst_transform[4],
            dst_transform[5],
        );

        let crs_transform = CrsTransform::new(&src_crs, &dst_crs)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        engine::warp(
            &src_array.view(),
            &src_affine,
            &dst_affine,
            dst_shape,
            &crs_transform,
            method,
            nodata,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(PyArray2::from_owned_array(py, result))
}
