//! PyO3 binding for batch CRS coordinate transformation.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::proj::pipeline::Pipeline;

/// Transform arrays of coordinates from one CRS to another.
///
/// Args:
///     x: 1D array of x coordinates (longitude or easting).
///     y: 1D array of y coordinates (latitude or northing).
///     src_crs: Source CRS string (e.g. "EPSG:4326").
///     dst_crs: Destination CRS string (e.g. "EPSG:32633").
///
/// Returns:
///     Tuple of (x_out, y_out) arrays in the destination CRS.
#[pyfunction]
#[pyo3(signature = (x, y, src_crs, dst_crs))]
#[allow(clippy::type_complexity)]
pub fn transform_points<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    src_crs: &str,
    dst_crs: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let x_view = x.as_array();
    let y_view = y.as_array();

    let n = x_view.len();
    let y_len = y_view.len();
    if n != y_len {
        return Err(PyValueError::new_err(format!(
            "x and y must have same length, got {} and {}",
            n, y_len
        )));
    }

    let mut coords: Vec<(f64, f64)> = x_view
        .iter()
        .zip(y_view.iter())
        .map(|(&xi, &yi)| (xi, yi))
        .collect();

    // Swap CRS args: Pipeline does dst->src, so Pipeline(dst, src).transform_inv = src->dst
    let src_crs = src_crs.to_string();
    let dst_crs = dst_crs.to_string();

    let coords = py.allow_threads(move || -> PyResult<Vec<(f64, f64)>> {
        let pipeline =
            Pipeline::new(&dst_crs, &src_crs).map_err(|e| PyValueError::new_err(e.to_string()))?;
        pipeline
            .transform_inv_batch(&mut coords)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(coords)
    })?;

    let (xs, ys): (Vec<f64>, Vec<f64>) = coords.into_iter().unzip();

    Ok((
        PyArray1::from_owned_array(py, ndarray::Array1::from(xs)),
        PyArray1::from_owned_array(py, ndarray::Array1::from(ys)),
    ))
}
