//! PyO3 binding for reproject_array with multi-dtype dispatch.

use ndarray::Array2;
use num_traits::NumCast;
use numpy::{PyArray2, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::affine::Affine;
use crate::proj::pipeline::Pipeline;
use crate::resample::ResamplingMethod;
use crate::warp::engine;

/// Reproject for a specific element type. Called after dtype dispatch.
#[allow(clippy::too_many_arguments)]
fn reproject_typed<T>(
    py: Python<'_>,
    src: &Bound<'_, PyUntypedArray>,
    nodata: Option<f64>,
    src_crs: String,
    src_transform: [f64; 6],
    dst_crs: String,
    dst_transform: [f64; 6],
    dst_shape: (usize, usize),
    method: ResamplingMethod,
) -> PyResult<PyObject>
where
    T: numpy::Element + Copy + NumCast + PartialEq + Default + Send,
{
    let typed = src.downcast::<PyArray2<T>>()?;
    let src_array: Array2<T> = typed.readonly().as_array().to_owned();

    // For float types, NaN is the default fill; for integer types, use 0 (T::default)
    let fill: T = match nodata {
        Some(nd) => NumCast::from(nd).unwrap_or(T::default()),
        None => NumCast::from(f64::NAN).unwrap_or(T::default()),
    };
    let nodata_t: Option<T> = nodata.and_then(|nd| NumCast::from(nd));

    let result: Array2<T> = py.allow_threads(move || {
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

        let pipeline =
            Pipeline::new(&src_crs, &dst_crs).map_err(|e| PyValueError::new_err(e.to_string()))?;

        engine::warp_generic(
            &src_array.view(),
            &src_affine,
            &dst_affine,
            dst_shape,
            &pipeline,
            method,
            nodata_t,
            fill,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(PyArray2::from_owned_array(py, result).into_any().unbind())
}

/// Reproject a 2D array from one CRS to another.
///
/// Supports multiple dtypes: float32, float64, uint8, uint16, int16.
/// For float types, NaN is used as the default fill value.
/// For integer types, 0 is used unless nodata is specified.
///
/// Args:
///     src: Input 2D numpy array (float32, float64, uint8, uint16, or int16).
///     src_crs: Source CRS string (e.g. "EPSG:32633" or PROJ string).
///     src_transform: Source affine transform as 6-element tuple (a, b, c, d, e, f)
///         in rasterio convention: (pixel_width, rot_x, x_origin, rot_y, -pixel_height, y_origin).
///     dst_crs: Destination CRS string.
///     dst_transform: Destination affine transform as 6-element tuple (same convention).
///     dst_shape: Output shape as (rows, cols) tuple.
///     resampling: Resampling method name ("nearest", "bilinear", "cubic", "lanczos", or "average").
///     nodata: Optional nodata value.
///
/// Returns:
///     Reprojected 2D array with the same dtype as the input.
#[pyfunction]
#[pyo3(signature = (src, src_crs, src_transform, dst_crs, dst_transform, dst_shape, resampling="nearest", nodata=None))]
#[allow(clippy::too_many_arguments)]
pub fn reproject_array(
    py: Python<'_>,
    src: &Bound<'_, PyUntypedArray>,
    src_crs: &str,
    src_transform: [f64; 6],
    dst_crs: &str,
    dst_transform: [f64; 6],
    dst_shape: (usize, usize),
    resampling: &str,
    nodata: Option<f64>,
) -> PyResult<PyObject> {
    let method = ResamplingMethod::from_name(resampling)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown resampling method: {resampling}")))?;

    let src_crs = src_crs.to_string();
    let dst_crs = dst_crs.to_string();

    let dt = src.dtype();

    if dt.is_equiv_to(&numpy::dtype::<f64>(py)) {
        return reproject_typed::<f64>(
            py,
            src,
            nodata,
            src_crs,
            src_transform,
            dst_crs,
            dst_transform,
            dst_shape,
            method,
        );
    }
    if dt.is_equiv_to(&numpy::dtype::<f32>(py)) {
        return reproject_typed::<f32>(
            py,
            src,
            nodata,
            src_crs,
            src_transform,
            dst_crs,
            dst_transform,
            dst_shape,
            method,
        );
    }
    if dt.is_equiv_to(&numpy::dtype::<u8>(py)) {
        return reproject_typed::<u8>(
            py,
            src,
            nodata,
            src_crs,
            src_transform,
            dst_crs,
            dst_transform,
            dst_shape,
            method,
        );
    }
    if dt.is_equiv_to(&numpy::dtype::<u16>(py)) {
        return reproject_typed::<u16>(
            py,
            src,
            nodata,
            src_crs,
            src_transform,
            dst_crs,
            dst_transform,
            dst_shape,
            method,
        );
    }
    if dt.is_equiv_to(&numpy::dtype::<i16>(py)) {
        return reproject_typed::<i16>(
            py,
            src,
            nodata,
            src_crs,
            src_transform,
            dst_crs,
            dst_transform,
            dst_shape,
            method,
        );
    }

    Err(PyValueError::new_err(format!(
        "Unsupported dtype: {}. Supported: float32, float64, uint8, uint16, int16",
        dt,
    )))
}
