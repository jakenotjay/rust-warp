//! PyO3 binding for reproject_array with multi-dtype dispatch.

use ndarray::{Array2, ArrayView2};
use num_traits::NumCast;
use numpy::{
    PyArray2, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray2, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::affine::Affine;
use crate::proj::approx::LinearApprox;
use crate::proj::pipeline::Pipeline;
use crate::resample::{self, ResamplingMethod};
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
    T: numpy::Element + Copy + NumCast + PartialEq + Default + Send + Sync,
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
    if dt.is_equiv_to(&numpy::dtype::<i8>(py)) {
        return reproject_typed::<i8>(
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
        "Unsupported dtype: {}. Supported: float32, float64, uint8, uint16, int16, int8",
        dt,
    )))
}

/// Compute the source pixel coordinate grid for a reprojection.
///
/// For each destination pixel, computes the corresponding source pixel
/// coordinates using the same transform chain as reproject_array, but
/// without doing any resampling. This allows comparing rust-warp's
/// coordinate mapping against pyproj's.
///
/// Args:
///     src_crs: Source CRS string.
///     src_transform: Source affine transform as 6-element tuple.
///     dst_crs: Destination CRS string.
///     dst_transform: Destination affine transform as 6-element tuple.
///     dst_shape: Output shape as (rows, cols) tuple.
///
/// Returns:
///     Tuple of (src_col_grid, src_row_grid) — two 2D float64 arrays of
///     shape dst_shape containing the source pixel coordinates.
#[pyfunction]
#[pyo3(signature = (src_crs, src_transform, dst_crs, dst_transform, dst_shape))]
#[allow(clippy::type_complexity)]
pub fn transform_grid<'py>(
    py: Python<'py>,
    src_crs: &str,
    src_transform: [f64; 6],
    dst_crs: &str,
    dst_transform: [f64; 6],
    dst_shape: (usize, usize),
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let src_crs = src_crs.to_string();
    let dst_crs = dst_crs.to_string();

    let (col_grid, row_grid) =
        py.allow_threads(move || -> PyResult<(Array2<f64>, Array2<f64>)> {
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
            let src_affine_inv = src_affine
                .inverse()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            let pipeline = Pipeline::new(&src_crs, &dst_crs)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            let (dst_rows, dst_cols) = dst_shape;
            let mut col_grid = Array2::from_elem(dst_shape, f64::NAN);
            let mut row_grid = Array2::from_elem(dst_shape, f64::NAN);

            let approx = LinearApprox::default();

            for row in 0..dst_rows {
                let mut src_cols_buf = vec![0.0_f64; dst_cols];
                let mut src_rows_buf = vec![0.0_f64; dst_cols];

                if approx
                    .transform_scanline(
                        &pipeline,
                        &dst_affine,
                        &src_affine_inv,
                        row,
                        dst_cols,
                        &mut src_cols_buf,
                        &mut src_rows_buf,
                    )
                    .is_ok()
                {
                    for col in 0..dst_cols {
                        col_grid[(row, col)] = src_cols_buf[col];
                        row_grid[(row, col)] = src_rows_buf[col];
                    }
                }
            }

            Ok((col_grid, row_grid))
        })?;

    Ok((
        PyArray2::from_owned_array(py, col_grid),
        PyArray2::from_owned_array(py, row_grid),
    ))
}

/// Reproject using pre-computed source pixel coordinate grids.
///
/// Takes pre-computed source pixel coordinates (e.g. from pyproj) and
/// samples the source array at those locations. This bypasses projection
/// entirely and tests only the resampling kernel.
///
/// Args:
///     src: Input 2D numpy array (float32, float64, uint8, uint16, or int16).
///     src_col_grid: 2D float64 array of source column coordinates.
///     src_row_grid: 2D float64 array of source row coordinates.
///     resampling: Resampling method name.
///     nodata: Optional nodata value.
///
/// Returns:
///     Resampled 2D array with same dtype as input.
#[pyfunction]
#[pyo3(signature = (src, src_col_grid, src_row_grid, resampling="nearest", nodata=None))]
#[allow(clippy::too_many_arguments)]
pub fn reproject_with_grid(
    py: Python<'_>,
    src: &Bound<'_, PyUntypedArray>,
    src_col_grid: PyReadonlyArray2<'_, f64>,
    src_row_grid: PyReadonlyArray2<'_, f64>,
    resampling: &str,
    nodata: Option<f64>,
) -> PyResult<PyObject> {
    let method = ResamplingMethod::from_name(resampling)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown resampling method: {resampling}")))?;

    let dt = src.dtype();

    if dt.is_equiv_to(&numpy::dtype::<f64>(py)) {
        return reproject_with_grid_typed::<f64>(
            py,
            src,
            &src_col_grid,
            &src_row_grid,
            method,
            nodata,
        );
    }
    if dt.is_equiv_to(&numpy::dtype::<f32>(py)) {
        return reproject_with_grid_typed::<f32>(
            py,
            src,
            &src_col_grid,
            &src_row_grid,
            method,
            nodata,
        );
    }
    if dt.is_equiv_to(&numpy::dtype::<u8>(py)) {
        return reproject_with_grid_typed::<u8>(
            py,
            src,
            &src_col_grid,
            &src_row_grid,
            method,
            nodata,
        );
    }
    if dt.is_equiv_to(&numpy::dtype::<u16>(py)) {
        return reproject_with_grid_typed::<u16>(
            py,
            src,
            &src_col_grid,
            &src_row_grid,
            method,
            nodata,
        );
    }
    if dt.is_equiv_to(&numpy::dtype::<i16>(py)) {
        return reproject_with_grid_typed::<i16>(
            py,
            src,
            &src_col_grid,
            &src_row_grid,
            method,
            nodata,
        );
    }
    if dt.is_equiv_to(&numpy::dtype::<i8>(py)) {
        return reproject_with_grid_typed::<i8>(
            py,
            src,
            &src_col_grid,
            &src_row_grid,
            method,
            nodata,
        );
    }

    Err(PyValueError::new_err(format!(
        "Unsupported dtype: {}. Supported: float32, float64, uint8, uint16, int16, int8",
        dt,
    )))
}

/// Typed implementation of reproject_with_grid.
fn reproject_with_grid_typed<T>(
    py: Python<'_>,
    src: &Bound<'_, PyUntypedArray>,
    src_col_grid: &PyReadonlyArray2<'_, f64>,
    src_row_grid: &PyReadonlyArray2<'_, f64>,
    method: ResamplingMethod,
    nodata: Option<f64>,
) -> PyResult<PyObject>
where
    T: numpy::Element + Copy + NumCast + PartialEq + Default + Send + Sync,
{
    let typed = src.downcast::<PyArray2<T>>()?;
    let src_array: Array2<T> = typed.readonly().as_array().to_owned();
    let col_grid: Array2<f64> = src_col_grid.as_array().to_owned();
    let row_grid: Array2<f64> = src_row_grid.as_array().to_owned();

    let fill: T = match nodata {
        Some(nd) => NumCast::from(nd).unwrap_or(T::default()),
        None => NumCast::from(f64::NAN).unwrap_or(T::default()),
    };
    let nodata_t: Option<T> = nodata.and_then(|nd| NumCast::from(nd));

    let result: Array2<T> = py.allow_threads(move || {
        sample_with_grid(
            &src_array.view(),
            &col_grid,
            &row_grid,
            method,
            nodata_t,
            fill,
        )
    });

    Ok(PyArray2::from_owned_array(py, result).into_any().unbind())
}

/// Sample a source array at pre-computed coordinate locations.
fn sample_with_grid<T>(
    src: &ArrayView2<'_, T>,
    col_grid: &Array2<f64>,
    row_grid: &Array2<f64>,
    method: ResamplingMethod,
    nodata: Option<T>,
    fill: T,
) -> Array2<T>
where
    T: Copy + NumCast + PartialEq + Default + Send + Sync,
{
    let dst_shape = (col_grid.nrows(), col_grid.ncols());
    let mut dst = Array2::from_elem(dst_shape, fill);

    let radius = method.kernel_radius();
    let src_rows_f = src.nrows() as f64;
    let src_cols_f = src.ncols() as f64;
    // Default scale for average kernel (1:1)
    let scale = (1.0, 1.0);

    for row in 0..dst_shape.0 {
        for col in 0..dst_shape.1 {
            let src_col = col_grid[(row, col)];
            let src_row = row_grid[(row, col)];

            if src_col.is_nan() || src_row.is_nan() {
                continue;
            }

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
                ResamplingMethod::Cubic => resample::cubic::sample(src, src_col, src_row, nodata),
                ResamplingMethod::Lanczos => {
                    resample::lanczos::sample(src, src_col, src_row, nodata)
                }
                ResamplingMethod::Average => {
                    resample::average::sample(src, src_col, src_row, nodata, scale)
                }
            };

            if let Some(v) = val {
                dst[(row, col)] = v;
            }
        }
    }

    dst
}
