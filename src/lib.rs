use pyo3::prelude::*;

pub mod error;
pub mod affine;
pub mod proj;
pub mod resample;
pub mod warp;
pub mod chunk;
mod py;

/// A Python module implemented in Rust.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py::register(m)?;
    Ok(())
}
