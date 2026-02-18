use pyo3::prelude::*;

pub mod affine;
pub mod chunk;
pub mod error;
pub mod proj;
mod py;
pub mod resample;
pub mod warp;

/// A Python module implemented in Rust.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py::register(m)?;
    Ok(())
}
