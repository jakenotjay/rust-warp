use pyo3::prelude::*;

mod reproject;

/// Register all Python-visible functions and types.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(reproject::reproject_array, m)?)?;
    Ok(())
}

/// Smoke-test function to verify the extension loads.
#[pyfunction]
fn hello() -> String {
    "Hello from rust-warp!".to_string()
}
