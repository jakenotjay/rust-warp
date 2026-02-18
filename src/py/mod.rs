use pyo3::prelude::*;

/// Register all Python-visible functions and types.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}

/// Smoke-test function to verify the extension loads.
#[pyfunction]
fn hello() -> String {
    "Hello from rust-warp!".to_string()
}
