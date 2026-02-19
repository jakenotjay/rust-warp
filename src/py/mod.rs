use pyo3::prelude::*;

mod plan;
mod reproject;
mod transform;

/// Register all Python-visible functions and types.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(reproject::reproject_array, m)?)?;
    m.add_function(wrap_pyfunction!(transform::transform_points, m)?)?;
    m.add_function(wrap_pyfunction!(plan::plan_reproject, m)?)?;
    Ok(())
}

/// Smoke-test function to verify the extension loads.
#[pyfunction]
fn hello() -> String {
    "Hello from rust-warp!".to_string()
}
