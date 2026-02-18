pub mod albers_equal_area;
pub mod approx;
pub mod common;
pub mod crs;
pub mod ellipsoid;
pub mod equirectangular;
pub mod lambert_conformal;
pub mod mercator;
pub mod pipeline;
pub mod sinusoidal;
pub mod stereographic;
pub mod transverse_mercator;

use crate::error::ProjError;

/// Trait for map projections supporting forward and inverse transforms.
pub trait Projection: Send + Sync {
    /// Forward: (lon_rad, lat_rad) -> (easting, northing)
    fn forward(&self, lon: f64, lat: f64) -> Result<(f64, f64), ProjError>;

    /// Inverse: (easting, northing) -> (lon_rad, lat_rad)
    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), ProjError>;

    /// Batch forward transform (default: loop, override for SIMD).
    fn forward_batch(&self, coords: &mut [(f64, f64)]) -> Result<(), ProjError> {
        for c in coords.iter_mut() {
            *c = self.forward(c.0, c.1)?;
        }
        Ok(())
    }

    /// Batch inverse transform.
    fn inverse_batch(&self, coords: &mut [(f64, f64)]) -> Result<(), ProjError> {
        for c in coords.iter_mut() {
            *c = self.inverse(c.0, c.1)?;
        }
        Ok(())
    }

    fn ellipsoid(&self) -> &ellipsoid::Ellipsoid;
}
