pub mod common;
pub mod crs;
pub mod ellipsoid;

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
