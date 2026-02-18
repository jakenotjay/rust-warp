//! Equirectangular (Plate Carrée) projection.
//!
//! forward: x = a·(λ - λ₀)·cos(φ₁), y = a·(φ - φ₀)
//! inverse: λ = λ₀ + x/(a·cos(φ₁)), φ = φ₀ + y/a

use crate::error::ProjError;
use crate::proj::ellipsoid::Ellipsoid;
use crate::proj::Projection;

pub struct Equirectangular {
    ellipsoid: Ellipsoid,
    lon0: f64,
    lat0: f64,
    cos_lat_ts: f64,
    false_easting: f64,
    false_northing: f64,
}

impl Equirectangular {
    pub fn new(
        ellipsoid: Ellipsoid,
        lon0: f64,
        lat0: f64,
        lat_ts: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        Self {
            ellipsoid,
            lon0,
            lat0,
            cos_lat_ts: lat_ts.cos(),
            false_easting,
            false_northing,
        }
    }

    /// Create an identity-like equirectangular for EPSG:4326 treated as projected.
    /// lon0=0, lat0=0, lat_ts=0, FE=0, FN=0, using WGS84.
    pub fn epsg_4326() -> Self {
        use crate::proj::ellipsoid::WGS84;
        Self::new(WGS84, 0.0, 0.0, 0.0, 0.0, 0.0)
    }
}

impl Projection for Equirectangular {
    fn forward(&self, lon: f64, lat: f64) -> Result<(f64, f64), ProjError> {
        let x = self.ellipsoid.a * (lon - self.lon0) * self.cos_lat_ts + self.false_easting;
        let y = self.ellipsoid.a * (lat - self.lat0) + self.false_northing;
        Ok((x, y))
    }

    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), ProjError> {
        let lon = self.lon0 + (x - self.false_easting) / (self.ellipsoid.a * self.cos_lat_ts);
        let lat = self.lat0 + (y - self.false_northing) / self.ellipsoid.a;
        Ok((lon, lat))
    }

    fn ellipsoid(&self) -> &Ellipsoid {
        &self.ellipsoid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::ellipsoid::WGS84;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_roundtrip() {
        let proj = Equirectangular::new(WGS84, 0.0, 0.0, 0.0, 0.0, 0.0);
        let lon = 10.0_f64.to_radians();
        let lat = 45.0_f64.to_radians();
        let (x, y) = proj.forward(lon, lat).unwrap();
        let (lon2, lat2) = proj.inverse(x, y).unwrap();
        assert_relative_eq!(lon2, lon, epsilon = 1e-12);
        assert_relative_eq!(lat2, lat, epsilon = 1e-12);
    }

    #[test]
    fn test_origin() {
        let proj = Equirectangular::new(WGS84, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (x, y) = proj.forward(0.0, 0.0).unwrap();
        assert_relative_eq!(x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(y, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_with_standard_parallel() {
        // With standard parallel at 30°, x should be scaled by cos(30°)
        let lat_ts = 30.0_f64.to_radians();
        let proj = Equirectangular::new(WGS84, 0.0, 0.0, lat_ts, 0.0, 0.0);
        let lon = 1.0_f64.to_radians();
        let (x, _) = proj.forward(lon, 0.0).unwrap();
        let expected_x = WGS84.a * lon * lat_ts.cos();
        assert_relative_eq!(x, expected_x, epsilon = 1e-6);
    }

    #[test]
    fn test_dateline() {
        let proj = Equirectangular::new(WGS84, 0.0, 0.0, 0.0, 0.0, 0.0);
        // ±180°
        let lon_east = PI;
        let lon_west = -PI;
        let (xe, _) = proj.forward(lon_east, 0.0).unwrap();
        let (xw, _) = proj.forward(lon_west, 0.0).unwrap();
        assert_relative_eq!(xe, -xw, epsilon = 1e-6);
    }

    #[test]
    fn test_epsg_4326_identity_like() {
        // EPSG:4326 as projected: forward(lon_rad, lat_rad) should give
        // x ≈ a * lon_rad, y ≈ a * lat_rad
        let proj = Equirectangular::epsg_4326();
        let lon = 15.0_f64.to_radians();
        let lat = 52.0_f64.to_radians();
        let (x, y) = proj.forward(lon, lat).unwrap();
        assert_relative_eq!(x, WGS84.a * lon, epsilon = 1e-6);
        assert_relative_eq!(y, WGS84.a * lat, epsilon = 1e-6);
    }
}
