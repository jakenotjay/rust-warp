//! Mercator projection — Normal (ellipsoidal) and Web Mercator (EPSG:3857).
//!
//! Normal Mercator (ellipsoidal with standard parallel):
//!   forward: x = a·k₀·(λ - λ₀), y = a·k₀·ln(tsfn(π/4 + φ/2, e))  [note: -ln(tsfn(φ,e))]
//!   inverse: λ = λ₀ + x/(a·k₀), φ = phi_from_ts(exp(-y/(a·k₀)), e)
//!
//! Web Mercator (EPSG:3857, spherical):
//!   forward: x = a·(λ - λ₀), y = a·ln(tan(π/4 + φ/2))
//!   inverse: λ = λ₀ + x/a, φ = 2·atan(exp(y/a)) - π/2

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use crate::error::ProjError;
use crate::proj::common::{msfn, phi_from_ts, tsfn};
use crate::proj::ellipsoid::{Ellipsoid, WGS84};
use crate::proj::Projection;

/// Ellipsoidal Mercator projection with a standard parallel.
pub struct Mercator {
    ellipsoid: Ellipsoid,
    lon0: f64,
    k0: f64,
    false_easting: f64,
    false_northing: f64,
}

impl Mercator {
    pub fn new(
        ellipsoid: Ellipsoid,
        lon0: f64,
        lat_ts: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        // Scale factor from standard parallel
        let k0 = msfn(lat_ts, ellipsoid.e2);
        Self {
            ellipsoid,
            lon0,
            k0,
            false_easting,
            false_northing,
        }
    }
}

impl Projection for Mercator {
    fn forward(&self, lon: f64, lat: f64) -> Result<(f64, f64), ProjError> {
        let e = self.ellipsoid.eccentricity();
        let x = self.ellipsoid.a * self.k0 * (lon - self.lon0) + self.false_easting;
        // y = a * k0 * (-ln(tsfn(φ, e)))
        // tsfn(φ,e) = tan(π/4 - φ/2) / ((1-e·sinφ)/(1+e·sinφ))^(e/2)
        // For positive latitudes, tsfn < 1 so -ln(tsfn) > 0 → y > 0
        let y = self.ellipsoid.a * self.k0 * (-tsfn(lat, e).ln()) + self.false_northing;
        Ok((x, y))
    }

    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), ProjError> {
        let e = self.ellipsoid.eccentricity();
        let lon = self.lon0 + (x - self.false_easting) / (self.ellipsoid.a * self.k0);
        let ts = (-(y - self.false_northing) / (self.ellipsoid.a * self.k0)).exp();
        let lat = phi_from_ts(ts, e);
        Ok((lon, lat))
    }

    fn ellipsoid(&self) -> &Ellipsoid {
        &self.ellipsoid
    }
}

/// Web Mercator projection (EPSG:3857) — spherical approximation.
pub struct WebMercator {
    ellipsoid: Ellipsoid,
    lon0: f64,
}

/// Maximum latitude for Web Mercator (≈85.0511°), where the projection
/// is bounded to a square.
const MAX_LAT_3857: f64 = 1.4844222297453324; // atan(sinh(π)) in radians

impl WebMercator {
    /// Create a Web Mercator projection with EPSG:3857 parameters.
    pub fn new() -> Self {
        Self {
            ellipsoid: WGS84,
            lon0: 0.0,
        }
    }
}

impl Default for WebMercator {
    fn default() -> Self {
        Self::new()
    }
}

impl Projection for WebMercator {
    fn forward(&self, lon: f64, lat: f64) -> Result<(f64, f64), ProjError> {
        // Clamp latitude to valid range
        let lat = lat.clamp(-MAX_LAT_3857, MAX_LAT_3857);
        let x = self.ellipsoid.a * (lon - self.lon0);
        let y = self.ellipsoid.a * (FRAC_PI_4 + lat / 2.0).tan().ln();
        Ok((x, y))
    }

    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), ProjError> {
        let lon = self.lon0 + x / self.ellipsoid.a;
        let lat = 2.0 * (y / self.ellipsoid.a).exp().atan() - FRAC_PI_2;
        Ok((lon, lat))
    }

    fn ellipsoid(&self) -> &Ellipsoid {
        &self.ellipsoid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_web_mercator_origin() {
        let proj = WebMercator::new();
        let (x, y) = proj.forward(0.0, 0.0).unwrap();
        assert_relative_eq!(x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(y, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_web_mercator_roundtrip() {
        let proj = WebMercator::new();
        let cases: &[(f64, f64)] = &[
            (0.0, 0.0),
            (10.0, 45.0),
            (-73.9857, 40.7484), // NYC
            (139.6917, 35.6895), // Tokyo
            (-180.0, 0.0),
            (180.0, 0.0),
        ];
        for &(lon_deg, lat_deg) in cases {
            let lon = lon_deg.to_radians();
            let lat = lat_deg.to_radians();
            let (x, y) = proj.forward(lon, lat).unwrap();
            let (lon2, lat2) = proj.inverse(x, y).unwrap();
            assert_relative_eq!(lon2, lon, epsilon = 1e-10);
            assert_relative_eq!(lat2, lat, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_web_mercator_epsg3857_reference() {
        // Known EPSG:3857 values:
        // (0°, 0°) → (0, 0)
        // (180°, 0°) → (20037508.34, 0)
        let proj = WebMercator::new();
        let (x, _) = proj.forward(PI, 0.0).unwrap();
        assert_relative_eq!(x, 20_037_508.342_789_244, epsilon = 0.01);
    }

    #[test]
    fn test_web_mercator_polar_clamp() {
        let proj = WebMercator::new();
        // Latitude beyond ±85.06° should be clamped, not produce infinity
        let (_, y) = proj.forward(0.0, FRAC_PI_2).unwrap();
        assert!(y.is_finite(), "y should be finite at pole, got {y}");
    }

    #[test]
    fn test_ellipsoidal_mercator_roundtrip() {
        let proj = Mercator::new(WGS84, 0.0, 0.0, 0.0, 0.0);
        let cases: &[(f64, f64)] = &[
            (0.0, 0.0),
            (10.0, 45.0),
            (-73.9857, 40.7484),
            (139.6917, 35.6895),
        ];
        for &(lon_deg, lat_deg) in cases {
            let lon = lon_deg.to_radians();
            let lat = lat_deg.to_radians();
            let (x, y) = proj.forward(lon, lat).unwrap();
            let (lon2, lat2) = proj.inverse(x, y).unwrap();
            assert_relative_eq!(lon2, lon, epsilon = 1e-10);
            assert_relative_eq!(lat2, lat, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ellipsoidal_mercator_origin() {
        let proj = Mercator::new(WGS84, 0.0, 0.0, 0.0, 0.0);
        let (x, y) = proj.forward(0.0, 0.0).unwrap();
        assert_relative_eq!(x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(y, 0.0, epsilon = 1e-6);
    }
}
