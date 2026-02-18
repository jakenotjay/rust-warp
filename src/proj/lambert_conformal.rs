//! Lambert Conformal Conic projection — 1SP and 2SP variants.
//!
//! Uses `tsfn`, `msfn`, `phi_from_ts` from common.rs.

use crate::error::ProjError;
use crate::proj::common::{msfn, phi_from_ts, tsfn};
use crate::proj::ellipsoid::Ellipsoid;
use crate::proj::Projection;

pub struct LambertConformalConic {
    ellipsoid: Ellipsoid,
    lon0: f64,
    n: f64,     // cone constant
    f_val: f64, // F = m₁/(n·t₁ⁿ)
    rho0: f64,  // ρ₀ = a·F·t₀ⁿ
    false_easting: f64,
    false_northing: f64,
}

impl LambertConformalConic {
    /// Create a Lambert Conformal Conic with two standard parallels (2SP).
    pub fn new_2sp(
        ellipsoid: Ellipsoid,
        lon0: f64,
        lat0: f64,
        lat1: f64,
        lat2: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let e = ellipsoid.eccentricity();
        let e2 = ellipsoid.e2;

        let m1 = msfn(lat1, e2);
        let m2 = msfn(lat2, e2);
        let t0 = tsfn(lat0, e);
        let t1 = tsfn(lat1, e);
        let t2 = tsfn(lat2, e);

        let n = if (lat1 - lat2).abs() > 1e-10 {
            (m1.ln() - m2.ln()) / (t1.ln() - t2.ln())
        } else {
            lat1.sin()
        };

        let f_val = m1 / (n * t1.powf(n));
        let rho0 = ellipsoid.a * f_val * t0.powf(n);

        Self {
            ellipsoid,
            lon0,
            n,
            f_val,
            rho0,
            false_easting,
            false_northing,
        }
    }

    /// Create a Lambert Conformal Conic with one standard parallel (1SP).
    pub fn new_1sp(
        ellipsoid: Ellipsoid,
        lon0: f64,
        lat0: f64,
        k0: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let e = ellipsoid.eccentricity();
        let e2 = ellipsoid.e2;

        let n = lat0.sin();
        let m0 = msfn(lat0, e2);
        let t0 = tsfn(lat0, e);

        let f_val = m0 / (n * t0.powf(n)) * k0;
        let rho0 = ellipsoid.a * f_val * t0.powf(n);

        Self {
            ellipsoid,
            lon0,
            n,
            f_val,
            rho0,
            false_easting,
            false_northing,
        }
    }
}

impl Projection for LambertConformalConic {
    fn forward(&self, lon: f64, lat: f64) -> Result<(f64, f64), ProjError> {
        let e = self.ellipsoid.eccentricity();
        let t = tsfn(lat, e);
        let rho = self.ellipsoid.a * self.f_val * t.powf(self.n);
        let theta = self.n * (lon - self.lon0);

        let x = rho * theta.sin() + self.false_easting;
        let y = self.rho0 - rho * theta.cos() + self.false_northing;
        Ok((x, y))
    }

    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), ProjError> {
        let x_ = x - self.false_easting;
        let y_ = self.rho0 - (y - self.false_northing);

        // For n < 0, flip signs before computing angle and radius
        let (xn, yn) = if self.n < 0.0 { (-x_, -y_) } else { (x_, y_) };

        let rho = (xn * xn + yn * yn).sqrt();
        let theta = xn.atan2(yn); // atan2(x', y') — note order!

        let e = self.ellipsoid.eccentricity();
        let ts = (rho / (self.ellipsoid.a * self.f_val)).powf(1.0 / self.n);
        let lat = phi_from_ts(ts, e);
        let lon = self.lon0 + theta / self.n;

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

    #[test]
    fn test_2sp_roundtrip() {
        // France Lambert (similar to EPSG:2154: RGF93 / Lambert-93)
        // lat1=44°, lat2=49°, lat0=46.5°, lon0=3°
        let proj = LambertConformalConic::new_2sp(
            WGS84,
            3.0_f64.to_radians(),
            46.5_f64.to_radians(),
            44.0_f64.to_radians(),
            49.0_f64.to_radians(),
            700_000.0,
            6_600_000.0,
        );

        let cases: &[(f64, f64)] = &[
            (3.0, 46.5),    // origin
            (2.35, 48.86),  // Paris
            (-1.55, 47.22), // Nantes
            (7.75, 48.58),  // Strasbourg
        ];
        for &(lon_deg, lat_deg) in cases {
            let lon = lon_deg.to_radians();
            let lat = lat_deg.to_radians();
            let (x, y) = proj.forward(lon, lat).unwrap();
            let (lon2, lat2) = proj.inverse(x, y).unwrap();
            assert_relative_eq!(lon2, lon, epsilon = 1e-9);
            assert_relative_eq!(lat2, lat, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_1sp_roundtrip() {
        // LCC 1SP with lat0=45°, k0=1.0
        let proj = LambertConformalConic::new_1sp(
            WGS84,
            0.0_f64.to_radians(),
            45.0_f64.to_radians(),
            1.0,
            0.0,
            0.0,
        );

        let lon = 5.0_f64.to_radians();
        let lat = 48.0_f64.to_radians();
        let (x, y) = proj.forward(lon, lat).unwrap();
        let (lon2, lat2) = proj.inverse(x, y).unwrap();
        assert_relative_eq!(lon2, lon, epsilon = 1e-9);
        assert_relative_eq!(lat2, lat, epsilon = 1e-9);
    }

    #[test]
    fn test_origin_point() {
        let proj = LambertConformalConic::new_2sp(
            WGS84,
            3.0_f64.to_radians(),
            46.5_f64.to_radians(),
            44.0_f64.to_radians(),
            49.0_f64.to_radians(),
            700_000.0,
            6_600_000.0,
        );

        // At the origin, x should be FE, y should be FN
        let (x, y) = proj
            .forward(3.0_f64.to_radians(), 46.5_f64.to_radians())
            .unwrap();
        assert_relative_eq!(x, 700_000.0, epsilon = 1.0);
        assert_relative_eq!(y, 6_600_000.0, epsilon = 1.0);
    }

    #[test]
    fn test_us_state_plane_like() {
        // US State Plane-like: lat1=33°, lat2=45°, lat0=39°, lon0=-96°
        let proj = LambertConformalConic::new_2sp(
            WGS84,
            (-96.0_f64).to_radians(),
            39.0_f64.to_radians(),
            33.0_f64.to_radians(),
            45.0_f64.to_radians(),
            0.0,
            0.0,
        );

        let cases: &[(f64, f64)] = &[
            (-96.0, 39.0),  // origin
            (-74.0, 40.7),  // NYC
            (-87.6, 41.9),  // Chicago
            (-118.2, 34.0), // LA
        ];
        for &(lon_deg, lat_deg) in cases {
            let lon = lon_deg.to_radians();
            let lat = lat_deg.to_radians();
            let (x, y) = proj.forward(lon, lat).unwrap();
            let (lon2, lat2) = proj.inverse(x, y).unwrap();
            assert_relative_eq!(lon2, lon, epsilon = 1e-9);
            assert_relative_eq!(lat2, lat, epsilon = 1e-9);
        }
    }
}
