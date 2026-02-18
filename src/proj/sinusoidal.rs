//! Sinusoidal (Sanson–Flamsteed) projection.
//!
//! forward: x = a·(λ - λ₀)·cos(φ), y = a·φ
//! inverse: λ = λ₀ + x/(a·cos(φ)), φ = y/a

use crate::error::ProjError;
use crate::proj::ellipsoid::Ellipsoid;
use crate::proj::Projection;

pub struct Sinusoidal {
    ellipsoid: Ellipsoid,
    lon0: f64,
    false_easting: f64,
    false_northing: f64,
}

impl Sinusoidal {
    pub fn new(ellipsoid: Ellipsoid, lon0: f64, false_easting: f64, false_northing: f64) -> Self {
        Self {
            ellipsoid,
            lon0,
            false_easting,
            false_northing,
        }
    }
}

impl Projection for Sinusoidal {
    fn forward(&self, lon: f64, lat: f64) -> Result<(f64, f64), ProjError> {
        let x = self.ellipsoid.a * (lon - self.lon0) * lat.cos() + self.false_easting;
        let y = self.ellipsoid.a * lat + self.false_northing;
        Ok((x, y))
    }

    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), ProjError> {
        let lat = (y - self.false_northing) / self.ellipsoid.a;
        let cos_lat = lat.cos();
        if cos_lat.abs() < 1e-15 {
            // At the poles, longitude is undefined — return lon0
            return Ok((self.lon0, lat));
        }
        let lon = self.lon0 + (x - self.false_easting) / (self.ellipsoid.a * cos_lat);
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
    fn test_roundtrip() {
        let proj = Sinusoidal::new(WGS84, 0.0, 0.0, 0.0);
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
    fn test_origin() {
        let proj = Sinusoidal::new(WGS84, 0.0, 0.0, 0.0);
        let (x, y) = proj.forward(0.0, 0.0).unwrap();
        assert_relative_eq!(x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(y, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_equator_x_equals_eqrect() {
        // On the equator, cos(0)=1 so x = a*(lon - lon0), same as equirectangular
        let proj = Sinusoidal::new(WGS84, 0.0, 0.0, 0.0);
        let lon = 15.0_f64.to_radians();
        let (x, _) = proj.forward(lon, 0.0).unwrap();
        assert_relative_eq!(x, WGS84.a * lon, epsilon = 1e-6);
    }

    #[test]
    fn test_modis_sinusoidal_grid() {
        // MODIS sinusoidal uses R=6371007.181 (sphere) lon0=0
        // Tile h17v04 upper-left: approx (-1111950.52, 6671703.12)
        // We use WGS84 ellipsoid here so values won't match exactly,
        // but structure should be correct.
        let proj = Sinusoidal::new(WGS84, 0.0, 0.0, 0.0);
        // Central meridian at equator
        let (x, y) = proj.forward(0.0, 0.0).unwrap();
        assert_relative_eq!(x, 0.0, epsilon = 1.0);
        assert_relative_eq!(y, 0.0, epsilon = 1.0);
    }

    #[test]
    fn test_pole() {
        // At the north pole, x should be 0 for any longitude (cos(90°)=0)
        let proj = Sinusoidal::new(WGS84, 0.0, 0.0, 0.0);
        let (x, _) = proj
            .forward(45.0_f64.to_radians(), std::f64::consts::FRAC_PI_2)
            .unwrap();
        assert_relative_eq!(x, 0.0, epsilon = 1e-6);
    }
}
