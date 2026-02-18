//! Albers Equal Area Conic projection.
//!
//! Area-preserving conic using `qsfn` from common.rs.

use crate::error::ProjError;
use crate::proj::common::{msfn, qsfn};
use crate::proj::ellipsoid::Ellipsoid;
use crate::proj::Projection;

pub struct AlbersEqualArea {
    ellipsoid: Ellipsoid,
    lon0: f64,
    n: f64,
    c: f64,
    rho0: f64,
    false_easting: f64,
    false_northing: f64,
}

impl AlbersEqualArea {
    pub fn new(
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
        let q0 = qsfn(lat0, e);
        let q1 = qsfn(lat1, e);
        let q2 = qsfn(lat2, e);

        let n = if (lat1 - lat2).abs() > 1e-10 {
            (m1 * m1 - m2 * m2) / (q2 - q1)
        } else {
            lat1.sin()
        };

        let c = m1 * m1 + n * q1;
        let rho0 = ellipsoid.a * (c - n * q0).abs().sqrt() / n;

        Self {
            ellipsoid,
            lon0,
            n,
            c,
            rho0,
            false_easting,
            false_northing,
        }
    }
}

impl Projection for AlbersEqualArea {
    fn forward(&self, lon: f64, lat: f64) -> Result<(f64, f64), ProjError> {
        let e = self.ellipsoid.eccentricity();
        let q = qsfn(lat, e);
        let theta = self.n * (lon - self.lon0);
        let rho = self.ellipsoid.a * (self.c - self.n * q).abs().sqrt() / self.n;

        let x = rho * theta.sin() + self.false_easting;
        let y = self.rho0 - rho * theta.cos() + self.false_northing;
        Ok((x, y))
    }

    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), ProjError> {
        let x_ = x - self.false_easting;
        let y_ = self.rho0 - (y - self.false_northing);

        let (xn, yn) = if self.n < 0.0 { (-x_, -y_) } else { (x_, y_) };

        let rho = (xn * xn + yn * yn).sqrt();
        let theta = xn.atan2(yn);

        let q = (self.c - (rho * self.n / self.ellipsoid.a).powi(2)) / self.n;

        // Inverse of qsfn: find φ from q by Newton iteration
        let e = self.ellipsoid.eccentricity();
        let e2 = self.ellipsoid.e2;
        let mut lat = (q / 2.0).asin(); // initial guess

        for _ in 0..15 {
            let sin_lat = lat.sin();
            let esinlat = e * sin_lat;
            let one_minus = 1.0 - esinlat * esinlat;
            let q_est = qsfn(lat, e);
            let dq_dphi = (1.0 - e2) * 2.0 * lat.cos() / (one_minus * one_minus);
            let delta = (q - q_est) / dq_dphi;
            lat += delta;
            if delta.abs() < 1e-12 {
                break;
            }
        }

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
    fn test_roundtrip() {
        // USGS Albers (EPSG:5070): lat1=29.5°, lat2=45.5°, lat0=23°, lon0=-96°
        let proj = AlbersEqualArea::new(
            WGS84,
            (-96.0_f64).to_radians(),
            23.0_f64.to_radians(),
            29.5_f64.to_radians(),
            45.5_f64.to_radians(),
            0.0,
            0.0,
        );

        let cases: &[(f64, f64)] = &[
            (-96.0, 23.0),  // origin
            (-96.0, 39.0),  // on central meridian
            (-74.0, 40.7),  // NYC
            (-87.6, 41.9),  // Chicago
            (-118.2, 34.0), // LA
            (-122.4, 37.8), // SF
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
    fn test_origin() {
        let proj = AlbersEqualArea::new(
            WGS84,
            (-96.0_f64).to_radians(),
            23.0_f64.to_radians(),
            29.5_f64.to_radians(),
            45.5_f64.to_radians(),
            0.0,
            0.0,
        );

        let (x, y) = proj
            .forward((-96.0_f64).to_radians(), 23.0_f64.to_radians())
            .unwrap();
        assert_relative_eq!(x, 0.0, epsilon = 1.0);
        assert_relative_eq!(y, 0.0, epsilon = 1.0);
    }

    #[test]
    fn test_with_false_easting_northing() {
        // EPSG:5070 parameters: NAD83 / Conus Albers
        let proj = AlbersEqualArea::new(
            WGS84,
            (-96.0_f64).to_radians(),
            23.0_f64.to_radians(),
            29.5_f64.to_radians(),
            45.5_f64.to_radians(),
            0.0,
            0.0,
        );

        // Central meridian at lat0 should give x=FE, y=FN
        let (x, y) = proj
            .forward((-96.0_f64).to_radians(), 23.0_f64.to_radians())
            .unwrap();
        assert_relative_eq!(x, 0.0, epsilon = 1.0);
        assert_relative_eq!(y, 0.0, epsilon = 1.0);

        // Point off-center should roundtrip
        let lon = (-80.0_f64).to_radians();
        let lat = 35.0_f64.to_radians();
        let (x, y) = proj.forward(lon, lat).unwrap();
        let (lon2, lat2) = proj.inverse(x, y).unwrap();
        assert_relative_eq!(lon2, lon, epsilon = 1e-9);
        assert_relative_eq!(lat2, lat, epsilon = 1e-9);
    }
}
