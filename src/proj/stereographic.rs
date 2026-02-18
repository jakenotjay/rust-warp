//! Stereographic projection — Polar and Oblique variants.
//!
//! Polar Stereographic: EPSG:3031 (Antarctic), EPSG:3413 (Arctic)
//! Oblique Stereographic: general case

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use crate::error::ProjError;
use crate::proj::common::{phi_from_ts, tsfn};
use crate::proj::ellipsoid::Ellipsoid;
use crate::proj::Projection;

/// Polar Stereographic projection.
pub struct PolarStereographic {
    ellipsoid: Ellipsoid,
    lon0: f64,
    is_north: bool,
    false_easting: f64,
    false_northing: f64,
    // Precomputed
    akm: f64, // a * k0 * m_c / t_c for variant A (lat_ts given), or a * 2 * k0 / sqrt((1+e)^(1+e) * (1-e)^(1-e))
}

impl PolarStereographic {
    /// Create a Polar Stereographic from latitude of true scale.
    pub fn new(
        ellipsoid: Ellipsoid,
        lon0: f64,
        lat_ts: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let is_north = lat_ts >= 0.0;
        let e = ellipsoid.eccentricity();
        let e2 = ellipsoid.e2;

        // If lat_ts is at the pole, use the simpler formula
        let akm = if (lat_ts.abs() - FRAC_PI_2).abs() < 1e-10 {
            // Variant B: lat_ts at pole, k0 given (usually 0.994 for UPS)
            let k0 = 0.994; // default for UPS
            let ep = (1.0 + e).powf(1.0 + e) * (1.0 - e).powf(1.0 - e);
            ellipsoid.a * 2.0 * k0 / ep.sqrt()
        } else {
            // Variant A: lat_ts given
            let sin_ts = lat_ts.abs().sin();
            let cos_ts = lat_ts.abs().cos();
            let m_c = cos_ts / (1.0 - e2 * sin_ts * sin_ts).sqrt();
            let t_c = tsfn(lat_ts.abs(), e);
            ellipsoid.a * m_c / t_c
        };

        Self {
            ellipsoid,
            lon0,
            is_north,
            false_easting,
            false_northing,
            akm,
        }
    }

    /// EPSG:3031 — Antarctic Polar Stereographic
    pub fn antarctic() -> Self {
        use crate::proj::ellipsoid::WGS84;
        Self::new(
            WGS84,
            0.0,                      // lon0 = 0°
            (-71.0_f64).to_radians(), // lat_ts = -71°
            0.0,
            0.0,
        )
    }

    /// EPSG:3413 — Arctic NSIDC Polar Stereographic North
    pub fn arctic() -> Self {
        use crate::proj::ellipsoid::WGS84;
        Self::new(
            WGS84,
            (-45.0_f64).to_radians(), // lon0 = -45°
            70.0_f64.to_radians(),    // lat_ts = 70°
            0.0,
            0.0,
        )
    }
}

impl Projection for PolarStereographic {
    fn forward(&self, lon: f64, lat: f64) -> Result<(f64, f64), ProjError> {
        let e = self.ellipsoid.eccentricity();

        let (lat_adj, sign) = if self.is_north {
            (lat, 1.0)
        } else {
            (-lat, -1.0)
        };

        let t = tsfn(lat_adj, e);
        let rho = self.akm * t;
        let dlam = lon - self.lon0;

        let x = sign * rho * dlam.sin() + self.false_easting;
        let y = -sign * rho * dlam.cos() + self.false_northing;
        Ok((x, y))
    }

    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), ProjError> {
        let e = self.ellipsoid.eccentricity();

        let x_ = x - self.false_easting;
        let y_ = y - self.false_northing;

        let (x_adj, y_adj) = if self.is_north { (x_, -y_) } else { (-x_, y_) };

        let rho = (x_adj * x_adj + y_adj * y_adj).sqrt();
        let t = rho / self.akm;
        let lat_adj = phi_from_ts(t, e);

        let lon = self.lon0 + x_adj.atan2(y_adj);
        let lat = if self.is_north { lat_adj } else { -lat_adj };

        Ok((lon, lat))
    }

    fn ellipsoid(&self) -> &Ellipsoid {
        &self.ellipsoid
    }
}

/// Oblique Stereographic (Double) projection.
pub struct ObliqueStereographic {
    ellipsoid: Ellipsoid,
    lon0: f64,
    k0: f64,
    false_easting: f64,
    false_northing: f64,
    // Precomputed conformal sphere parameters
    n_conf: f64,
    sin_chi0: f64,
    cos_chi0: f64,
    r_sphere: f64, // radius of conformal sphere
}

impl ObliqueStereographic {
    pub fn new(
        ellipsoid: Ellipsoid,
        lon0: f64,
        lat0: f64,
        k0: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let e2 = ellipsoid.e2;
        let e = ellipsoid.eccentricity();
        let sin0 = lat0.sin();
        // Conformal sphere parameters (Roussilhe)
        let n_conf = (1.0 - e2 * sin0 * sin0).sqrt();
        let s1 = (1.0 + sin0) / (1.0 - sin0);
        let s2 = (1.0 - e * sin0) / (1.0 + e * sin0);
        let w1 = (s1 * s2.powf(e)).powf(n_conf / 2.0);

        // R = sqrt(rho_0 * nu_0) = a * sqrt(1-e²) / (1 - e² * sin²φ₀)
        let r_sphere = ellipsoid.a * (1.0 - e2).sqrt() / (1.0 - e2 * sin0 * sin0);

        // chi0 is the conformal latitude of lat0
        let chi0 = (2.0 * (w1 * ((FRAC_PI_4 + lat0 / 2.0).tan())).atan() - FRAC_PI_2)
            .clamp(-FRAC_PI_2, FRAC_PI_2);

        let sin_chi0 = chi0.sin();
        let cos_chi0 = chi0.cos();

        Self {
            ellipsoid,
            lon0,
            k0,
            false_easting,
            false_northing,
            n_conf,
            sin_chi0,
            cos_chi0,
            r_sphere,
        }
    }

    fn geodetic_to_conformal(&self, lon: f64, lat: f64) -> (f64, f64) {
        let e = self.ellipsoid.eccentricity();
        let sin_lat = lat.sin();
        let s1 = (1.0 + sin_lat) / (1.0 - sin_lat);
        let s2 = (1.0 - e * sin_lat) / (1.0 + e * sin_lat);
        let w = (s1 * s2.powf(e)).powf(self.n_conf / 2.0);

        let chi = (2.0 * (w * (FRAC_PI_4 + lat / 2.0).tan()).atan() - FRAC_PI_2)
            .clamp(-FRAC_PI_2, FRAC_PI_2);
        let lambda = self.n_conf * (lon - self.lon0);
        (lambda, chi)
    }

    fn conformal_to_geodetic(&self, lambda: f64, chi: f64) -> (f64, f64) {
        let e = self.ellipsoid.eccentricity();
        let lon = lambda / self.n_conf + self.lon0;

        // Inverse: chi → φ by iteration
        let mut phi = chi;
        for _ in 0..15 {
            let sin_phi = phi.sin();
            let s1 = (1.0 + sin_phi) / (1.0 - sin_phi);
            let s2 = (1.0 - e * sin_phi) / (1.0 + e * sin_phi);
            let w = (s1 * s2.powf(e)).powf(self.n_conf / 2.0);
            let chi_est = 2.0 * (w * (FRAC_PI_4 + phi / 2.0).tan()).atan() - FRAC_PI_2;
            let dphi = (chi - chi_est) * (1.0 - self.ellipsoid.e2 * sin_phi * sin_phi).powi(2)
                / ((1.0 - self.ellipsoid.e2) * phi.cos());
            // Simpler iteration: φ_{n+1} = φ_n + (chi - chi(φ_n)) * correction
            phi += dphi;
            if dphi.abs() < 1e-12 {
                break;
            }
        }
        (lon, phi)
    }
}

impl Projection for ObliqueStereographic {
    fn forward(&self, lon: f64, lat: f64) -> Result<(f64, f64), ProjError> {
        let (lambda, chi) = self.geodetic_to_conformal(lon, lat);

        let sin_chi = chi.sin();
        let cos_chi = chi.cos();
        let cos_lambda = lambda.cos();
        let sin_lambda = lambda.sin();

        let b_denom = 1.0 + self.sin_chi0 * sin_chi + self.cos_chi0 * cos_chi * cos_lambda;
        let b = 2.0 * self.r_sphere * self.k0 / b_denom;

        let x = b * cos_chi * sin_lambda + self.false_easting;
        let y = b * (self.cos_chi0 * sin_chi - self.sin_chi0 * cos_chi * cos_lambda)
            + self.false_northing;

        Ok((x, y))
    }

    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), ProjError> {
        let x_ = (x - self.false_easting) / (2.0 * self.r_sphere * self.k0);
        let y_ = (y - self.false_northing) / (2.0 * self.r_sphere * self.k0);

        let _ = (x_, y_); // normalized versions unused; work in actual distances
        let xd = x - self.false_easting;
        let yd = y - self.false_northing;
        let rho = (xd * xd + yd * yd).sqrt();
        let two_rk = 2.0 * self.r_sphere * self.k0;
        let c = 2.0 * (rho / two_rk).atan();

        let sin_c = c.sin();
        let cos_c = c.cos();

        let chi = if rho < 1e-10 {
            self.sin_chi0.asin()
        } else {
            (cos_c * self.sin_chi0 + yd * sin_c * self.cos_chi0 / rho).asin()
        };

        let lambda = if self.cos_chi0.abs() < 1e-10 {
            // Polar case
            if self.sin_chi0 > 0.0 {
                xd.atan2(-yd)
            } else {
                xd.atan2(yd)
            }
        } else {
            (xd * sin_c).atan2(rho * self.cos_chi0 * cos_c - yd * self.sin_chi0 * sin_c)
        };

        let (lon, lat) = self.conformal_to_geodetic(lambda, chi);
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
    fn test_polar_antarctic_roundtrip() {
        let proj = PolarStereographic::antarctic();
        let cases: &[(f64, f64)] = &[(0.0, -75.0), (90.0, -80.0), (-120.0, -70.0), (45.0, -65.0)];
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
    fn test_polar_arctic_roundtrip() {
        let proj = PolarStereographic::arctic();
        let cases: &[(f64, f64)] = &[(-45.0, 75.0), (0.0, 80.0), (90.0, 85.0), (-90.0, 70.0)];
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
    fn test_polar_south_pole() {
        let proj = PolarStereographic::antarctic();
        let (x, y) = proj.forward(0.0, -FRAC_PI_2).unwrap();
        assert_relative_eq!(x, 0.0, epsilon = 1.0);
        assert_relative_eq!(y, 0.0, epsilon = 1.0);
    }

    #[test]
    fn test_polar_north_pole() {
        let proj = PolarStereographic::arctic();
        let (x, y) = proj.forward((-45.0_f64).to_radians(), FRAC_PI_2).unwrap();
        assert_relative_eq!(x, 0.0, epsilon = 1.0);
        assert_relative_eq!(y, 0.0, epsilon = 1.0);
    }

    #[test]
    fn test_oblique_roundtrip() {
        // Netherlands RD New-like: lat0=52.156, lon0=5.387, k0=0.9999079
        let proj = ObliqueStereographic::new(
            WGS84,
            5.387_638_89_f64.to_radians(),
            52.156_160_56_f64.to_radians(),
            0.999_907_9,
            155_000.0,
            463_000.0,
        );

        let cases: &[(f64, f64)] = &[
            (5.387, 52.156), // origin
            (4.9, 52.37),    // Amsterdam area
            (5.5, 51.44),    // Eindhoven area
        ];
        for &(lon_deg, lat_deg) in cases {
            let lon = lon_deg.to_radians();
            let lat = lat_deg.to_radians();
            let (x, y) = proj.forward(lon, lat).unwrap();
            let (lon2, lat2) = proj.inverse(x, y).unwrap();
            assert_relative_eq!(lon2, lon, epsilon = 1e-8);
            assert_relative_eq!(lat2, lat, epsilon = 1e-8);
        }
    }
}
