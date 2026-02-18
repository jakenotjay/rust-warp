//! Transverse Mercator projection — Krüger n-series, 6th order.
//!
//! Implements the Karney (2011) formulation with 6th-order α/β series coefficients.
//! This is the projection underlying all UTM zones.

use crate::error::ProjError;
use crate::proj::ellipsoid::{Ellipsoid, WGS84};
use crate::proj::Projection;

pub struct TransverseMercator {
    ellipsoid: Ellipsoid,
    lon0: f64,
    k0: f64,
    false_easting: f64,
    false_northing: f64,
    // Precomputed constants
    a_hat: f64,      // A = a/(1+n) * (1 + n²/4 + n⁴/64)
    alpha: [f64; 6], // Forward series coefficients
    beta: [f64; 6],  // Inverse series coefficients
    m0: f64,         // Normalized meridional arc at lat0
}

impl TransverseMercator {
    pub fn new(
        ellipsoid: Ellipsoid,
        lon0: f64,
        lat0: f64,
        k0: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        let n = ellipsoid.n;
        let n2 = n * n;
        let n3 = n2 * n;
        let n4 = n3 * n;
        let n5 = n4 * n;
        let n6 = n5 * n;

        let a_hat = ellipsoid.a / (1.0 + n) * (1.0 + n2 / 4.0 + n4 / 64.0);

        let alpha = Self::alpha_coefficients(n, n2, n3, n4, n5, n6);
        let beta = Self::beta_coefficients(n, n2, n3, n4, n5, n6);

        let m0 = Self::meridional_arc_normalized(lat0, n);

        Self {
            ellipsoid,
            lon0,
            k0,
            false_easting,
            false_northing,
            a_hat,
            alpha,
            beta,
            m0,
        }
    }

    /// Create a Transverse Mercator for a UTM zone.
    pub fn utm_zone(zone: u8, north: bool) -> Self {
        let lon0 = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();
        let false_northing = if north { 0.0 } else { 10_000_000.0 };
        Self::new(WGS84, lon0, 0.0, 0.9996, 500_000.0, false_northing)
    }

    /// Forward series coefficients α₁..α₆ (Krüger, 6th order).
    fn alpha_coefficients(n: f64, n2: f64, n3: f64, n4: f64, n5: f64, n6: f64) -> [f64; 6] {
        [
            // α₁
            n / 2.0 - 2.0 / 3.0 * n2 + 5.0 / 16.0 * n3 + 41.0 / 180.0 * n4 - 127.0 / 288.0 * n5
                + 7891.0 / 37800.0 * n6,
            // α₂
            13.0 / 48.0 * n2 - 3.0 / 5.0 * n3 + 557.0 / 1440.0 * n4 + 281.0 / 630.0 * n5
                - 1983433.0 / 1935360.0 * n6,
            // α₃
            61.0 / 240.0 * n3 - 103.0 / 140.0 * n4
                + 15061.0 / 26880.0 * n5
                + 167603.0 / 181440.0 * n6,
            // α₄
            49561.0 / 161280.0 * n4 - 179.0 / 168.0 * n5 + 6601661.0 / 7257600.0 * n6,
            // α₅
            34729.0 / 80640.0 * n5 - 3418889.0 / 1995840.0 * n6,
            // α₆
            212378941.0 / 319334400.0 * n6,
        ]
    }

    /// Inverse series coefficients β₁..β₆ (Krüger, 6th order).
    fn beta_coefficients(n: f64, n2: f64, n3: f64, n4: f64, n5: f64, n6: f64) -> [f64; 6] {
        [
            // β₁
            n / 2.0 - 2.0 / 3.0 * n2 + 37.0 / 96.0 * n3 - 1.0 / 360.0 * n4 - 81.0 / 512.0 * n5
                + 96199.0 / 604800.0 * n6,
            // β₂
            1.0 / 48.0 * n2 + 1.0 / 15.0 * n3 - 437.0 / 1440.0 * n4 + 46.0 / 105.0 * n5
                - 1118711.0 / 3870720.0 * n6,
            // β₃
            17.0 / 480.0 * n3 - 37.0 / 840.0 * n4 - 209.0 / 4480.0 * n5 + 5569.0 / 90720.0 * n6,
            // β₄
            4397.0 / 161280.0 * n4 - 11.0 / 504.0 * n5 - 830251.0 / 7257600.0 * n6,
            // β₅
            4583.0 / 161280.0 * n5 - 108847.0 / 3991680.0 * n6,
            // β₆
            20648693.0 / 638668800.0 * n6,
        ]
    }

    /// Normalized meridional arc distance (ξ₀).
    fn meridional_arc_normalized(phi: f64, n: f64) -> f64 {
        let n2 = n * n;
        let n3 = n2 * n;
        let n4 = n3 * n;

        let a2 = -3.0 / 2.0 * n + 9.0 / 16.0 * n3;
        let a4 = 15.0 / 16.0 * n2 - 15.0 / 32.0 * n4;
        let a6 = -35.0 / 48.0 * n3;
        let a8 = 315.0 / 512.0 * n4;

        phi + a2 * (2.0 * phi).sin()
            + a4 * (4.0 * phi).sin()
            + a6 * (6.0 * phi).sin()
            + a8 * (8.0 * phi).sin()
    }

    /// Convert geodetic tangent τ to conformal tangent τ'.
    fn tau_to_tau_prime(&self, tau: f64) -> f64 {
        let e = self.ellipsoid.eccentricity();
        let tau1 = (1.0 + tau * tau).sqrt(); // = sec(φ) = hypot(1, τ)
        let sigma = (e * (e * tau / tau1).atanh()).sinh();
        tau * (1.0 + sigma * sigma).sqrt() - sigma * tau1
    }

    /// Convert conformal tangent τ' back to geodetic tangent τ via Newton iteration.
    fn tau_prime_to_tau(&self, tau_prime: f64) -> f64 {
        let e = self.ellipsoid.eccentricity();
        let e2 = self.ellipsoid.e2;
        let mut tau = tau_prime; // initial guess

        for _ in 0..15 {
            let tau1 = (1.0 + tau * tau).sqrt();
            let sigma = (e * (e * tau / tau1).atanh()).sinh();
            let tau_prime_est = tau * (1.0 + sigma * sigma).sqrt() - sigma * tau1;
            let dtau = (tau_prime - tau_prime_est) * (1.0 + (1.0 - e2) * tau * tau)
                / ((1.0 - e2) * tau1 * (1.0 + tau_prime_est * tau_prime_est).sqrt());
            tau += dtau;
            if dtau.abs() < 1e-12 * (1.0 + tau.abs()) {
                break;
            }
        }
        tau
    }
}

impl Projection for TransverseMercator {
    fn forward(&self, lon: f64, lat: f64) -> Result<(f64, f64), ProjError> {
        let dlam = lon - self.lon0;

        // Convert geodetic tangent to conformal tangent
        let tau = lat.tan();
        let tau_prime = self.tau_to_tau_prime(tau);

        // ξ' = atan2(τ', cos(Δλ))
        let xi_prime = tau_prime.atan2(dlam.cos());
        // η' = asinh(sin(Δλ) / hypot(τ', cos(Δλ)))
        let eta_prime =
            (dlam.sin() / (tau_prime * tau_prime + dlam.cos() * dlam.cos()).sqrt()).asinh();

        // Apply α series (forward)
        let mut xi = xi_prime;
        let mut eta = eta_prime;
        for (j, &a) in self.alpha.iter().enumerate() {
            let k = 2.0 * (j as f64 + 1.0);
            xi += a * (k * xi_prime).sin() * (k * eta_prime).cosh();
            eta += a * (k * xi_prime).cos() * (k * eta_prime).sinh();
        }

        let x = self.k0 * self.a_hat * eta + self.false_easting;
        let y = self.k0 * self.a_hat * (xi - self.m0) + self.false_northing;

        Ok((x, y))
    }

    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), ProjError> {
        let eta = (x - self.false_easting) / (self.k0 * self.a_hat);
        let xi = (y - self.false_northing) / (self.k0 * self.a_hat) + self.m0;

        // Apply β series (inverse)
        let mut xi_prime = xi;
        let mut eta_prime = eta;
        for (j, &b) in self.beta.iter().enumerate() {
            let k = 2.0 * (j as f64 + 1.0);
            xi_prime -= b * (k * xi).sin() * (k * eta).cosh();
            eta_prime -= b * (k * xi).cos() * (k * eta).sinh();
        }

        // τ' = sin(ξ') / hypot(sinh(η'), cos(ξ'))
        let sinh_eta = eta_prime.sinh();
        let cos_xi = xi_prime.cos();
        let sin_xi = xi_prime.sin();
        let tau_prime = sin_xi / (sinh_eta * sinh_eta + cos_xi * cos_xi).sqrt();

        // Recover geodetic tangent τ from conformal tangent τ'
        let tau = self.tau_prime_to_tau(tau_prime);

        let lat = tau.atan();
        let lon = self.lon0 + sinh_eta.atan2(cos_xi);

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

    #[test]
    fn test_roundtrip_utm33() {
        let tm = TransverseMercator::utm_zone(33, true);
        let cases: &[(f64, f64)] = &[
            (15.0, 52.0), // Berlin area (central meridian)
            (12.0, 50.0), // near zone boundary
            (18.0, 50.0), // near other boundary
            (15.0, 0.0),  // equator
            (15.0, 80.0), // high latitude
            (13.5, 52.5), // off-center
        ];
        for &(lon_deg, lat_deg) in cases {
            let lon = lon_deg.to_radians();
            let lat = lat_deg.to_radians();
            let (x, y) = tm.forward(lon, lat).unwrap();
            let (lon2, lat2) = tm.inverse(x, y).unwrap();
            assert_relative_eq!(lon2, lon, epsilon = 1e-9);
            assert_relative_eq!(lat2, lat, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_roundtrip_utm17() {
        let tm = TransverseMercator::utm_zone(17, true);
        let lon = (-74.0_f64).to_radians();
        let lat = 40.7_f64.to_radians();
        let (x, y) = tm.forward(lon, lat).unwrap();
        let (lon2, lat2) = tm.inverse(x, y).unwrap();
        assert_relative_eq!(lon2, lon, epsilon = 1e-9);
        assert_relative_eq!(lat2, lat, epsilon = 1e-9);
    }

    #[test]
    fn test_central_meridian_easting() {
        let tm = TransverseMercator::utm_zone(33, true);
        let (e, _) = tm
            .forward(15.0_f64.to_radians(), 45.0_f64.to_radians())
            .unwrap();
        assert_relative_eq!(e, 500_000.0, epsilon = 0.01);
    }

    #[test]
    fn test_utm_zone33n_known_point() {
        // (15°E, 52°N) → UTM Zone 33N
        // On central meridian → easting = 500000
        // Verify northing against proj4rs
        let tm = TransverseMercator::utm_zone(33, true);
        let lon = 15.0_f64.to_radians();
        let lat = 52.0_f64.to_radians();
        let (e, n) = tm.forward(lon, lat).unwrap();
        assert_relative_eq!(e, 500_000.0, epsilon = 1.0);
        // Northing should be approximately 5.76M
        assert!(n > 5_760_000.0 && n < 5_762_000.0, "northing = {n}");
    }

    #[test]
    fn test_utm_zone_central_meridian() {
        let tm1 = TransverseMercator::utm_zone(1, true);
        let tm33 = TransverseMercator::utm_zone(33, true);
        let tm60 = TransverseMercator::utm_zone(60, true);

        assert_relative_eq!(tm1.lon0, (-177.0_f64).to_radians(), epsilon = 1e-10);
        assert_relative_eq!(tm33.lon0, 15.0_f64.to_radians(), epsilon = 1e-10);
        assert_relative_eq!(tm60.lon0, 177.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn test_southern_hemisphere() {
        let tm = TransverseMercator::utm_zone(33, false);
        let lon = 15.0_f64.to_radians();
        let lat = (-30.0_f64).to_radians();
        let (x, y) = tm.forward(lon, lat).unwrap();
        assert!(y > 0.0, "Southing should be positive with FN=10M, got {y}");
        let (lon2, lat2) = tm.inverse(x, y).unwrap();
        assert_relative_eq!(lon2, lon, epsilon = 1e-9);
        assert_relative_eq!(lat2, lat, epsilon = 1e-9);
    }

    #[test]
    fn test_oslo_reference() {
        let tm = TransverseMercator::utm_zone(32, true);
        let lon = 10.75_f64.to_radians();
        let lat = 59.91_f64.to_radians();
        let (e, n) = tm.forward(lon, lat).unwrap();
        assert!(e > 200_000.0 && e < 800_000.0, "easting = {e}");
        assert!(n > 6_000_000.0 && n < 7_000_000.0, "northing = {n}");
        let (lon2, lat2) = tm.inverse(e, n).unwrap();
        assert_relative_eq!(lon2, lon, epsilon = 1e-9);
        assert_relative_eq!(lat2, lat, epsilon = 1e-9);
    }

    #[test]
    fn test_multiple_zones() {
        for zone in [1, 10, 17, 30, 33, 45, 60] {
            let tm = TransverseMercator::utm_zone(zone, true);
            let cm_deg = (zone as f64 - 1.0) * 6.0 - 180.0 + 3.0;
            // Test point 2° off center
            let lon = (cm_deg + 2.0).to_radians();
            let lat = 45.0_f64.to_radians();
            let (x, y) = tm.forward(lon, lat).unwrap();
            let (lon2, lat2) = tm.inverse(x, y).unwrap();
            assert_relative_eq!(lon2, lon, epsilon = 1e-9);
            assert_relative_eq!(lat2, lat, epsilon = 1e-9);
        }
    }
}
