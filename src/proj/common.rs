//! Common helpers for projection math (meridional arc, latitude conversions, etc.).

use super::ellipsoid::Ellipsoid;

/// Compute the meridional arc length from the equator to latitude phi.
/// Uses the series expansion in powers of n (third flattening).
pub fn meridional_arc(ellipsoid: &Ellipsoid, phi: f64) -> f64 {
    let n = ellipsoid.n;
    let n2 = n * n;
    let n3 = n2 * n;
    let n4 = n3 * n;

    let a = ellipsoid.a / (1.0 + n) * (1.0 + n2 / 4.0 + n4 / 64.0);

    let a0 = 1.0;
    let a2 = -3.0 / 2.0 * n + 9.0 / 16.0 * n3;
    let a4 = 15.0 / 16.0 * n2 - 15.0 / 32.0 * n4;
    let a6 = -35.0 / 48.0 * n3;
    let a8 = 315.0 / 512.0 * n4;

    a * (a0 * phi
        + a2 * (2.0 * phi).sin()
        + a4 * (4.0 * phi).sin()
        + a6 * (6.0 * phi).sin()
        + a8 * (8.0 * phi).sin())
}

/// Isometric latitude factor: tan(π/4 - φ/2) / ((1 - e·sin(φ))/(1 + e·sin(φ)))^(e/2)
///
/// Used by Mercator, Lambert Conformal, and Stereographic projections.
/// `e` is the first eccentricity.
pub fn tsfn(phi: f64, e: f64) -> f64 {
    let sin_phi = phi.sin();
    let esinphi = e * sin_phi;
    let half_e = e / 2.0;
    (std::f64::consts::FRAC_PI_4 - phi / 2.0).tan()
        / ((1.0 - esinphi) / (1.0 + esinphi)).powf(half_e)
}

/// Meridional radius factor: cos(φ) / sqrt(1 - e²·sin²(φ))
///
/// Used by Lambert Conformal and Albers Equal Area projections.
pub fn msfn(phi: f64, e2: f64) -> f64 {
    let sin_phi = phi.sin();
    phi.cos() / (1.0 - e2 * sin_phi * sin_phi).sqrt()
}

/// Inverse of `tsfn`: recover latitude from isometric latitude factor `ts`.
///
/// Iterative solution using Halley-accelerated convergence.
/// `e` is the first eccentricity.
pub fn phi_from_ts(ts: f64, e: f64) -> f64 {
    let half_pi = std::f64::consts::FRAC_PI_2;
    let mut phi = half_pi - 2.0 * ts.atan();

    for _ in 0..15 {
        let esinphi = e * phi.sin();
        let phi_new =
            half_pi - 2.0 * (ts * ((1.0 - esinphi) / (1.0 + esinphi)).powf(e / 2.0)).atan();
        if (phi_new - phi).abs() < 1e-15 {
            return phi_new;
        }
        phi = phi_new;
    }
    phi
}

/// Authalic latitude helper `qsfn`:
/// (1-e²)[sin(φ)/(1-e²sin²(φ)) - (1/(2e))·ln((1-e·sinφ)/(1+e·sinφ))]
///
/// Used by Albers Equal Area projection.
pub fn qsfn(phi: f64, e: f64) -> f64 {
    let sin_phi = phi.sin();
    let e2 = e * e;
    let esinphi = e * sin_phi;
    let one_minus_e2sin2 = 1.0 - esinphi * esinphi;

    (1.0 - e2)
        * (sin_phi / one_minus_e2sin2
            - (1.0 / (2.0 * e)) * ((1.0 - esinphi) / (1.0 + esinphi)).ln())
}

/// Authalic latitude: converts geodetic latitude to authalic (equal-area) latitude.
///
/// `phi` is geodetic latitude in radians, `e` is first eccentricity.
pub fn authalic_latitude(phi: f64, e: f64) -> f64 {
    let q = qsfn(phi, e);
    let qp = qsfn(std::f64::consts::FRAC_PI_2, e);
    (q / qp).asin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::ellipsoid::WGS84;
    use approx::assert_relative_eq;

    #[test]
    fn test_meridional_arc_equator() {
        let m = meridional_arc(&WGS84, 0.0);
        assert_relative_eq!(m, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_meridional_arc_positive() {
        let m = meridional_arc(&WGS84, std::f64::consts::FRAC_PI_4);
        // Arc to 45 degrees should be ~4984944m (approx)
        assert!(m > 4_900_000.0 && m < 5_100_000.0);
    }

    #[test]
    fn test_tsfn_equator() {
        // At φ=0, tsfn should equal tan(π/4) = 1.0
        let e = WGS84.eccentricity();
        let ts = tsfn(0.0, e);
        assert_relative_eq!(ts, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_tsfn_45deg() {
        let e = WGS84.eccentricity();
        let phi = std::f64::consts::FRAC_PI_4;
        let ts = tsfn(phi, e);
        // ts at 45° should be between 0 and 1
        assert!(ts > 0.0 && ts < 1.0, "tsfn(45°) = {ts}");
    }

    #[test]
    fn test_phi_from_ts_roundtrip() {
        let e = WGS84.eccentricity();
        for &deg in &[0.0_f64, 30.0, 45.0, 60.0, 75.0, -30.0, -60.0] {
            let phi = deg.to_radians();
            let ts = tsfn(phi, e);
            let phi_back = phi_from_ts(ts, e);
            assert_relative_eq!(phi_back, phi, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_msfn_equator() {
        // At equator: cos(0)/sqrt(1-0) = 1.0
        let m = msfn(0.0, WGS84.e2);
        assert_relative_eq!(m, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_msfn_45deg() {
        let phi = std::f64::consts::FRAC_PI_4;
        let m = msfn(phi, WGS84.e2);
        // Should be close to cos(45°)/sqrt(1 - e²*sin²(45°))
        let expected = phi.cos() / (1.0 - WGS84.e2 * phi.sin().powi(2)).sqrt();
        assert_relative_eq!(m, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_qsfn_equator() {
        let e = WGS84.eccentricity();
        let q = qsfn(0.0, e);
        assert_relative_eq!(q, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_qsfn_pole() {
        let e = WGS84.eccentricity();
        let q = qsfn(std::f64::consts::FRAC_PI_2, e);
        // qp should be close to 2*(1 - (1-e²)/(2e)*ln((1-e)/(1+e)))
        assert!(q > 1.99 && q < 2.01, "qsfn(90°) = {q}");
    }

    #[test]
    fn test_authalic_latitude_equator() {
        let e = WGS84.eccentricity();
        let beta = authalic_latitude(0.0, e);
        assert_relative_eq!(beta, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_authalic_latitude_pole() {
        let e = WGS84.eccentricity();
        let beta = authalic_latitude(std::f64::consts::FRAC_PI_2, e);
        assert_relative_eq!(beta, std::f64::consts::FRAC_PI_2, epsilon = 1e-12);
    }
}
