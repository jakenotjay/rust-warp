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

    a * (a0 * phi + a2 * (2.0 * phi).sin() + a4 * (4.0 * phi).sin()
        + a6 * (6.0 * phi).sin()
        + a8 * (8.0 * phi).sin())
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
}
