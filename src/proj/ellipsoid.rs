/// Reference ellipsoid parameters.
#[derive(Clone, Copy, Debug)]
pub struct Ellipsoid {
    /// Semi-major axis (metres)
    pub a: f64,
    /// Flattening (dimensionless)
    pub f: f64,
    /// Semi-minor axis: a * (1 - f)
    pub b: f64,
    /// First eccentricity: sqrt(2f - f^2)
    pub e: f64,
    /// First eccentricity squared
    pub e2: f64,
    /// Second eccentricity squared: e^2 / (1 - e^2)
    pub ep2: f64,
    /// Third flattening: f / (2 - f)
    pub n: f64,
}

impl Ellipsoid {
    pub const fn new(a: f64, f: f64) -> Self {
        let b = a * (1.0 - f);
        let e2 = 2.0 * f - f * f;
        // Can't use .sqrt() in const fn, so we store e2 and compute e at runtime
        // For const contexts, e is set to 0 and must be accessed via e2.sqrt()
        let ep2 = e2 / (1.0 - e2);
        let n = f / (2.0 - f);
        Self {
            a,
            f,
            b,
            e: 0.0, // Use e2.sqrt() at runtime
            e2,
            ep2,
            n,
        }
    }

    /// Get the first eccentricity (computed at runtime).
    pub fn eccentricity(&self) -> f64 {
        self.e2.sqrt()
    }
}

pub const WGS84: Ellipsoid = Ellipsoid::new(6_378_137.0, 1.0 / 298.257_223_563);
pub const GRS80: Ellipsoid = Ellipsoid::new(6_378_137.0, 1.0 / 298.257_222_101);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_wgs84_constants() {
        assert_relative_eq!(WGS84.a, 6_378_137.0);
        assert_relative_eq!(WGS84.b, 6_356_752.314_245_179, epsilon = 0.001);
        assert_relative_eq!(WGS84.eccentricity(), 0.081_819_190_842_622, epsilon = 1e-12);
        assert_relative_eq!(WGS84.n, 0.001_679_220_386_383_705, epsilon = 1e-12);
    }

    #[test]
    fn test_grs80_close_to_wgs84() {
        // WGS84 and GRS80 differ only slightly
        assert_relative_eq!(WGS84.a, GRS80.a);
        assert!((WGS84.f - GRS80.f).abs() < 1e-8);
    }
}
