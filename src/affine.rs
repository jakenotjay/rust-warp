/// A 2D affine transform representing a geotransform.
///
/// Maps pixel coordinates (col, row) to projected coordinates (x, y):
///   x = a * col + b * row + c
///   y = d * col + e * row + f
///
/// In GDAL convention: [c, a, b, f, d, e]
/// We store as: [a, b, c, d, e, f]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl Affine {
    pub fn new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        Self { a, b, c, d, e, f }
    }

    /// Create from a GDAL-style geotransform array [c, a, b, f, d, e].
    pub fn from_gdal(gt: &[f64; 6]) -> Self {
        Self {
            a: gt[1],
            b: gt[2],
            c: gt[0],
            d: gt[4],
            e: gt[5],
            f: gt[3],
        }
    }

    /// Convert to GDAL-style geotransform array [c, a, b, f, d, e].
    pub fn to_gdal(&self) -> [f64; 6] {
        [self.c, self.a, self.b, self.f, self.d, self.e]
    }

    /// Apply the forward transform: (col, row) -> (x, y).
    pub fn forward(&self, col: f64, row: f64) -> (f64, f64) {
        let x = self.a * col + self.b * row + self.c;
        let y = self.d * col + self.e * row + self.f;
        (x, y)
    }

    /// Compute the inverse affine transform.
    pub fn inverse(&self) -> Result<Affine, crate::error::WarpError> {
        let det = self.a * self.e - self.b * self.d;
        if det.abs() < f64::EPSILON {
            return Err(crate::error::WarpError::Affine(
                "Singular affine transform (determinant is zero)".into(),
            ));
        }
        let inv_det = 1.0 / det;
        Ok(Affine {
            a: self.e * inv_det,
            b: -self.b * inv_det,
            c: (self.b * self.f - self.e * self.c) * inv_det,
            d: -self.d * inv_det,
            e: self.a * inv_det,
            f: (self.d * self.c - self.a * self.f) * inv_det,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_forward_identity() {
        let aff = Affine::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let (x, y) = aff.forward(5.0, 10.0);
        assert_relative_eq!(x, 5.0);
        assert_relative_eq!(y, 10.0);
    }

    #[test]
    fn test_forward_with_offset_and_scale() {
        // 10m resolution, top-left at (500000, 6000000), north-up
        let aff = Affine::new(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0);
        let (x, y) = aff.forward(0.0, 0.0);
        assert_relative_eq!(x, 500000.0);
        assert_relative_eq!(y, 6000000.0);

        let (x, y) = aff.forward(100.0, 100.0);
        assert_relative_eq!(x, 501000.0);
        assert_relative_eq!(y, 5999000.0);
    }

    #[test]
    fn test_inverse_roundtrip() {
        let aff = Affine::new(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0);
        let inv = aff.inverse().unwrap();
        let (col, row) = inv.forward(501000.0, 5999000.0);
        assert_relative_eq!(col, 100.0, epsilon = 1e-10);
        assert_relative_eq!(row, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_singular_affine() {
        let aff = Affine::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(aff.inverse().is_err());
    }

    #[test]
    fn test_gdal_roundtrip() {
        let gt = [500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0];
        let aff = Affine::from_gdal(&gt);
        let gt2 = aff.to_gdal();
        for (a, b) in gt.iter().zip(gt2.iter()) {
            assert_relative_eq!(a, b);
        }
    }
}
