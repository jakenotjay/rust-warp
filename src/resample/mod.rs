//! Resampling kernels for the warp engine.

pub mod average;
pub mod bilinear;
pub mod cubic;
pub mod lanczos;
pub mod nearest;

use num_traits::NumCast;

/// Available resampling methods.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResamplingMethod {
    Nearest,
    Bilinear,
    Cubic,
    Lanczos,
    Average,
}

impl ResamplingMethod {
    /// Parse from a string name.
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "nearest" => Some(Self::Nearest),
            "bilinear" => Some(Self::Bilinear),
            "cubic" => Some(Self::Cubic),
            "lanczos" => Some(Self::Lanczos),
            "average" => Some(Self::Average),
            _ => None,
        }
    }

    /// Kernel radius in pixels (how far from center the kernel reaches).
    pub fn kernel_radius(&self) -> f64 {
        match self {
            Self::Nearest => 0.5,
            Self::Bilinear => 1.0,
            Self::Cubic => 2.0,
            Self::Lanczos => 3.0,
            Self::Average => 1.0,
        }
    }
}

/// Check whether a value should be treated as nodata.
///
/// Returns `true` if:
/// - `val` equals `nodata` sentinel (exact equality), or
/// - `val` is NaN (for float types, detected via NumCast to f64).
pub fn is_nodata_value<T: Copy + NumCast + PartialEq>(val: T, nodata: Option<T>) -> bool {
    if let Some(nd) = nodata {
        if val == nd {
            return true;
        }
    }
    // Check NaN via f64 conversion
    if let Some(f) = <f64 as NumCast>::from(val) {
        if f.is_nan() {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_nodata_sentinel() {
        assert!(is_nodata_value(-9999.0_f64, Some(-9999.0)));
        assert!(!is_nodata_value(42.0_f64, Some(-9999.0)));
        assert!(!is_nodata_value(42.0_f64, None));
    }

    #[test]
    fn test_is_nodata_nan() {
        assert!(is_nodata_value(f64::NAN, None));
        assert!(is_nodata_value(f64::NAN, Some(-9999.0)));
        assert!(is_nodata_value(f32::NAN, None));
    }

    #[test]
    fn test_is_nodata_integer() {
        assert!(is_nodata_value(0_i32, Some(0)));
        assert!(!is_nodata_value(1_i32, Some(0)));
        assert!(!is_nodata_value(42_i32, None));
    }

    #[test]
    fn test_is_nodata_normal_float() {
        assert!(!is_nodata_value(1.0_f64, None));
        assert!(!is_nodata_value(0.0_f64, None));
        assert!(!is_nodata_value(-1.0_f64, None));
    }
}
