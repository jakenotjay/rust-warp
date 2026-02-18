//! Resampling kernels for the warp engine.

pub mod bilinear;
pub mod nearest;

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
