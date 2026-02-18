//! Pipeline — CRS-to-CRS transform chain that dispatches between
//! native pure-Rust projections and proj4rs fallback.

use crate::error::ProjError;
use crate::proj::crs::CrsTransform;
use crate::proj::mercator::WebMercator;
use crate::proj::transverse_mercator::TransverseMercator;
use crate::proj::Projection;

/// Describes a CRS endpoint in the pipeline.
enum CrsEndpoint {
    /// Geographic CRS — coordinates are in degrees externally, radians internally.
    /// No forward/inverse needed — identity pass-through on (lon_rad, lat_rad).
    Geographic,
    /// Projected CRS — coordinates are in meters.
    Projected(Box<dyn Projection>),
}

/// A CRS-to-CRS transform pipeline.
///
/// For supported EPSG codes, uses native pure-Rust projection math.
/// Falls back to proj4rs for anything else.
pub enum Pipeline {
    /// Both src and dst are recognized native projections.
    #[allow(private_interfaces)]
    Native { src: CrsEndpoint, dst: CrsEndpoint },
    /// Fallback to proj4rs for unsupported CRSes.
    Proj4rs(Box<CrsTransform>),
}

impl Pipeline {
    /// Create a Pipeline from source and destination CRS strings.
    ///
    /// Tries to parse both as EPSG codes and create native projections.
    /// Falls back to proj4rs if either is unrecognized.
    pub fn new(src_crs: &str, dst_crs: &str) -> Result<Self, ProjError> {
        let src = parse_epsg(src_crs);
        let dst = parse_epsg(dst_crs);

        if let (Some(src_ep), Some(dst_ep)) = (src, dst) {
            Ok(Pipeline::Native {
                src: src_ep,
                dst: dst_ep,
            })
        } else {
            let ct = CrsTransform::new(src_crs, dst_crs)?;
            Ok(Pipeline::Proj4rs(Box::new(ct)))
        }
    }

    /// Transform a single point from destination CRS to source CRS.
    ///
    /// This is the direction the warp engine needs: given an output pixel
    /// coordinate in dst CRS, find the corresponding source CRS coordinate.
    ///
    /// Input/output coordinates are in CRS native units (degrees for geographic,
    /// metres for projected).
    pub fn transform_inv(&self, x: f64, y: f64) -> Result<(f64, f64), ProjError> {
        match self {
            Pipeline::Native { src, dst } => {
                // Step 1: dst coords → (lon_rad, lat_rad)
                let (lon, lat) = match dst {
                    CrsEndpoint::Geographic => (x.to_radians(), y.to_radians()),
                    CrsEndpoint::Projected(proj) => proj.inverse(x, y)?,
                };

                // Step 2: (lon_rad, lat_rad) → src coords
                match src {
                    CrsEndpoint::Geographic => Ok((lon.to_degrees(), lat.to_degrees())),
                    CrsEndpoint::Projected(proj) => proj.forward(lon, lat),
                }
            }
            Pipeline::Proj4rs(ct) => ct.transform_inv(x, y),
        }
    }

    /// Batch transform from destination CRS to source CRS.
    ///
    /// Coordinates are modified in-place. All coordinates use CRS native units.
    pub fn transform_inv_batch(&self, coords: &mut [(f64, f64)]) -> Result<(), ProjError> {
        match self {
            Pipeline::Native { src, dst } => {
                for c in coords.iter_mut() {
                    let (lon, lat) = match dst {
                        CrsEndpoint::Geographic => (c.0.to_radians(), c.1.to_radians()),
                        CrsEndpoint::Projected(proj) => proj.inverse(c.0, c.1)?,
                    };

                    *c = match src {
                        CrsEndpoint::Geographic => (lon.to_degrees(), lat.to_degrees()),
                        CrsEndpoint::Projected(proj) => proj.forward(lon, lat)?,
                    };
                }
                Ok(())
            }
            Pipeline::Proj4rs(ct) => ct.transform_inv_batch(coords),
        }
    }
}

/// Try to parse an EPSG code and return a `CrsEndpoint`.
fn parse_epsg(crs: &str) -> Option<CrsEndpoint> {
    let code = crs
        .strip_prefix("EPSG:")
        .or_else(|| crs.strip_prefix("epsg:"))?
        .parse::<u32>()
        .ok()?;

    match code {
        // Geographic CRS — identity pass-through (degrees ↔ radians handled by Pipeline)
        4326 => Some(CrsEndpoint::Geographic),

        // Web Mercator
        3857 => Some(CrsEndpoint::Projected(Box::new(WebMercator::new()))),

        // UTM North: EPSG:326XX (zone 1–60)
        32601..=32660 => {
            let zone = (code - 32600) as u8;
            Some(CrsEndpoint::Projected(Box::new(
                TransverseMercator::utm_zone(zone, true),
            )))
        }

        // UTM South: EPSG:327XX (zone 1–60)
        32701..=32760 => {
            let zone = (code - 32700) as u8;
            Some(CrsEndpoint::Projected(Box::new(
                TransverseMercator::utm_zone(zone, false),
            )))
        }

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_native_utm33_to_4326() {
        let pipe = Pipeline::new("EPSG:4326", "EPSG:32633").unwrap();
        // Input: UTM 33N coords → output: lon/lat in degrees
        let (lon, lat) = pipe.transform_inv(500000.0, 5760000.0).unwrap();
        assert!(lon > 14.0 && lon < 16.0, "lon = {lon}");
        assert!(lat > 51.0 && lat < 53.0, "lat = {lat}");
    }

    #[test]
    fn test_native_4326_to_utm33() {
        let pipe = Pipeline::new("EPSG:32633", "EPSG:4326").unwrap();
        // Input: lon/lat in degrees → output: UTM 33N coords
        let (e, n) = pipe.transform_inv(15.0, 52.0).unwrap();
        assert_relative_eq!(e, 500_000.0, epsilon = 1.0);
        assert!(n > 5_760_000.0 && n < 5_762_000.0, "northing = {n}");
    }

    #[test]
    fn test_native_matches_proj4rs() {
        // Verify the native pipeline produces the same results as proj4rs
        let native = Pipeline::new("EPSG:4326", "EPSG:32633").unwrap();
        let fallback = {
            let ct = CrsTransform::new("EPSG:4326", "EPSG:32633").unwrap();
            Pipeline::Proj4rs(Box::new(ct))
        };

        let test_points: &[(f64, f64)] = &[
            (500000.0, 5760000.0),
            (400000.0, 6000000.0),
            (600000.0, 5500000.0),
        ];

        for &(x, y) in test_points {
            let (lon_n, lat_n) = native.transform_inv(x, y).unwrap();
            let (lon_p, lat_p) = fallback.transform_inv(x, y).unwrap();
            // Should match within 1mm (< 0.001 degrees ~ 110m, so use tighter bound)
            assert_relative_eq!(lon_n, lon_p, epsilon = 1e-6);
            assert_relative_eq!(lat_n, lat_p, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_native_web_mercator() {
        let pipe = Pipeline::new("EPSG:4326", "EPSG:3857").unwrap();
        // (0, 0) in Web Mercator → (0°, 0°)
        let (lon, lat) = pipe.transform_inv(0.0, 0.0).unwrap();
        assert_relative_eq!(lon, 0.0, epsilon = 1e-10);
        assert_relative_eq!(lat, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_native_utm_to_webmerc() {
        let pipe = Pipeline::new("EPSG:3857", "EPSG:32633").unwrap();
        let (x, y) = pipe.transform_inv(500000.0, 5760000.0).unwrap();
        // Should produce valid Web Mercator coordinates
        assert!(x.abs() < 20_037_509.0, "x = {x}");
        assert!(y.abs() < 20_037_509.0, "y = {y}");
    }

    #[test]
    fn test_fallback_to_proj4rs() {
        // Use a CRS that's not natively supported → should fallback to proj4rs
        let pipe = Pipeline::new("EPSG:4326", "EPSG:2154").unwrap();
        assert!(matches!(pipe, Pipeline::Proj4rs(_)));
    }

    #[test]
    fn test_batch_transform() {
        let pipe = Pipeline::new("EPSG:4326", "EPSG:32633").unwrap();
        let mut coords = vec![(500000.0, 5760000.0), (510000.0, 5770000.0)];
        pipe.transform_inv_batch(&mut coords).unwrap();
        for (lon, lat) in &coords {
            assert!(*lon > 14.0 && *lon < 16.0, "lon = {lon}");
            assert!(*lat > 51.0 && *lat < 53.0, "lat = {lat}");
        }
    }

    #[test]
    fn test_identity_same_crs() {
        let pipe = Pipeline::new("EPSG:32633", "EPSG:32633").unwrap();
        let (x, y) = pipe.transform_inv(500000.0, 5760000.0).unwrap();
        assert_relative_eq!(x, 500000.0, epsilon = 0.01);
        assert_relative_eq!(y, 5760000.0, epsilon = 0.01);
    }
}
